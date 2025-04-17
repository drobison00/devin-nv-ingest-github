# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=broad-except

import json
import logging
import math
import time
from collections import defaultdict
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from concurrent.futures import wait
from concurrent.futures import FIRST_COMPLETED
from typing import Any, Type, Callable, Set
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from nv_ingest_api.util.service_clients.client_base import MessageBrokerClientBase
from nv_ingest_api.util.service_clients.rest.rest_client import RestClient
from nv_ingest_client.primitives import BatchJobSpec
from nv_ingest_client.primitives import JobSpec
from nv_ingest_client.primitives.jobs import JobState
from nv_ingest_client.primitives.jobs import JobStateEnum
from nv_ingest_client.primitives.tasks import Task
from nv_ingest_client.primitives.tasks import TaskType
from nv_ingest_client.primitives.tasks import is_valid_task_type
from nv_ingest_client.primitives.tasks import task_factory
from nv_ingest_client.util.util import create_job_specs_for_batch

logger = logging.getLogger(__name__)


class DataDecodeException(Exception):
    """
    Exception raised for errors in decoding data.

    Attributes:
        message -- explanation of the error
        data -- the data that failed to decode, optionally
    """

    def __init__(self, message="Data decoding error", data=None):
        self.message = message
        self.data = data
        super().__init__(f"{message}: {data}")

    def __str__(self):
        return f"{self.__class__.__name__}({self.message}, Data={self.data})"


class _ConcurrentProcessor:
    """
    Manages the concurrent submission and result fetching of jobs using a client's
    worker pool, handling concurrency limits, timeouts, retries, and callbacks.
    """

    def __init__(
        self,
        client: "NvIngestClient",
        job_indices: List[str],
        job_queue_id: Optional[str],
        concurrency_limit: int,
        timeout: Tuple[int, Optional[float]],
        max_job_retries: Optional[int],
        retry_delay: float,
        completion_callback: Optional[Callable[[Any, str], None]],
        fail_on_submit_error: bool,
        data_only: bool,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the concurrent processor.

        Parameters
        ----------
        client : NvIngestClient
            The ingest client used for job submission and fetching.
        job_indices : list of str
            List of job identifiers to process.
        job_queue_id : str or None
            Identifier for the job queue, required for submission.
        concurrency_limit : int
            Maximum number of jobs to have in flight at once.
        timeout : tuple of (int, float or None)
            Timeout settings for fetch attempts: (connect_timeout, read_timeout).
        max_job_retries : int or None
            Maximum number of fetch retries for 'not ready' jobs. None for infinite.
        retry_delay : float
            Delay in seconds between retry cycles.
        completion_callback : callable or None
            Function called upon each successful fetch, with signature
            (result_data, job_index).
        fail_on_submit_error : bool
            If True, abort processing on any submission error.
        data_only : bool
            If True, extract and return only the 'data' field from results.
        verbose : bool, default=False
            If True, enable detailed logging.
        """
        self.client = client
        self.job_queue_id = job_queue_id
        self.concurrency_limit = concurrency_limit
        self.timeout = timeout
        self.max_job_retries = max_job_retries
        self.retry_delay = retry_delay
        self.completion_callback = completion_callback
        self.fail_on_submit_error = fail_on_submit_error
        self.data_only = data_only
        self.verbose = verbose

        # --- State Variables ---
        self.all_job_indices: Set[str] = set(job_indices)
        self.job_indices_to_submit: List[str] = list(job_indices)  # Queue for initial submission
        self.retry_job_ids: List[str] = []  # Queue for jobs needing fetch retry
        self.active_futures: Dict[Future, str] = {}  # Future -> job_index
        self.retry_counts: Dict[str, int] = defaultdict(int)
        self.results: List[Any] = []
        self.failures: List[Tuple[str, str]] = []  # (job_index, error_message)
        self.processed_count: int = 0
        self.total_jobs: int = len(job_indices)
        self._needs_retry_delay: bool = False  # Flag to indicate if a retry delay is needed

    # --- Core Execution Method ---

    def run(self) -> Tuple[List[Any], List[Tuple[str, str]]]:
        """
        Executes the concurrent processing loop until all jobs are processed or
        an unrecoverable error occurs. Uses wait(FIRST_COMPLETED) to process results.

        Returns:
            A tuple containing two lists: (successful_results, failures).
        """
        if not hasattr(self.client, "_worker_pool") or not isinstance(self.client._worker_pool, ThreadPoolExecutor):
            raise AttributeError("Client object must have a '_worker_pool' attribute of type ThreadPoolExecutor")
        self.executor = self.client._worker_pool

        logger.info(
            f"Starting concurrent processing for {self.total_jobs} jobs "
            f"with concurrency limit {self.concurrency_limit}."
        )

        while self.processed_count < self.total_jobs:
            # --- Fill Concurrency Slots ---
            # Initiate async submission for new jobs & submit fetch tasks for new/retry.
            submitted_fetch_count = self._fill_concurrency_slots()
            if submitted_fetch_count > 0 and self.verbose:
                logger.debug(f"Submitted {submitted_fetch_count} new/retry job fetches.")

            # --- Wait for and Process Completed Fetch Futures ---
            if not self.active_futures:
                # If nothing is active, check if we should exit or just wait
                if not self.job_indices_to_submit and not self.retry_job_ids:
                    # Nothing running, nothing pending submission/retry - should be done.
                    if self.processed_count < self.total_jobs:
                        logger.warning(
                            "No active futures or pending jobs, but processed_count < total_jobs. Possible state issue."
                        )
                    if self.verbose:
                        logger.debug("No active futures or pending jobs. Exiting loop.")
                    break  # Exit main loop
                else:
                    # Nothing active, but jobs waiting for submission/retry (likely after a delay)
                    if self.verbose:
                        logger.debug("No active futures, but jobs pending submission/retry.")
                    # Avoid busy-waiting if only retries/submissions remain after delay
                    if not self._needs_retry_delay and self.retry_delay <= 0:
                        time.sleep(0.05)
                    continue  # Go back to top, apply delay if needed, try submitting again

            # Wait for at least one active fetch future to complete
            wait_timeout = max(self.retry_delay, 1.0) if self.retry_delay > 0 else 1.0
            # Ensure timeout is reasonable if delay is very small
            wait_timeout = max(wait_timeout, 0.1)

            try:
                # Use wait with FIRST_COMPLETED
                done, not_done = wait(
                    list(self.active_futures.keys()), timeout=wait_timeout, return_when=FIRST_COMPLETED
                )
            except Exception as wait_err:
                # Should be unlikely with standard usage, but handle defensively
                logger.error(f"Unexpected error during concurrent.futures.wait: {wait_err}", exc_info=True)
                # Potentially break or implement recovery strategy depending on severity
                time.sleep(1)  # Avoid tight loop on unexpected errors
                continue

            if not done:
                # Timeout expired before any future completed
                if self.verbose:
                    logger.debug(
                        f"Wait timeout ({wait_timeout}s) expired, " f"{len(self.active_futures)} fetches still active."
                    )
                # No results to process, loop will continue, potentially apply delay, submit more.
                continue

            # Process all futures that completed during the wait
            if self.verbose:
                logger.debug(f"{len(done)} fetch future(s) completed.")
            for future in done:
                # _process_one_completed_future handles result/exception and removes from active_futures
                self._process_one_completed_future(future)

            # Loop continues: apply delay -> fill slots -> wait again

        self._log_final_status()
        return self.results, self.failures

    # --- Helper Methods ---

    def _apply_retry_delay_if_needed(self):
        """Applies the retry delay if the flag is set."""
        if self._needs_retry_delay and self.retry_delay > 0:
            if self.verbose:
                logger.debug(f"Applying retry delay: {self.retry_delay}s before next submissions.")
            time.sleep(self.retry_delay)
            self._needs_retry_delay = False

    def _fill_concurrency_slots(self) -> int:
        """
        Initiates async submission for new jobs, validates submission state using
        client._ensure_submitted, and submits fetch tasks for valid jobs until
        concurrency limit is reached.

        Returns:
            The number of new fetch tasks successfully submitted to the executor.
        """
        submitted_fetch_count = 0
        available_slots = self.concurrency_limit - len(self.active_futures)

        if available_slots <= 0:
            return 0

        # --- Step 1: Initiate Asynchronous Submission for a Batch of New Jobs ---
        successfully_initiated_new: List[str] = []
        num_new_to_submit = min(available_slots, len(self.job_indices_to_submit))

        if num_new_to_submit > 0:
            new_batch_indices = self.job_indices_to_submit[:num_new_to_submit]
            if self.verbose:
                logger.debug(f"Attempting batch async submission for {len(new_batch_indices)} new jobs.")

            if not self.job_queue_id:
                logger.error("Cannot submit new jobs: job_queue_id is not set.")
                for job_index in new_batch_indices:
                    self._handle_processing_failure(job_index, "job_queue_id not set", is_submission_failure=True)
                    self.processed_count += 1
                self.job_indices_to_submit = self.job_indices_to_submit[num_new_to_submit:]
                if self.fail_on_submit_error:
                    raise ValueError("Cannot submit new jobs: job_queue_id is not set.")
                new_batch_indices = []  # Prevent fetch submission attempt later
            else:
                try:
                    # This call initiates background submission tasks
                    _ = self.client.submit_job_async(new_batch_indices, self.job_queue_id)
                    successfully_initiated_new = new_batch_indices
                    self.job_indices_to_submit = self.job_indices_to_submit[num_new_to_submit:]
                    if self.verbose:
                        logger.info(
                            f"Successfully initiated async submission for {len(successfully_initiated_new)} jobs."
                        )
                except Exception as e:
                    error_msg = f"Batch async submission initiation failed for {len(new_batch_indices)} jobs: {e}"
                    logger.error(error_msg, exc_info=True)
                    for job_index in new_batch_indices:
                        self._handle_processing_failure(
                            job_index, f"Batch submission initiation error: {e}", is_submission_failure=True
                        )
                        self.processed_count += 1
                    self.job_indices_to_submit = self.job_indices_to_submit[num_new_to_submit:]
                    successfully_initiated_new = []
                    if self.fail_on_submit_error:
                        raise RuntimeError(error_msg) from e

        # --- Step 2: Identify Retry Jobs to Fill Remaining Slots ---
        slots_after_new = available_slots - len(successfully_initiated_new)
        num_retries_to_submit = min(slots_after_new, len(self.retry_job_ids))

        retry_batch_indices: List[str] = []
        if num_retries_to_submit > 0:
            retry_batch_indices = self.retry_job_ids[:num_retries_to_submit]
            # Remove selected retries from the queue *before* validation
            self.retry_job_ids = self.retry_job_ids[num_retries_to_submit:]
            if self.verbose:
                logger.debug(f"Identified {len(retry_batch_indices)} retry jobs for validation.")

        # --- Step 3: Validate Submission State using _ensure_submitted ---
        # Validate *all* jobs we intend to fetch in this iteration (newly initiated + retries)
        indices_to_validate = successfully_initiated_new + retry_batch_indices
        validated_indices_to_fetch: List[str] = []

        if not indices_to_validate:
            # No new jobs were initiated and no retries were available/selected
            if self.verbose and available_slots > 0:
                logger.debug("No jobs available to validate/fetch this iteration.")
            return 0

        try:
            if self.verbose:
                logger.debug(
                    f"Validating submission state for {len(indices_to_validate)} "
                    f"jobs using client._ensure_submitted (BLOCKING)."
                )

            # This call BLOCKS until background submissions (for new jobs) are done
            # and checks state for all listed jobs.
            self.client._ensure_submitted(indices_to_validate)

            # If no exception, all jobs are considered valid for fetching
            validated_indices_to_fetch = indices_to_validate
            if self.verbose:
                logger.debug(f"Submission state validation passed for {len(validated_indices_to_fetch)} jobs.")

        except Exception as validation_err:
            # Validation failed for one or more jobs in the batch.
            error_msg = (
                f"Submission state validation failed (via _ensure_submitted) for batch "
                f"({len(indices_to_validate)} jobs): {validation_err}"
            )
            logger.error(error_msg, exc_info=True)  # Log full trace for validation errors

            # Assume the entire batch being validated failed validation. Mark all as failed.
            # (Refinement possible if exception precisely identifies failed jobs)
            logger.warning(
                f"Marking all {len(indices_to_validate)} jobs in the validated batch as failed due to validation error."
            )
            for job_index in indices_to_validate:
                # If a job was pulled from retry_job_ids, ensure it doesn't get added back implicitly
                self._handle_processing_failure(
                    job_index, f"Batch state validation failed: {validation_err}", is_submission_failure=True
                )
                self.processed_count += 1

            validated_indices_to_fetch = []  # Ensure none are submitted for fetch

            if self.fail_on_submit_error:
                raise RuntimeError(
                    f"Stopping due to batch state validation failure: {validation_err}"
                ) from validation_err

        # --- Step 4: Submit Fetch Tasks for Validated Jobs ---
        if not validated_indices_to_fetch:
            if self.verbose:
                logger.debug("No jobs passed state validation for fetch submission this iteration.")
            return 0

        if self.verbose:
            logger.debug(f"Submitting fetch tasks for {len(validated_indices_to_fetch)} validated jobs.")

        for job_index in validated_indices_to_fetch:
            # Ensure we don't exceed concurrency limit if validation took time and other futures finished
            if len(self.active_futures) < self.concurrency_limit:
                if self._submit_fetch_task(job_index):
                    submitted_fetch_count += 1
            else:
                # This shouldn't happen often if validation is reasonably fast, but handle defensively
                logger.warning(
                    f"Concurrency limit reached before submitting fetch task for validated job "
                    f"{job_index}. Will attempt next iteration."
                )
                # Put jobs that couldn't be submitted back into appropriate queues
                if job_index in successfully_initiated_new:
                    # It's a new job, it needs to be fetched eventually, but wasn't a retry yet.
                    # Add to a temporary list or directly to retry_job_ids? Let's add to retry_job_ids
                    # as it needs fetching later.
                    if job_index not in self.retry_job_ids:
                        self.retry_job_ids.insert(0, job_index)
                elif job_index in retry_batch_indices:
                    # It was already a retry job, put it back at the front of the retry queue
                    if job_index not in self.retry_job_ids:
                        self.retry_job_ids.insert(0, job_index)
                # We didn't actually submit the fetch, so don't increment submitted_fetch_count

        return submitted_fetch_count

    def _submit_fetch_task(self, job_index: str) -> bool:
        """
        Submits a fetch task for a single job index to the executor.

        Args:
            job_index: The job index to fetch.

        Returns:
            True if the fetch task was successfully submitted, False otherwise.
        """

        is_retry_attempt = job_index in self.retry_counts  # Check if it was previously timed out

        try:
            if self.verbose:
                log_prefix = (
                    f"Retrying fetch for job {job_index} (Attempt {self.retry_counts[job_index] + 1})"
                    if is_retry_attempt
                    else f"Submitting fetch task for job {job_index}"
                )
                logger.debug(log_prefix)

            future = self.executor.submit(
                self.client._fetch_job_result,
                job_index,
                timeout=self.timeout,
                data_only=self.data_only,
            )
            self.active_futures[future] = job_index
            return True  # Fetch task successfully submitted

        except Exception as e:
            # Handle error during the submission of the fetch task itself
            error_msg = f"Failed submitting fetch task for job {job_index}: {e}"
            logger.error(error_msg)
            # Treat as a terminal failure - it won't be tracked by a future
            self._handle_processing_failure(
                job_index, error_msg, is_submission_failure=True
            )  # Mark as submission-related failure
            self.processed_count += 1  # Increment count as no future will track this job
            if self.fail_on_submit_error:
                raise RuntimeError(f"Stopping due to error submitting fetch task for {job_index}: {e}") from e
            return False  # Fetch task not submitted

    def _initiate_processing(self, job_index: str, is_retry: bool) -> bool:
        """
        Initiates async submission if needed (for new jobs) and submits the fetch task.

        Args:
            job_index: The index of the job to process.
            is_retry: True if this is a retry fetch attempt.

        Returns:
            True if the fetch task was successfully submitted to the executor,
            False otherwise (e.g., submission initiation failed pre-emptively).

        Raises:
            RuntimeError: If `fail_on_submit_error` is True and an error occurs
                          during the call to `submit_job_async`.
        """

        submission_initiated = False
        try:
            # --- Step 1: Initiate Asynchronous Submission (if needed) ---
            if not is_retry:
                if not self.job_queue_id:
                    # This should ideally be checked earlier, but double-check here
                    raise ValueError("job_queue_id must be provided to submit new jobs.")

                if self.verbose:
                    logger.debug(f"Initiating async submission for new job {job_index} via submit_job_async.")

                _ = self.client.submit_job_async(job_index, self.job_queue_id)
                submission_initiated = True  # Assume success if no exception is raised

            # --- Step 2: Submit the Fetch Task ---
            if self.verbose and is_retry:
                logger.info(f"Retrying fetch for job {job_index} (Attempt {self.retry_counts[job_index] + 1})")
            elif self.verbose and not is_retry:
                logger.debug(f"Submitting fetch task for job {job_index} (submission initiated separately).")

            # Submit the task to fetch the result (_fetch_job_result should handle retries internally via exceptions)
            future = self.executor.submit(
                self.client._fetch_job_result,
                job_index,
                timeout=self.timeout,
                data_only=self.data_only,
            )
            self.active_futures[future] = job_index
            return True  # Fetch task successfully submitted

        except Exception as e:
            # This catches errors from:
            # 1. The synchronous part of client.submit_job_async() (e.g., bad initial state).
            # 2. The self.executor.submit() call for the fetch task itself.
            component = (
                "initiating async submission" if not is_retry and not submission_initiated else "submitting fetch task"
            )
            error_msg = f"Failed {component} for job {job_index}: {e}"
            logger.error(error_msg)

            # This failure means the fetch future was *not* created or added,
            # or the submission couldn't even be queued.
            # Treat as a terminal failure for this job_index.
            self._handle_processing_failure(job_index, error_msg, is_submission_failure=True)
            self.processed_count += 1  # Increment count as no future will track this job.

            if self.fail_on_submit_error:
                # If failure occurred during the submission initiation part specifically
                if not is_retry and not submission_initiated:
                    raise RuntimeError(f"Stopping due to error initiating submission for {job_index}: {e}") from e
                # Else, error was submitting fetch task, still raise if flag is set
                raise RuntimeError(f"Stopping due to error submitting fetch task for {job_index}: {e}") from e

            return False  # Fetch task not submitted

    def _process_one_completed_future(self, future: Future):
        """Processes a single future that has completed."""
        job_index = self.active_futures.pop(future)
        will_retry = False
        try:
            result_data, fetched_job_index, trace_id = future.result()
            if fetched_job_index != job_index:
                logger.warning(f"Mismatch: Future associated with {job_index} returned data for {fetched_job_index}")
            self._handle_processing_success(job_index, result_data, trace_id)

        except TimeoutError as e:  # Raised by _fetch_job_result for "Not Ready"
            will_retry = self._handle_fetch_timeout(job_index, e)

        except (ValueError, RuntimeError) as e:  # Raised by _fetch_job_result for terminal errors
            error_msg = f"Terminal error during fetch: {e}"
            logger.error(f"Job {job_index} failed: {error_msg}")
            self._handle_processing_failure(job_index, error_msg)

        except Exception as e:  # Other unexpected errors during future execution
            error_msg = f"Unexpected error processing fetch future: {e}"
            logger.exception(f"Job {job_index} failed: {error_msg}")
            self._handle_processing_failure(job_index, f"Unexpected future processing error: {e}")

        finally:
            if not will_retry:
                self.processed_count += 1
                if self.verbose:
                    logger.debug(
                        f"Job {job_index} finished processing (Success/Fail). "
                        f"Processed count: {self.processed_count}/{self.total_jobs}"
                    )

    def _submit_job_fetch(self, job_index: str, is_retry: bool) -> bool:
        """
        Submit a fetch task for a given job index, handling initial submission if needed.

        Parameters
        ----------
        job_index : str
            Identifier of the job to fetch.
        is_retry : bool
            True if this fetch is a retry attempt.

        Returns
        -------
        success : bool
            True if the fetch task was enqueued, False otherwise.
        """
        try:
            if not is_retry:
                # For new jobs, ensure they are submitted first
                job_state = self.client._get_and_check_job_state(job_index, required_state=[JobStateEnum.PENDING])
                if not job_state.job_id:  # Submit if not already submitted
                    self.client._submit_job(job_index, self.job_queue_id)
                    job_state.state = JobStateEnum.SUBMITTED  # Update local state assumption

            # Submit the fetch task (for both new and retries)
            if self.verbose and is_retry:
                logger.info(f"Retrying fetch for job {job_index} (Attempt {self.retry_counts[job_index] + 1})")

            future = self.executor.submit(
                self.client._fetch_job_result,
                job_index,
                timeout=self.timeout,
                data_only=self.data_only,
            )
            self.active_futures[future] = job_index
            return True

        except Exception as e:
            # Handle errors during initial check/submit or fetch submission
            error_msg = f"Failed to initiate fetch for job {job_index}: {e}"
            logger.error(error_msg)
            self._handle_processing_failure(job_index, error_msg, is_submission_failure=not is_retry)
            if self.fail_on_submit_error and not is_retry:
                raise RuntimeError(f"Stopping due to submission error for {job_index}: {e}") from e
            return False  # Fetch task not submitted

    # TODO
    def _wait_for_completion(self) -> Set[Future]:
        """
        Wait for at least one active fetch future to complete.

        Returns
        -------
        done : set of Future
            Completed futures. Empty if none were active or timed out.
        """
        if not self.active_futures:
            if self.verbose:
                logger.debug("No active futures to wait on.")
            return set()

        try:
            done, _ = wait(list(self.active_futures.keys()), return_when=FIRST_COMPLETED)

            return done
        except Exception as e:
            logger.error(f"Unexpected error during future wait: {e}", exc_info=True)
            return set()

    def _process_completed_futures(self, completed_futures: Set[Future]) -> None:
        """
        Process a set of completed fetch futures.

        Parameters
        ----------
        completed_futures : set of Future
            Futures whose results or exceptions must be handled.
        """
        for future in completed_futures:
            job_index = self.active_futures.pop(future)  # Remove processed future
            will_retry = False
            try:
                result_data, _, trace_id = future.result()
                self._handle_processing_success(job_index, result_data, trace_id)

            except TimeoutError as e:  # Raised by _fetch_job_result for "Not Ready"
                will_retry = self._handle_fetch_timeout(job_index, e)

            except (ValueError, RuntimeError) as e:  # Terminal error from _fetch_job_result
                error_msg = f"Terminal error during fetch: {e}"
                logger.error(f"Job {job_index} failed: {error_msg}")
                self._handle_processing_failure(job_index, error_msg)

            except Exception as e:  # Other unexpected errors during future processing
                error_msg = f"Unexpected error processing fetch future: {e}"
                logger.exception(f"Job {job_index} failed: {error_msg}")  # Log with stack trace
                self._handle_processing_failure(job_index, f"Unexpected future processing error: {e}")

            finally:
                if not will_retry:
                    self.processed_count += 1  # Increment only if job finished (success or permanent fail)

    def _handle_processing_success(self, job_index: str, result_data: Any, trace_id: Optional[str]) -> None:
        """
        Handle a successfully fetched job result.

        Parameters
        ----------
        job_index : str
            Identifier of the completed job.
        result_data : any
            Data returned by the fetch operation.
        trace_id : str or None
            Optional trace identifier from the fetch.
        """
        if self.verbose:
            logger.info(
                f"Successfully fetched result for job {job_index}" f"{' (Trace: ' + trace_id + ')' if trace_id else ''}"
            )
        self.results.append(result_data)
        # Reset retry count on success (though it shouldn't be needed)
        if job_index in self.retry_counts:
            del self.retry_counts[job_index]

        if self.completion_callback:
            try:
                self.completion_callback(result_data, job_index)
            except Exception as cb_err:
                logger.error(f"Error in completion_callback for job {job_index}: {cb_err}")

    def _handle_fetch_timeout(self, job_index: str, error: TimeoutError) -> bool:
        """
        Handle a fetch timeout (job not ready) and decide on retry.

        Parameters
        ----------
        job_index : str
            Identifier of the job that timed out.
        error : TimeoutError
            Exception instance indicating the timeout.

        Returns
        -------
        will_retry : bool
            True if the job has been queued for retry, False if it has failed permanently.
        """
        self.retry_counts[job_index] += 1
        if self.max_job_retries is None or self.retry_counts[job_index] <= self.max_job_retries:
            if self.verbose:
                logger.info(
                    f"Job {job_index} not ready (Attempt {self.retry_counts[job_index]}/"
                    f"{self.max_job_retries or 'inf'}). Will retry after delay."
                )
            self.retry_job_ids.append(job_index)  # Add back to retry queue
            self._needs_retry_delay = True  # Signal that a delay is needed
            return True  # Will retry
        else:
            error_msg = f"Exceeded max retries ({self.max_job_retries}) " f"waiting for readiness. Last reason: {error}"
            logger.error(f"Job {job_index} failed: {error_msg}")
            self._handle_processing_failure(job_index, error_msg)
            return False  # Will not retry

    def _handle_processing_failure(self, job_index: str, error_msg: str, is_submission_failure: bool = False) -> None:
        """
        Handle any failure during submission or fetching.

        Parameters
        ----------
        job_index : str
            Identifier of the job that failed.
        error_msg : str
            Description of the failure cause.
        is_submission_failure : bool, default=False
            True if the failure occurred during initial job submission.
        """
        self.failures.append((job_index, error_msg))
        # Attempt to mark state as FAILED on the server/client side if possible
        # This might be redundant if _fetch_job_result already did it.
        try:
            # Don't check required_state, just try to get the state object
            job_state = self.client._get_job_state_object(job_index)  # Assume this method exists
            if job_state and job_state.state not in [JobStateEnum.FAILED, JobStateEnum.COMPLETED]:
                job_state.state = JobStateEnum.FAILED
                if self.verbose:
                    logger.debug(f"Marked job {job_index} state as FAILED locally.")
                # Optionally: self.client.update_job_state(job_index, JobStateEnum.FAILED) if needed
        except Exception as state_update_err:
            if self.verbose:
                logger.warning(
                    f"Could not update state to FAILED for job {job_index} " f"after failure: {state_update_err}"
                )

        # If it's specifically a submission failure, don't increment processed_count here.
        # The calling function (_submit_job_fetch) will handle incrementing if needed.
        # For fetch failures, the caller (_process_completed_futures) increments if not retrying.

        # Clean up retry counts if the failure is terminal
        if job_index in self.retry_counts:
            del self.retry_counts[job_index]

    def _log_final_status(self) -> None:
        """
        Log the final processing summary and check for consistency.

        Notes
        -----
        Warns if the number of processed jobs does not match the total.
        """
        # Final check for consistency
        if self.processed_count != self.total_jobs:
            logger.warning(
                f"Processing loop finished, but processed count ({self.processed_count}) "
                f"doesn't match total jobs ({self.total_jobs}). "
                f"Results: {len(self.results)}, Failures: {len(self.failures)}, "
                f"Still Active: {len(self.active_futures)}, "
                f"Pending Submit: {len(self.job_indices_to_submit)}, "
                f"Pending Retry: {len(self.retry_job_ids)}"
            )
            # You might want to add logic here to mark any remaining jobs in queues/active as failed.

        logger.info(f"Concurrent processing finished. " f"Success: {len(self.results)}, Failures: {len(self.failures)}")


class NvIngestClient:
    """
    A client class for interacting with the nv-ingest service, supporting custom client allocators.
    """

    def __init__(
        self,
        message_client_allocator: Type[MessageBrokerClientBase] = RestClient,
        message_client_hostname: Optional[str] = "localhost",
        message_client_port: Optional[int] = 7670,
        message_client_kwargs: Optional[Dict[str, Any]] = None,
        msg_counter_id: Optional[str] = "nv-ingest-message-id",
        worker_pool_size: int = 1,
    ) -> None:
        """
        Initialize the NvIngestClient.

        Parameters
        ----------
        message_client_allocator : Type[MessageBrokerClientBase], optional
            Callable that creates the message broker client. Defaults to RestClient.
        message_client_hostname : str, optional
            Hostname of the REST/message service. Defaults to "localhost".
        message_client_port : int, optional
            Port of the REST/message service. Defaults to 7670.
        message_client_kwargs : dict, optional
            Extra keyword arguments passed to the client allocator.
        msg_counter_id : str, optional
            Identifier for message counting. Defaults to "nv-ingest-message-id".
        worker_pool_size : int, optional
            Number of workers in the thread pool. Defaults to 1.

        Returns
        -------
        None
        """

        self._current_message_id = 0
        self._job_states = {}
        self._job_index_to_job_spec = {}
        self._message_client_hostname = message_client_hostname or "localhost"
        self._message_client_port = message_client_port or 7670
        self._message_counter_id = msg_counter_id or "nv-ingest-message-id"
        self._message_client_kwargs = message_client_kwargs or {}

        logger.debug("Instantiate NvIngestClient:\n%s", str(self))
        self._message_client = message_client_allocator(
            host=self._message_client_hostname,
            port=self._message_client_port,
            **self._message_client_kwargs,
        )

        # Initialize the worker pool with the specified size
        self._worker_pool = ThreadPoolExecutor(max_workers=worker_pool_size)

        self._telemetry = {}

    def __str__(self) -> str:
        """
        Returns a string representation of the NvIngestClient configuration and runtime state.

        Returns
        -------
        str
            A string representation of the client showing the Redis configuration.
        """
        info = "NvIngestClient:\n"
        info += f" message_client_host: {self._message_client_hostname}\n"
        info += f" message_client_port: {self._message_client_port}\n"

        return info

    def _generate_job_index(self) -> str:
        """
        Generates a unique job ID by combining a UUID with an incremented value from Redis.

        Returns
        -------
        str
            A unique job ID in the format of "<UUID>_<Redis incremented value>".  IF the client
            is a RedisClient. In the case of a RestClient it is simply the UUID.
        """

        job_index = str(self._current_message_id)
        self._current_message_id += 1

        return job_index

    def _pop_job_state(self, job_index: str) -> JobState:
        """
        Deletes the job with the specified ID from the job tracking dictionary.

        Parameters
        ----------
        job_index : str
            The ID of the job to delete.
        """

        job_state = self._get_and_check_job_state(job_index)
        self._job_states.pop(job_index)

        return job_state

    def _get_and_check_job_state(
        self,
        job_index: str,
        required_state: Optional[Union[JobStateEnum, List[JobStateEnum]]] = None,
    ) -> JobState:
        """
        Retrieve and optionally validate the state of a job.

        Parameters
        ----------
        job_index : str
            The client-side identifier of the job.
        required_state : JobStateEnum or list of JobStateEnum, optional
            State or list of states the job must currently be in. If provided and
            the job is not in one of these states, an error is raised.

        Returns
        -------
        JobState
            The state object for the specified job.

        Raises
        ------
        ValueError
            If the job does not exist or is not in an allowed state.
        """
        if required_state and not isinstance(required_state, list):
            required_state = [required_state]
        job_state = self._job_states.get(job_index)

        if not job_state:
            raise ValueError(f"Job with ID {job_index} does not exist in JobStates: {self._job_states}")
        if required_state and (job_state.state not in required_state):
            raise ValueError(
                f"Job with ID {job_state.job_spec.job_id} has invalid state "
                f"{job_state.state}, expected {required_state}"
            )

        return job_state

    def job_count(self) -> int:
        """
        Get the number of jobs currently tracked by the client.

        Returns
        -------
        int
            The total count of jobs in internal state tracking.
        """
        return len(self._job_states)

    def _add_single_job(self, job_spec: JobSpec) -> str:
        """
        Add a single job specification to internal tracking.

        Parameters
        ----------
        job_spec : JobSpec
            The specification object describing the job.

        Returns
        -------
        str
            The newly generated job index.
        """
        job_index = self._generate_job_index()

        self._job_states[job_index] = JobState(job_spec=job_spec)

        return job_index

    def add_job(self, job_spec: Union[BatchJobSpec, JobSpec]) -> Union[str, List[str]]:
        """
        Add one or more jobs to the client for later processing.

        Parameters
        ----------
        job_spec : JobSpec or BatchJobSpec
            A single job specification or a batch containing multiple specs.

        Returns
        -------
        str or list of str
            The job index for a single spec, or a list of indices for a batch.

        Raises
        ------
        ValueError
            If an unsupported type is provided.
        """
        if isinstance(job_spec, JobSpec):
            job_index = self._add_single_job(job_spec)
            self._job_index_to_job_spec[job_index] = job_spec

            return job_index
        elif isinstance(job_spec, BatchJobSpec):
            job_indexes = []
            for _, job_specs in job_spec.job_specs.items():
                for job in job_specs:
                    job_index = self._add_single_job(job)
                    job_indexes.append(job_index)
                    self._job_index_to_job_spec[job_index] = job
            return job_indexes
        else:
            raise ValueError(f"Unexpected type: {type(job_spec)}")

    def create_job(
        self,
        payload: Dict[str, Any],
        source_id: str,
        source_name: str,
        document_type: Optional[str] = None,
        tasks: Optional[List[Task]] = None,
        extended_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Construct and register a new job from provided metadata.

        Parameters
        ----------
        payload : dict
            The data payload for the job.
        source_id : str
            Identifier of the data source.
        source_name : str
            Human-readable name for the source.
        document_type : str, optional
            Type of document (inferred from source_name if omitted).
        tasks : list of Task, optional
            Initial set of processing tasks to attach.
        extended_options : dict, optional
            Extra parameters for advanced configuration.

        Returns
        -------
        str
            The client-side job index.

        Raises
        ------
        ValueError
            If job creation parameters are invalid.
        """

        document_type = document_type or source_name.split(".")[-1]
        job_spec = JobSpec(
            payload=payload or {},
            tasks=tasks,
            document_type=document_type,
            source_id=source_id,
            source_name=source_name,
            extended_options=extended_options,
        )

        job_id = self.add_job(job_spec)
        return job_id

    def add_task(self, job_index: str, task: Task) -> None:
        """
        Attach an existing Task object to a pending job.

        Parameters
        ----------
        job_index : str
            The client-side identifier of the target job.
        task : Task
            The task instance to add.
        """
        job_state = self._get_and_check_job_state(job_index, required_state=JobStateEnum.PENDING)

        job_state.job_spec.add_task(task)

    def create_task(
        self,
        job_index: Union[str, int],
        task_type: TaskType,
        task_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Create and attach a new task to a pending job by type and parameters.

        Parameters
        ----------
        job_index : str or int
            Identifier of the job to modify.
        task_type : TaskType
            Enum specifying the kind of task to create.
        task_params : dict, optional
            Parameters for the new task.

        Raises
        ------
        ValueError
            If the job does not exist or is not pending.
        """
        task_params = task_params or {}

        return self.add_task(job_index, task_factory(task_type, **task_params))

    def _fetch_job_result(
        self,
        job_index: str,
        timeout: Tuple[int, Optional[float]] = (100, None),
        data_only: bool = False,
    ) -> Tuple[Any, str, Optional[str]]:
        """
        Retrieve the result of a submitted job, handling status codes.

        Parameters
        ----------
        job_index : str
            Client-side job identifier.
        timeout : tuple
            Timeouts (connect, read) for the fetch operation.
        data_only : bool, optional
            If True, return only the 'data' portion of the payload.

        Returns
        -------
        result_data : any
            Parsed job result or full JSON payload.
        job_index : str
            Echoes the client-side job ID.
        trace_id : str or None
            Trace identifier from the message client.

        Raises
        ------
        TimeoutError
            If the job is not yet ready (HTTP 202).
        RuntimeError
            For terminal server errors (HTTP 404/500, etc.).
        ValueError
            On JSON decoding errors or missing state.
        Exception
            For unexpected issues.
        """
        try:
            # Get job state using the client-side index
            job_state = self._get_and_check_job_state(
                job_index, required_state=[JobStateEnum.SUBMITTED, JobStateEnum.SUBMITTED_ASYNC]
            )

            # Validate server_job_id before making the call
            server_job_id = job_state.job_id
            if not server_job_id:
                error_msg = (
                    f"Cannot fetch job index {job_index}: Server Job ID is missing or invalid in state"
                    f" {job_state.state}."
                )
                logger.error(error_msg)
                job_state.state = JobStateEnum.FAILED
                raise ValueError(error_msg)

            # Fetch using the *server-assigned* job ID
            response = self._message_client.fetch_message(server_job_id, timeout)
            job_state.trace_id = response.trace_id  # Store trace ID from this fetch attempt

            # --- Handle ResponseSchema Code ---
            if response.response_code == 0:  # Success (e.g., HTTP 200)
                try:
                    # Don't change state here yet, only after successful processing
                    logger.debug(
                        f"Received successful response for job index {job_index} (Server ID: {server_job_id}). "
                        f"Decoding JSON."
                    )

                    response_json = json.loads(response.response)
                    result_data = response_json.get("data") if data_only else response_json

                    # Mark state as PROCESSING *after* successful decode, just before returning
                    job_state.state = JobStateEnum.PROCESSING
                    # Pop state *only* after successful processing is complete
                    self._pop_job_state(job_index)
                    logger.debug(
                        f"Successfully processed and removed job index {job_index} (Server ID: {server_job_id})"
                    )
                    return result_data, job_index, job_state.trace_id

                except json.JSONDecodeError as err:
                    logger.error(
                        f"Failed to decode JSON response for job index {job_index} (Server ID: {server_job_id}):"
                        f" {err}. Response text: {response.response[:500]}"
                    )
                    job_state.state = JobStateEnum.FAILED  # Mark as failed due to decode error
                    raise ValueError(f"Error decoding job result JSON: {err}") from err
                except Exception as e:
                    # Catch other potential errors during processing of successful response
                    logger.exception(
                        f"Error processing successful response for job index {job_index} (Server ID: {server_job_id}):"
                        f" {e}"
                    )
                    job_state.state = JobStateEnum.FAILED
                    raise  # Re-raise unexpected errors

            elif response.response_code == 2:  # Job Not Ready (e.g., HTTP 202)
                # Raise TimeoutError to signal the calling retry loop in fetch_job_result
                logger.debug(
                    f"Job index {job_index} (Server ID: {server_job_id}) not ready (Response Code: 2). Signaling retry."
                )
                # Do not change job state here, remains SUBMITTED
                raise TimeoutError(f"Job not ready: {response.response_reason}")

            else:  # Failure from RestClient (response_code == 1, including 404, 400, 500, conn errors)
                # Log the failure reason from the ResponseSchema
                error_msg = (
                    f"Terminal failure fetching result for client index {job_index} (Server ID: {server_job_id}). "
                    f"Code: {response.response_code}, Reason: {response.response_reason}"
                )
                logger.error(error_msg)
                job_state.state = JobStateEnum.FAILED  # Mark job as failed in the client
                # Do NOT pop the state for failed jobs here
                # Raise RuntimeError to indicate a terminal failure for this fetch attempt
                raise RuntimeError(error_msg)

        except (TimeoutError, ValueError, RuntimeError):
            # Re-raise specific handled exceptions
            raise
        except Exception as err:
            # Catch unexpected errors during the process (e.g., in _get_and_check_job_state)
            logger.exception(f"Unexpected error during fetch process for job index {job_index}: {err}")
            # Attempt to mark state as FAILED if possible and state object exists
            if "job_state" in locals() and hasattr(job_state, "state"):
                job_state.state = JobStateEnum.FAILED
            raise  # Re-raise the original exception

    def fetch_job_result_cli(
        self,
        job_ids: Union[str, List[str]],
        data_only: bool = False,
    ) -> List[Tuple[Any, str, Optional[str]]]:
        """
        Fetch job results via CLI semantics (synchronous list return).

        Parameters
        ----------
        job_ids : str or list of str
            Single or multiple client-side job identifiers.
        data_only : bool, optional
            If True, extract only the 'data' field. Default is False.

        Returns
        -------
        list of (result_data, job_index, trace_id)
            List of tuples for each fetched job.
        """
        if isinstance(job_ids, str):
            job_ids = [job_ids]

        return [self._fetch_job_result(job_id, data_only=data_only) for job_id in job_ids]

    def process_jobs_concurrently(
        self,
        job_indices: Union[str, List[str]],
        job_queue_id: Optional[str] = None,
        concurrency_limit: int = 64,
        timeout: int = 100,
        max_job_retries: Optional[int] = None,
        retry_delay: float = 5.0,
        fail_on_submit_error: bool = False,
        completion_callback: Optional[Callable[[Any, str], None]] = None,
        return_failures: bool = False,
        data_only: bool = True,
        verbose: bool = False,
    ) -> Union[List[Any], Tuple[List[Any], List[Tuple[str, str]]]]:
        """
        Submit and fetch multiple jobs concurrently.

        Parameters
        ----------
        job_indices : str or list of str
            Single or multiple job indices to process.
        job_queue_id : str, optional
            Queue identifier for submission.
        concurrency_limit : int, optional
            Max number of simultaneous in-flight jobs. Default is 128.
        timeout : int, optional
            Timeout in seconds per fetch attempt. Default is 100.
        max_job_retries : int, optional
            Max retries for 'not ready' jobs. None for infinite. Default is None.
        retry_delay : float, optional
            Delay in seconds between retry cycles. Default is 5.0.
        fail_on_submit_error : bool, optional
            If True, abort on submission error. Default is False.
        completion_callback : callable, optional
            Called on each successful fetch as (result_data, job_index).
        return_failures : bool, optional
            If True, return (results, failures). Default is False.
        data_only : bool, optional
            If True, return only payload 'data'. Default is True.
        verbose : bool, optional
            If True, enable debug logging. Default is False.

        Returns
        -------
        results : list
            List of successful job results when `return_failures` is False.
        results, failures : tuple
            Tuple of (successful results, failure tuples) when `return_failures` is True.

        Raises
        ------
        RuntimeError
            If `fail_on_submit_error` is True and a submission fails.
        """
        # Normalize single index to list
        if isinstance(job_indices, str):
            job_indices = [job_indices]

        # Handle empty input
        if not job_indices:
            return ([], []) if return_failures else []

        # Prepare timeout tuple for fetch calls
        effective_timeout: Tuple[int, None] = (timeout, None)

        # Delegate to the concurrent processor
        processor = _ConcurrentProcessor(
            client=self,
            job_indices=job_indices,
            job_queue_id=job_queue_id,
            concurrency_limit=concurrency_limit,
            timeout=effective_timeout,
            max_job_retries=max_job_retries,
            retry_delay=retry_delay,
            completion_callback=completion_callback,
            fail_on_submit_error=fail_on_submit_error,
            data_only=data_only,
            verbose=verbose,
        )

        results, failures = processor.run()

        if return_failures:
            return results, failures

        if failures:
            logger.warning(f"{len(failures)} job(s) failed during concurrent processing." " Check logs for details.")
        return results

    def _ensure_submitted(self, job_ids: Union[str, List[str]]) -> None:
        """
        Block until all specified jobs have been marked submitted.

        Parameters
        ----------
        job_ids : str or list of str
            One or more job indices expected to reach a SUBMITTED state.
        """
        if isinstance(job_ids, str):
            job_ids = [job_ids]  # Ensure job_ids is always a list

        submission_futures = {}
        for job_id in job_ids:
            job_state = self._get_and_check_job_state(
                job_id,
                required_state=[JobStateEnum.SUBMITTED, JobStateEnum.SUBMITTED_ASYNC],
            )
            if job_state.state == JobStateEnum.SUBMITTED_ASYNC:
                submission_futures[job_state.future] = job_state

        for future in as_completed(submission_futures.keys()):
            job_state = submission_futures[future]
            job_state.state = JobStateEnum.SUBMITTED
            job_state.trace_id = future.result()[0]  # Trace_id from `submit_job` endpoint submission
            job_state.future = None

    def fetch_job_result_async(self, job_ids: Union[str, List[str]], data_only: bool = True) -> Dict[Future, str]:
        """
        Fetches job results for a list or a single job ID asynchronously and returns a mapping of futures to job IDs.

        Parameters:
            job_ids (Union[str, List[str]]): A single job ID or a list of job IDs.
            timeout (float): Timeout (connect, read) for fetching each job result, in seconds.
            data_only (bool): Whether to return only the data part of the job result.

        Returns:
            Dict[Future, str]: A dictionary mapping each future to its corresponding job ID.
        """
        if isinstance(job_ids, str):
            job_ids = [job_ids]  # Ensure job_ids is always a list

        # Make sure all jobs have actually been submitted before launching fetches.
        self._ensure_submitted(job_ids)

        future_to_job_id = {}
        for job_id in job_ids:
            job_state = self._get_and_check_job_state(job_id)
            future = self._worker_pool.submit(self.fetch_job_result_cli, job_id, data_only)
            job_state.future = future
            future_to_job_id[future] = job_id

        return future_to_job_id

    def _submit_job(
        self,
        job_index: str,
        job_queue_id: str,
    ) -> Optional[Dict]:
        """
        Submits a job to a specified job queue and optionally waits for a response if blocking is True.

        Parameters
        ----------
        job_index : str
            The unique identifier of the job to be submitted.
        job_queue_id : str
            The ID of the job queue where the job will be submitted.

        Returns
        -------
        Optional[Dict]
            The job result if blocking is True and a result is available before the timeout, otherwise None.

        Raises
        ------
        Exception
            If submitting the job fails.
        """

        job_state = self._get_and_check_job_state(
            job_index, required_state=[JobStateEnum.PENDING, JobStateEnum.SUBMITTED_ASYNC]
        )

        try:
            message = json.dumps(job_state.job_spec.to_dict())

            response = self._message_client.submit_message(job_queue_id, message, for_nv_ingest=True)
            x_trace_id = response.trace_id
            transaction_id = response.transaction_id
            job_id = "" if transaction_id is None else transaction_id.replace('"', "")
            logger.debug(f"Submitted job {job_index} to queue {job_queue_id} and got back job ID {job_id}")

            job_state.state = JobStateEnum.SUBMITTED
            job_state.job_id = job_id

            # Free up memory -- payload should never be used again, and we don't want to keep it around.
            job_state.job_spec.payload = None

            return x_trace_id
        except Exception as err:
            err_msg = f"Failed to submit job {job_index} to queue {job_queue_id}: {err}"
            logger.exception(err_msg)
            job_state.state = JobStateEnum.FAILED

            raise

    def submit_job(
        self,
        job_indices: Union[str, List[str]],
        job_queue_id: str,
        batch_size: int = 10,
    ) -> List[str]:
        """
        Submit one or more jobs in batches.

        Parameters
        ----------
        job_indices : str or list of str
            Job indices to submit.
        job_queue_id : str
            Queue identifier for submission.
        batch_size : int, optional
            Maximum number of jobs per batch. Default is 10.

        Returns
        -------
        list of str
            Trace identifiers for each submitted job.

        Raises
        ------
        Exception
            Propagates first error if any job in a batch fails.
        """
        if isinstance(job_indices, str):
            job_indices = [job_indices]

        results = []
        total_batches = math.ceil(len(job_indices) / batch_size)

        submission_errors = []
        for batch_num in range(total_batches):
            batch_start = batch_num * batch_size
            batch_end = batch_start + batch_size
            batch = job_indices[batch_start:batch_end]

            # Submit each batch of jobs
            for job_id in batch:
                try:
                    x_trace_id = self._submit_job(job_id, job_queue_id)
                except Exception as e:  # Even if one fails, we should continue with the rest of the batch.
                    submission_errors.append(e)
                    continue
                results.append(x_trace_id)

        if submission_errors:
            error_msg = str(submission_errors[0])
            if len(submission_errors) > 1:
                error_msg += f"... [{len(submission_errors) - 1} more messages truncated]"
            raise type(submission_errors[0])(error_msg)

        return results

    def submit_job_async(self, job_indices: Union[str, List[str]], job_queue_id: str) -> Dict[Future, str]:
        """
        Asynchronously submits one or more jobs to a specified job queue using a thread pool.
        This method handles both single job ID or a list of job IDs.

        Parameters
        ----------
        job_indices : Union[str, List[str]]
            A single job ID or a list of job IDs to be submitted.
        job_queue_id : str
            The ID of the job queue where the jobs will be submitted.

        Returns
        -------
        Dict[Future, str]
            A dictionary mapping futures to their respective job IDs for later retrieval of outcomes.

        Notes
        -----
        - This method queues the jobs for asynchronous submission and returns a mapping of futures to job IDs.
        - It does not wait for any of the jobs to complete.
        - Ensure that each job is in the proper state before submission.
        """

        if isinstance(job_indices, str):
            job_indices = [job_indices]  # Convert single job_id to a list

        future_to_job_index = {}
        for job_index in job_indices:
            job_state = self._get_and_check_job_state(job_index, JobStateEnum.PENDING)
            job_state.state = JobStateEnum.SUBMITTED_ASYNC

            future = self._worker_pool.submit(self.submit_job, job_index, job_queue_id)
            job_state.future = future
            future_to_job_index[future] = job_index

        return future_to_job_index

    def create_jobs_for_batch(self, files_batch: List[str], tasks: Dict[str, Any]) -> List[str]:
        """
        Create and submit job specifications (JobSpecs) for a batch of files, returning the job IDs.
        This function takes a batch of files, processes each file to extract its content and type,
        creates a job specification (JobSpec) for each file, and adds tasks from the provided task
        list. It then submits the jobs to the client and collects their job IDs.

        Parameters
        ----------
        files_batch : List[str]
            A list of file paths to be processed. Each file is assumed to be in a format compatible
            with the `extract_file_content` function, which extracts the file's content and type.
        tasks : Dict[str, Any]
            A dictionary of tasks to be added to each job. The keys represent task names, and the
            values represent task specifications or configurations. Standard tasks include "split",
            "extract", "store", "caption", "dedup", "filter", "embed".

        Returns
        -------
        Tuple[List[JobSpec], List[str]]
            A Tuple containing the list of JobSpecs and list of job IDs corresponding to the submitted jobs.
            Each job ID is returned by the client's `add_job` method.

        Raises
        ------
        ValueError
            If there is an error extracting the file content or type from any of the files, a
            ValueError will be logged, and the corresponding file will be skipped.

        Notes
        -----
        - The function assumes that a utility function `extract_file_content` is defined elsewhere,
          which extracts the content and type from the provided file paths.
        - For each file, a `JobSpec` is created with relevant metadata, including document type and
          file content. Various tasks are conditionally added based on the provided `tasks` dictionary.
        - The job specification includes tracing options with a timestamp (in nanoseconds) for
          diagnostic purposes.

        Examples
        --------
        Suppose you have a batch of files and tasks to process:
        >>> files_batch = ["file1.txt", "file2.pdf"]
        >>> tasks = {"split": ..., "extract_txt": ..., "store": ...}
        >>> client = NvIngestClient()
        >>> job_ids = client.create_job_specs_for_batch(files_batch, tasks)
        >>> print(job_ids)
        ['job_12345', 'job_67890']

        In this example, jobs are created and submitted for the files in `files_batch`, with the
        tasks in `tasks` being added to each job specification. The returned job IDs are then
        printed.

        See Also
        --------
        create_job_specs_for_batch: Function that creates job specifications for a batch of files.
        JobSpec : The class representing a job specification.
        """
        if not isinstance(tasks, dict):
            raise ValueError("`tasks` must be a dictionary of task names -> task specifications.")

        job_specs = create_job_specs_for_batch(files_batch)

        job_ids = []
        for job_spec in job_specs:
            logger.debug(f"Tasks: {tasks.keys()}")
            for task in tasks:
                logger.debug(f"Task: {task}")

            file_type = job_spec.document_type

            seen_tasks = set()  # For tracking tasks and rejecting duplicate tasks.

            for task_name, task_config in tasks.items():
                if task_name.lower().startswith("extract_"):
                    task_file_type = task_name.split("_", 1)[1]
                    if file_type.lower() != task_file_type.lower():
                        continue
                elif not is_valid_task_type(task_name.upper()):
                    raise ValueError(f"Invalid task type: '{task_name}'")

                if str(task_config) in seen_tasks:
                    raise ValueError(f"Duplicate task detected: {task_name} with config {task_config}")

                job_spec.add_task(task_config)

                seen_tasks.add(str(task_config))

            job_id = self.add_job(job_spec)
            job_ids.append(job_id)

        return job_ids
