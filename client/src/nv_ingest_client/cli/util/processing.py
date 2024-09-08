# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import concurrent
import copy
import json
import logging
import os
import re
import time
from collections import defaultdict
from concurrent.futures import as_completed
from statistics import mean
from statistics import median
from typing import Dict, Any, Tuple
from typing import List
from typing import Tuple
from typing import Type

from click import style
from nv_ingest_client.client import NvIngestClient
from nv_ingest_client.message_clients.rest.rest_client import RestClient
from nv_ingest_client.primitives import JobSpec
from nv_ingest_client.util.file_processing.extract import extract_file_content
from nv_ingest_client.util.util import check_ingest_result
from nv_ingest_client.util.util import estimate_page_count
from pydantic import BaseModel
from pydantic import ValidationError
from tqdm import tqdm

logger = logging.getLogger(__name__)


def highlight_error_in_original(original_str: str, task_name: str, error_detail: Dict[str, Any]) -> str:
    """
    Highlights the error-causing text in the original JSON string based on the error type.

    This function identifies errors in the original JSON string and highlights the specific
    part of the string responsible for the error. It handles two main types of errors:
    - For 'extra fields' errors, it highlights the extra field that caused the issue.
    - For 'missing fields' errors, it appends a clear message indicating the missing field.

    Parameters
    ----------
    original_str : str
        The original JSON string that caused the error. The function will modify and return this
        string with the problematic fields highlighted.

    task_name : str
        The name of the task associated with the error. This is included in the error message when
        highlighting missing fields.

    error_detail : Dict[str, Any]
        A dictionary containing details about the error. The dictionary should contain the following keys:
        - 'type' (str): The type of error (e.g., "value_error.extra", "value_error.missing").
        - 'loc' (List[Any]): A list of keys/indices indicating the location of the error in the JSON structure.

    Returns
    -------
    str
        The original string with the error-causing field highlighted. If the error is related to
        extra fields, the problematic field is highlighted in blue and bold. If the error is due
        to missing fields, a message indicating the missing field is appended to the string.

    Notes
    -----
    - The function uses the `style` method (assumed to be defined elsewhere, likely from a
      terminal or text formatting library) to apply color and bold formatting to the highlighted text.
    - The 'loc' key in `error_detail` is a list that represents the path to the error-causing field in the JSON.

    Examples
    --------
    Suppose there is an error in the original JSON string due to an extra field:

    >>> original_str = '{"name": "file1.txt", "extra_field": "some_value"}'
    >>> task_name = "validate_document"
    >>> error_detail = {
    ...     "type": "value_error.extra",
    ...     "loc": ["extra_field"]
    ... }
    >>> highlighted_str = highlight_error_in_original(original_str, task_name, error_detail)
    >>> print(highlighted_str)
    {"name": "file1.txt", "extra_field": "some_value"}  # The 'extra_field' will be highlighted

    In this case, the function will highlight the `extra_field` in blue and bold in the returned string.

    See Also
    --------
    style : Function used to apply formatting to the error-causing text (e.g., coloring or bolding).
    """

    error_type = error_detail["type"]
    error_location = "->".join(map(str, error_detail["loc"]))
    if error_type == "value_error.extra":
        error_key = error_detail["loc"][-1]
        highlighted_key = style(error_key, fg="blue", bold=True)
        highlighted_str = original_str.replace(f'"{error_key}"', highlighted_key)
    elif error_type in ["value_error.missing", "value_error.any_str.min_length"]:
        missing_message = style(f"'{error_location}'", fg="blue", bold=True)
        highlighted_str = (
            f"{original_str}\n(Schema Error): Missing required parameter for task '{task_name}'"
            f" {missing_message}\n -> {original_str}"
        )
    else:
        error_key = error_detail["loc"][-1]
        highlighted_key = style(error_key, fg="blue", bold=True)
        highlighted_str = original_str.replace(f'"{error_key}"', highlighted_key)

    return highlighted_str


def format_validation_error(e: ValidationError, task_id, original_str: str) -> str:
    """
    Formats validation errors with appropriate highlights and returns a detailed error message.
    """
    error_messages = []
    for error in e.errors():
        error_message = f"(Schema Error): {error['msg']}"
        highlighted_str = highlight_error_in_original(original_str, task_id, error)
        error_messages.append(f"{error_message}\n -> {highlighted_str}")

    return "\n".join(error_messages)


def check_schema(schema: Type[BaseModel], options: dict, task_id: str, original_str: str) -> BaseModel:
    try:
        return schema(**options)
    except ValidationError as e:
        error_message = format_validation_error(e, task_id, original_str)
        raise ValueError(error_message) from e


def report_stage_statistics(
        stage_elapsed_times: defaultdict(list), total_trace_elapsed: float, abs_elapsed: float
) -> None:
    """
    Reports the statistics for each processing stage, including average, median, total time spent,
    and their respective percentages of the total processing time.

    Parameters
    ----------
    stage_elapsed_times : defaultdict(list)
        A defaultdict containing lists of elapsed times for each processing stage, in nanoseconds.
    total_trace_elapsed : float
        The total elapsed time across all processing stages, in nanoseconds.
    abs_elapsed : float
        The absolute elapsed time from the start to the end of processing, in nanoseconds.

    Notes
    -----
    This function logs the average, median, and total time for each stage, along with the percentage of total
    computation.
    It also calculates and logs the unresolved time, if any, that is not accounted for by the recorded stages.
    """

    for stage, times in stage_elapsed_times.items():
        if times:
            avg_time = mean(times)
            med_time = median(times)
            total_stage_time = sum(times)
            percent_of_total = (total_stage_time / total_trace_elapsed * 100) if total_trace_elapsed > 0 else 0
            logger.info(
                f"{stage}: Avg: {avg_time / 1e6:.2f} ms, Median: {med_time / 1e6:.2f} ms, "
                f"Total Time: {total_stage_time / 1e6:.2f} ms, Total % of Trace Computation: {percent_of_total:.2f}%"
            )

    unresolved_time = abs_elapsed - total_trace_elapsed
    if unresolved_time > 0:
        percent_unresolved = unresolved_time / abs_elapsed * 100
        logger.info(
            f"Unresolved time: {unresolved_time / 1e6:.2f} ms, Percent of Total Elapsed: {percent_unresolved:.2f}%"
        )
    else:
        logger.info("No unresolved time detected. Trace times account for the entire elapsed duration.")


def report_overall_speed(total_pages_processed: int, start_time_ns: int, total_files: int) -> None:
    """
    Reports the overall processing speed based on the number of pages and files processed.

    Parameters
    ----------
    total_pages_processed : int
        The total number of pages processed.
    start_time_ns : int
        The nanosecond timestamp marking the start of processing.
    total_files : int
        The total number of files processed.

    Notes
    -----
    This function calculates the total elapsed time from the start of processing and reports the throughput
    in terms of pages and files processed per second.
    """

    total_elapsed_time_ns = time.time_ns() - start_time_ns
    total_elapsed_time_s = total_elapsed_time_ns / 1_000_000_000  # Convert nanoseconds to seconds

    throughput_pages = total_pages_processed / total_elapsed_time_s  # pages/sec
    throughput_files = total_files / total_elapsed_time_s  # files/sec

    logger.info(f"Processed {total_files} files in {total_elapsed_time_s:.2f} seconds.")
    logger.info(f"Total pages processed: {total_pages_processed}")
    logger.info(f"Throughput (Pages/sec): {throughput_pages:.2f}")
    logger.info(f"Throughput (Files/sec): {throughput_files:.2f}")


def report_statistics(
        start_time_ns: int,
        stage_elapsed_times: defaultdict,
        total_pages_processed: int,
        total_files: int,
) -> None:
    """
    Aggregates and reports statistics for the entire processing session.

    Parameters
    ----------
    start_time_ns : int
        The nanosecond timestamp marking the start of the processing.
    stage_elapsed_times : defaultdict(list)
        A defaultdict where each key is a processing stage and each value is a list
        of elapsed times in nanoseconds for that stage.
    total_pages_processed : int
        The total number of pages processed during the session.
    total_files : int
        The total number of files processed during the session.

    Notes
    -----
    This function calculates the absolute elapsed time from the start of processing to the current
    time and the total time taken by all stages.
    """

    abs_elapsed = time.time_ns() - start_time_ns
    total_trace_elapsed = sum(sum(times) for times in stage_elapsed_times.values())
    report_stage_statistics(stage_elapsed_times, total_trace_elapsed, abs_elapsed)
    report_overall_speed(total_pages_processed, start_time_ns, total_files)


def process_response(response, stage_elapsed_times):
    """
    Process the response to extract trace data and calculate elapsed time for each stage.

    Parameters
    ----------
    response : dict
        The response dictionary containing trace information for processing stages.
    stage_elapsed_times : defaultdict(list)
        A defaultdict to accumulate elapsed times for each processing stage.

    Notes
    -----
    The function iterates over trace data in the response, identifying entry and exit times for
    each stage, and calculates the elapsed time which is then appended to the respective stage in
    `stage_elapsed_times`.
    """

    trace_data = response.get("trace", {})
    for key, entry_time in trace_data.items():
        if "entry" in key:
            exit_key = key.replace("entry", "exit")
            exit_time = trace_data.get(exit_key)
            if exit_time:
                stage_name = key.split("::")[2]
                elapsed_time = exit_time - entry_time
                stage_elapsed_times[stage_name].append(elapsed_time)


def organize_documents_by_type(response_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Organize documents by their content type.

    This function takes a list of response documents, extracts the content type from
    each document's metadata, and organizes the documents into a dictionary, where the
    keys are content types and the values are lists of documents belonging to each type.

    Parameters
    ----------
    response_data : List[Dict[str, Any]]
        A list of documents, where each document is represented as a dictionary.
        Each dictionary must contain a 'metadata' field that may be either a JSON
        string or a dictionary. The metadata is expected to have a "content_metadata"
        field containing the document's type.

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        A dictionary mapping document types (as strings) to lists of documents.
        Each key represents a document type, and the associated value is a list of
        documents that belong to that type.

    Notes
    -----
    - The 'metadata' field of each document can be either a JSON string or a
      dictionary. If it is a string, it will be parsed into a dictionary.

    - The function assumes that each document has a valid "content_metadata" field,
      which contains a "type" key that indicates the document's type.

    - If a document type is encountered for the first time, a new entry is created
      in the result dictionary, and subsequent documents of the same type are added
      to the corresponding list.

    Examples
    --------
    Suppose `response_data` contains the following structure:

    >>> response_data = [
    ...     {"metadata": {"content_metadata": {"type": "report"}}},
    ...     {"metadata": '{"content_metadata": {"type": "summary"}}'},
    ...     {"metadata": {"content_metadata": {"type": "report"}}}
    ... ]
    >>> organize_documents_by_type(response_data)
    {'report': [{'metadata': {'content_metadata': {'type': 'report'}}},
                {'metadata': {'content_metadata': {'type': 'report'}}}],
     'summary': [{'metadata': {'content_metadata': {'type': 'summary'}}}]}

    In this example, the documents are organized into two types: "report" and "summary".

    See Also
    --------
    json.loads : Used to parse metadata if it is a JSON string.
    """

    doc_map = {}
    for document in response_data:
        doc_meta = document["metadata"]
        # TODO: fix this. doc_meta can be a json string or a dict.
        if isinstance(doc_meta, str):
            doc_meta = json.loads(doc_meta)
        doc_content_metadata = doc_meta["content_metadata"]
        doc_type = doc_content_metadata["type"]
        if doc_type not in doc_map:
            doc_map[doc_type] = []
        doc_map[doc_type].append(document)
    return doc_map


def save_response_data(response, output_directory):
    """
    Save the response data into categorized metadata JSON files.

    This function processes the response data, organizes it based on document
    types, and saves the organized data into a specified output directory as JSON
    files. The output files are named according to the document metadata and are
    saved in subdirectories based on the document type.

    Parameters
    ----------
    response : dict
        A dictionary containing the API response data. It must contain a "data"
        field, which is expected to be a list of document entries. Each document
        entry should contain metadata, which includes information about the
        document's source.

    output_directory : str
        The path to the directory where the JSON metadata files should be saved.
        Subdirectories will be created based on the document types, and the
        metadata files will be stored within these subdirectories.

    Returns
    -------
    None
        This function does not return any values. It writes output to the filesystem.

    Notes
    -----
    - The response dictionary must contain a "data" field, and this field must be
      a non-empty list of documents. If the data field is missing or empty, the
      function will return without saving any files.

    - The documents are organized by type, and each document's metadata is written
      to a JSON file within a subdirectory corresponding to the document's type.

    - The output file is named based on the "source_id" field of the document's
      source metadata, which is cleaned to ensure it is a valid filename.

    Examples
    --------
    Suppose `response` contains the following structure:

    >>> response = {
    ...     "data": [
    ...         {"metadata": {"source_metadata": {"source_id": "doc_1.txt"}}},
    ...         {"metadata": {"source_metadata": {"source_id": "doc_2.txt"}}}
    ...     ]
    ... }
    >>> output_directory = "output"
    >>> save_response_data(response, output_directory)

    In this example, the function will create directories under "output" for
    each document type and save corresponding metadata JSON files within those
    directories.

    See Also
    --------
    get_valid_filename : Cleans and validates the filename based on the source ID.
    organize_documents_by_type : Organizes the documents by their type for structured saving.
    """

    if ("data" not in response) or (not response["data"]):
        return

    response_data = response["data"]

    if not isinstance(response_data, list) or len(response_data) == 0:
        return

    doc_meta_base = response_data[0]["metadata"]
    source_meta = doc_meta_base["source_metadata"]
    doc_name = source_meta["source_id"]
    clean_doc_name = get_valid_filename(os.path.basename(doc_name))
    output_name = f"{clean_doc_name}.metadata.json"

    doc_map = organize_documents_by_type(response_data)
    for doc_type, documents in doc_map.items():
        doc_type_path = os.path.join(output_directory, doc_type)
        if not os.path.exists(doc_type_path):
            os.makedirs(doc_type_path)

        with open(os.path.join(doc_type_path, output_name), "w") as f:
            f.write(json.dumps(documents, indent=2))


def create_job_specs_for_batch(files_batch: List[str], tasks: Dict[str, Any], client: NvIngestClient) -> Tuple[List[JobSpec], List[str]]:
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
        "extract", "store", "caption", "dedup", "filter", "embed", and "vdb_upload".

    client : NvIngestClient
        An instance of NvIngestClient, which handles the job submission. The `add_job` method of
        the client is used to submit each job specification.

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
    >>> job_ids = create_job_specs_for_batch(files_batch, tasks, client)
    >>> print(job_ids)
    ['job_12345', 'job_67890']

    In this example, jobs are created and submitted for the files in `files_batch`, with the
    tasks in `tasks` being added to each job specification. The returned job IDs are then
    printed.

    See Also
    --------
    extract_file_content : Function that extracts the content and type of a file.
    JobSpec : The class representing a job specification.
    NvIngestClient : Client class used to submit jobs to a job processing system.
    """

    job_specs = []
    job_ids = []
    for file_name in files_batch:
        try:
            file_content, file_type = extract_file_content(file_name)  # Assume these are defined
            file_type = file_type.value
        except ValueError as ve:
            logger.error(f"Error extracting content from {file_name}: {ve}")
            continue

        job_spec = JobSpec(
            document_type=file_type,
            payload=file_content,
            source_id=file_name,
            source_name=file_name,
            extended_options={"tracing_options": {"trace": True, "ts_send": time.time_ns()}},
        )

        logger.debug(f"Tasks: {tasks.keys()}")
        for task in tasks:
            logger.debug(f"Task: {task}")

        # TODO(Devin): Formalize this later, don't have time right now.
        if "split" in tasks:
            job_spec.add_task(tasks["split"])

        if f"extract_{file_type}" in tasks:
            job_spec.add_task(tasks[f"extract_{file_type}"])

        if "store" in tasks:
            job_spec.add_task(tasks["store"])

        if "caption" in tasks:
            job_spec.add_task(tasks["caption"])

        if "dedup" in tasks:
            job_spec.add_task(tasks["dedup"])

        if "filter" in tasks:
            job_spec.add_task(tasks["filter"])

        if "embed" in tasks:
            job_spec.add_task(tasks["embed"])

        if "vdb_upload" in tasks:
            job_spec.add_task(tasks["vdb_upload"])

        job_id = client.add_job(job_spec)
        job_ids.append(job_id)
        job_specs.append(job_spec)

    return job_specs, job_ids


def generate_job_batch_for_iteration(
        client: Any,
        pbar: Any,
        files: List[str],
        tasks: Dict,
        processed: int,
        batch_size: int,
        retry_job_ids: List[str],
        fail_on_error: bool = False
) -> Tuple[List[str], Dict[str, str], int]:
    """
    Generates a batch of job specifications for the current iteration of file processing. This function handles retrying failed jobs and creating new jobs for unprocessed files. The job specifications are then submitted for processing.

    Parameters
    ----------
    client : Any
        The client object used to submit jobs asynchronously.
    pbar : Any
        The progress bar object to update the progress as jobs are processed.
    files : List[str]
        The list of file paths to be processed.
    tasks : Dict
        A dictionary of tasks to be executed as part of the job specifications.
    processed : int
        The number of files that have been processed so far.
    batch_size : int
        The maximum number of jobs to process in one batch.
    retry_job_ids : List[str]
        A list of job IDs that need to be retried due to previous failures.
    fail_on_error : bool, optional
        Whether to raise an error and stop processing if job specifications are missing, by default False.

    Returns
    -------
    Tuple[List[str], Dict[str, str], int]
        A tuple containing:
        - job_ids (List[str]): The list of job IDs created or retried in this iteration.
        - job_id_map_updates (Dict[str, str]): A dictionary mapping job IDs to their corresponding file names.
        - processed (int): The updated number of files processed.

    Raises
    ------
    RuntimeError
        If `fail_on_error` is True and there are missing job specifications, a RuntimeError is raised.
    """

    job_ids = []
    job_id_map_updates = {}
    cur_job_count = 0

    if retry_job_ids:
        job_ids.extend(retry_job_ids)
        cur_job_count = len(job_ids)

    if (cur_job_count < batch_size) and (processed < len(files)):
        new_job_count = min(batch_size - cur_job_count, len(files) - processed)
        batch_files = files[processed: processed + new_job_count]  # noqa: E203

        new_job_ids = create_job_specs_for_batch(batch_files, tasks, client)
        if len(new_job_ids) != new_job_count:
            missing_jobs = new_job_count - len(new_job_ids)
            error_msg = f"Missing {missing_jobs} job specs -- this is likely due to bad reads or file corruption"
            logger.warning(error_msg)

            if fail_on_error:
                raise RuntimeError(error_msg)

            pbar.update(missing_jobs)

        job_id_map_updates = {job_id: file for job_id, file in zip(new_job_ids, batch_files)}
        processed += new_job_count
        _ = client.submit_job_async(new_job_ids, "morpheus_task_queue")
        job_ids.extend(new_job_ids)

    return job_ids, job_id_map_updates, processed


def handle_future_result(
        future: concurrent.futures.Future,
        futures_dict: Dict[concurrent.futures.Future, str],
) -> Dict[str, Any]:
    """
    Handle the result of a completed future job, process annotations, and save the result.

    This function processes the result of a future, extracts annotations (if any), logs them,
    checks the validity of the ingest result, and optionally saves the result to the provided
    output directory. If the result indicates a failure, a retry list of job IDs is prepared.

    Parameters
    ----------
    future : concurrent.futures.Future
        A future object representing an asynchronous job. The result of this job will be
        processed once it completes.

    futures_dict : Dict[concurrent.futures.Future, str]
        A dictionary mapping future objects to job IDs. The job ID associated with the
        provided future is retrieved from this dictionary.

    Returns
    -------
    Dict[str, Any]

    Raises
    ------
    RuntimeError
        If the job result is invalid, this exception is raised with a description of the failure.

    Notes
    -----
    - The `future.result()` is assumed to return a tuple where the first element is the actual
      result (as a dictionary), and the second element (if present) can be ignored.
    - Annotations in the result (if any) are logged for debugging purposes.
    - The `check_ingest_result` function (assumed to be defined elsewhere) is used to validate
      the result. If the result is invalid, a `RuntimeError` is raised.
    - The function handles saving the result data to the specified output directory using the
      `save_response_data` function.

    Examples
    --------
    Suppose we have a future object representing a job, a dictionary of futures to job IDs,
    and a directory for saving results:

    >>> future = concurrent.futures.Future()
    >>> futures_dict = {future: "job_12345"}
    >>> job_id_map = {"job_12345": {...}}
    >>> output_directory = "/path/to/save"
    >>> result, retry_job_ids = handle_future_result(future, futures_dict, job_id_map, output_directory)

    In this example, the function processes the completed job and saves the result to the
    specified directory. If the job fails, it raises a `RuntimeError` and returns a list of
    retry job IDs.

    See Also
    --------
    check_ingest_result : Function to validate the result of the job.
    save_response_data : Function to save the result to a directory.
    """

    try:
        result, _ = future.result()[0]
        if ("annotations" in result) and result["annotations"]:
            annotations = result["annotations"]
            for key, value in annotations.items():
                logger.debug(f"Annotation: {key} -> {json.dumps(value, indent=2)}")

        failed, description = check_ingest_result(result)

        if failed:
            raise RuntimeError(f"{description}")
    except Exception as e:
        logger.debug(f"Error processing future result: {e}")
        raise e

    return result


def create_and_process_jobs(
        files: List[str],
        client: NvIngestClient,
        tasks: Dict[str, Any],
        output_directory: str,
        batch_size: int,
        timeout: int = 10,
        fail_on_error: bool = False,
) -> Tuple[int, Dict[str, List[float]], int]:
    """
    Process a list of files, creating and submitting jobs for each file, then fetch and handle the results.

    This function creates job specifications (JobSpecs) for a list of files, submits the jobs to the
    client, and processes the results asynchronously. It handles job retries for timeouts, logs failures,
    and limits the number of JobSpecs in memory to `batch_size * 2`. Progress is reported on a per-file basis,
    including the pages processed per second.

    Parameters
    ----------
    files : List[str]
        A list of file paths to be processed. Each file is used to create a job, which is then submitted to the client.

    client : NvIngestClient
        An instance of NvIngestClient used to submit jobs and fetch results asynchronously.

    tasks : Dict[str, Any]
        A dictionary of tasks to be added to each job. The keys represent task names, and the values represent task
        configurations. Tasks may include "split", "extract", "store", "caption", etc.

    output_directory : str
        The directory path where the processed job results will be saved. If an empty string or None is provided,
        results will not be saved.

    batch_size : int
        The number of jobs to process in each batch. Memory is limited to `batch_size * 2` jobs at any time.

    timeout : int, optional
        The timeout in seconds for each job to complete before it is retried. Default is 10 seconds.

    fail_on_error : bool, optional
        If True, the function will raise an error and stop processing when encountering an unrecoverable error.
        If False, the function logs the error and continues processing other jobs. Default is False.

    Returns
    -------
    Tuple[int, Dict[str, List[float]], int]
        A tuple containing:
        - `total_files` (int): The total number of files processed.
        - `trace_times` (Dict[str, List[float]]): A dictionary mapping job IDs to a list of trace times for
          diagnostic purposes.
        - `total_pages_processed` (int): The total number of pages processed from the files.

    Raises
    ------
    RuntimeError
        If `fail_on_error` is True and an error occurs during job submission or processing, a `RuntimeError` is raised.

    Notes
    -----
    - The function limits the number of JobSpecs in memory to `batch_size * 2` for efficient resource management.
    - It manages job retries for timeouts and logs decoding or processing errors.
    - The progress bar reports progress on a per-file basis and shows the pages processed per second.

    Examples
    --------
    Suppose we have a list of files and tasks to process:

    >>> files = ["file1.txt", "file2.pdf", "file3.docx"]
    >>> client = NvIngestClient()
    >>> tasks = {"split": ..., "extract": ..., "store": ...}
    >>> output_directory = "/path/to/output"
    >>> batch_size = 5
    >>> total_files, trace_times, total_pages_processed = create_and_process_jobs(
    ...     files, client, tasks, output_directory, batch_size
    ... )
    >>> print(f"Processed {total_files} files, {total_pages_processed} pages.")

    In this example, the function processes the files, submits jobs, fetches results, handles retries for
    timeouts, and logs failures. The number of files and pages processed is printed.

    See Also
    --------
    generate_job_batch_for_iteration : Function to generate job batches for processing.
    handle_future_result : Function to process and handle the result of completed future jobs.
    """

    total_files = len(files)
    total_pages_processed = 0
    trace_times = defaultdict(list)
    failed_jobs = []
    retry_job_ids = []
    job_id_map = {}
    retry_counts = defaultdict(int)
    file_page_counts = {file: estimate_page_count(file) for file in files}

    start_time_ns = time.time_ns()
    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
        processed = 0
        while (processed < len(files)) or retry_job_ids:
            # Process new batch of files or retry failed job IDs
            job_ids, job_id_map_updates, processed = generate_job_batch_for_iteration(
                client, pbar, files, tasks, processed, batch_size, retry_job_ids, fail_on_error
            )
            job_id_map.update(job_id_map_updates)
            retry_job_ids = []

            futures_dict = client.fetch_job_result_async(job_ids, timeout=timeout, data_only=False)
            for future in as_completed(futures_dict.keys()):

                retry = False
                job_id = futures_dict[future]
                source_name = job_id_map[job_id]
                try:
                    future_response = handle_future_result(future, futures_dict)

                    if output_directory:
                        save_response_data(future_response, output_directory)

                    total_pages_processed += file_page_counts[source_name]
                    elapsed_time = (time.time_ns() - start_time_ns) / 1e9
                    pages_per_sec = total_pages_processed / elapsed_time if elapsed_time > 0 else 0
                    pbar.set_postfix(pages_per_sec=f"{pages_per_sec:.2f}")

                    process_response(future_response, trace_times)

                except TimeoutError:
                    source_name = job_id_map[job_id]
                    retry_counts[source_name] += 1
                    retry_job_ids.append(job_id)  # Add job_id back to retry list
                    retry = True
                except json.JSONDecodeError as e:
                    source_name = job_id_map[job_id]
                    logger.error(f"Decoding while processing {job_id}({source_name}) {e}")
                    failed_jobs.append(f"{job_id}::{source_name}")
                except RuntimeError as e:
                    source_name = job_id_map[job_id]
                    logger.error(f"Error while processing {job_id}({source_name}) {e}")
                    failed_jobs.append(f"{job_id}::{source_name}")
                except Exception as e:
                    source_name = job_id_map[job_id]
                    logger.error(f"Unhandled error while processing {job_id}({source_name}) {e}")
                    failed_jobs.append(f"{job_id}::{source_name}")
                finally:
                    # Don't update progress bar if we're going to retry the job
                    if not retry:
                        pbar.update(1)

    return total_files, trace_times, total_pages_processed


def get_valid_filename(name):
    """
    Taken from https://github.com/django/django/blob/main/django/utils/text.py.
    Return the given string converted to a string that can be used for a clean
    filename. Remove leading and trailing spaces; convert other spaces to
    underscores; and remove anything that is not an alphanumeric, dash,
    underscore, or dot.
    >>> get_valid_filename("john's portrait in 2004.jpg")
    'johns_portrait_in_2004.jpg'
    """
    s = str(name).strip().replace(" ", "_")
    s = re.sub(r"(?u)[^-\w.]", "", s)
    if s in {"", ".", ".."}:
        raise ValueError("Could not derive file name from '%s'" % name)
    return s
