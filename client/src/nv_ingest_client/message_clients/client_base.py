# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from abc import abstractmethod

from nv_ingest_client.primitives.jobs.job_spec import JobSpec
from nv_ingest_client.primitives.jobs.job_state import JobState


class MessageClientBase(ABC):
    """
    Abstract base class for a messaging client to interface with various messaging systems.

    Provides a standard interface for sending and receiving messages with connection management
    and retry logic.
    """

    @abstractmethod
    def __init__(
        self,
        host: str,
        port: int,
        db: int = 0,
        max_retries: int = 0,
        max_backoff: int = 32,
        connection_timeout: int = 300,
        max_pool_size: int = 128,
        use_ssl: bool = False,
    ):
        """
        Initialize the messaging client with connection parameters.
        """

    @abstractmethod
    def get_client(self):
        """
        Returns the client instance, reconnecting if necessary.

        Returns:
            The client instance.
        """

    @abstractmethod
    def ping(self) -> bool:
        """
        Checks if the server is responsive.

        Returns:
            True if the server responds to a ping, False otherwise.
        """

    @abstractmethod
    def fetch_message(self, job_state: JobState, timeout: float = 0) -> str:
        """
        Fetches a message from the specified queue with retries on failure.

        Parameters:
            job_state (JobState): The JobState of the message to be fetched.
            timeout (float): The timeout in seconds for blocking until a message is available.

        Returns:
            The fetched message, or None if no message could be fetched.
        """

    @abstractmethod
    def submit_message(self, channel_name: str, message: JobSpec) -> str:
        """
        Submits a message to a specified queue with retries on failure.

        Parameters:
            channel_name (str): The name of the queue to submit the message to.
            message (JobSpec): The message to submit.
        """
