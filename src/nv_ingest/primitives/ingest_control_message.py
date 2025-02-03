from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, Generator


class ControlMessageTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    id: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class IngestControlMessage:
    """
    A Python-based stub for a control message class that matches the `ControlMessage` interface.
    """

    def __init__(self, message=None):
        """
        Initialize the IngestControlMessage.

        Parameters
        ----------
        message : Optional[IngestControlMessage or other]
            An optional argument that could be used to initialize this instance
            from another control message or Python object.
        """
        # We store tasks in an internal dictionary keyed by the task's 'id'.
        self._tasks: Dict[str, ControlMessageTask] = {}

    def add_task(self, task: ControlMessageTask):
        """
        Add a task to the control message, keyed by the task's unique 'id'.

        Parameters
        ----------
        task : ControlMessageTask
            A validated Pydantic model containing task information.

        Raises
        ------
        ValueError
            If a task with the same 'id' already exists in this control message.
        """
        if task.id in self._tasks:
            raise ValueError(
                f"Task with id '{task.id}' already exists in this control message. "
                "Tasks must be unique per-control message."
            )
        self._tasks[task.id] = task

    def get_tasks(self) -> Generator[ControlMessageTask, None, None]:
        """
        Return all tasks as a generator.

        Yields
        ------
        ControlMessageTask
            Each task in the control message.
        """
        for task in self._tasks.values():
            yield task

    def has_task(self, task_id: str) -> bool:
        """
        Check if a task with the given ID exists in this control message.

        Parameters
        ----------
        task_id : str
            The ID of the task to check.

        Returns
        -------
        bool
            True if the task exists, False otherwise.
        """
        return task_id in self._tasks

    def config(self, config=None):
        """
        Configure the control message or retrieve its current configuration.

        Parameters
        ----------
        config : dict, optional
            A dictionary-like object with configuration data.

        Returns
        -------
        dict
            If called without arguments, returns the current configuration.
        """
        pass

    def copy(self):
        """
        Create a copy of this control message.

        Returns
        -------
        IngestControlMessage
            A new copy of the current control message.
        """
        pass

    def get_metadata(self, key=None, default_value=None):
        """
        Retrieve metadata for a given key. If no key is provided, return all metadata.

        Parameters
        ----------
        key : str, optional
            The key for which we want to retrieve metadata.
        default_value : Any, optional
            A default value to return if the key is not found.

        Returns
        -------
        Any
            The metadata associated with `key` or a dict of all metadata if `key` is None.
        """
        pass

    def filter_timestamp(self, regex_filter):
        """
        Retrieve timestamps matching a regex filter.

        Parameters
        ----------
        regex_filter : str
            The regex pattern to match against timestamps.

        Returns
        -------
        dict
            A dictionary of matching timestamp entries.
        """
        pass

    def get_timestamp(self, key, fail_if_nonexist=False):
        """
        Retrieve a timestamp for a given key.

        Parameters
        ----------
        key : str
            The key associated with a timestamp.
        fail_if_nonexist : bool, optional
            If True, raise an error if the timestamp doesn't exist. Otherwise, return None.

        Returns
        -------
        datetime or None
            The timestamp if found; otherwise None (or raises an error if fail_if_nonexist is True).
        """
        pass

    def get_timestamps(self):
        """
        Retrieve all timestamps.

        Returns
        -------
        dict
            A dictionary of all timestamps stored in this control message.
        """
        pass

    def set_timestamp(self, key, timestamp):
        """
        Set a timestamp for a given key.

        Parameters
        ----------
        key : str
            The key to associate with the timestamp.
        timestamp : datetime or str
            The timestamp to store.
        """
        pass

    def has_metadata(self, key):
        """
        Check if a specific metadata key exists.

        Parameters
        ----------
        key : str
            The key to check for existence.

        Returns
        -------
        bool
            True if the metadata key exists, False otherwise.
        """
        pass

    def list_metadata(self):
        """
        List all metadata keys.

        Returns
        -------
        list
            A list of all metadata keys in this control message.
        """
        pass

    def payload(self, meta=None):
        """
        Get or set the payload (MessageMeta) for this control message.

        Parameters
        ----------
        meta : Any, optional
            A Python object or MessageMeta-like object to set as the payload.

        Returns
        -------
        Any
            The current payload if meta is None; otherwise returns nothing.
        """
        pass

    def tensors(self, tensor=None):
        """
        Get or set the tensor memory (TensorMemory) for this control message.

        Parameters
        ----------
        tensor : Any, optional
            A TensorMemory-like object to set as the tensor data.

        Returns
        -------
        Any
            The current tensor data if tensor is None; otherwise returns nothing.
        """
        pass

    def remove_task(self, task_type):
        """
        Remove a task from the control message.

        Parameters
        ----------
        task_type : Any
            The task type to remove.
        """
        pass

    def set_metadata(self, key, value):
        """
        Set a metadata entry for a given key.

        Parameters
        ----------
        key : str
            The metadata key.
        value : Any
            The value to set for this key.
        """
        pass

    def task_type(self, task_type=None):
        """
        Get or set the primary task type for this control message.

        Parameters
        ----------
        task_type : Any, optional
            If provided, set the control message to this task type.

        Returns
        -------
        Any
            The current task type if called without an argument.
        """
        pass
