# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime

import pandas as pd
import pydantic
import pytest
from unittest.mock import patch

from nv_ingest.primitives.ingest_control_message import IngestControlMessage
from ....import_checks import MORPHEUS_IMPORT_OK

if MORPHEUS_IMPORT_OK:
    from nv_ingest.modules.sources.message_broker_task_source import process_message

MODULE_UNDER_TEST = "nv_ingest.modules.sources.message_broker_task_source"


@pytest.fixture
def job_payload():
    return json.dumps(
        {
            "job_payload": {
                "content": ["sample content"],
                "source_name": ["source1"],
                "source_id": ["id1"],
                "document_type": ["pdf"],
            },
            "job_id": "12345",
            "tasks": [
                {
                    "type": "split",
                    "task_properties": {
                        "split_by": "word",
                        "split_length": 100,
                        "split_overlap": 0,
                    },
                },
                {
                    "type": "extract",
                    "task_properties": {
                        "document_type": "pdf",
                        "method": "OCR",
                        "params": {},
                    },
                },
                {"type": "embed", "task_properties": {}},
            ],
        }
    )


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
def test_process_message_valid_job(job_payload):
    """
    Test that process_message processes a valid job correctly using IngestControlMessage
    and storing a pandas DataFrame directly.
    """
    job = json.loads(job_payload)
    ts_fetched = datetime.now()

    result = process_message(job, ts_fetched)

    # Check that result is an instance of IngestControlMessage
    assert isinstance(result, IngestControlMessage)

    # Check that the metadata is set correctly
    assert result.get_metadata("job_id") == "12345"
    assert result.get_metadata("response_channel") == "12345"

    # Check that tasks are added
    expected_tasks = job["tasks"]
    tasks_in_message = list(result.get_tasks())
    assert len(tasks_in_message) == len(expected_tasks)

    # Check that the payload is set correctly and is a pandas DataFrame
    payload_df = result.payload()
    assert isinstance(payload_df, pd.DataFrame)

    for column in job["job_payload"]:
        assert column in payload_df.columns
        assert payload_df[column].to_list() == job["job_payload"][column]

    # Since do_trace_tagging is False by default
    assert result.get_metadata("config::add_trace_tagging") is None


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
def test_process_message_missing_job_id(job_payload):
    """
    Test that process_message raises an exception when 'job_id' is missing.
    """
    job = json.loads(job_payload)
    job.pop("job_id")
    ts_fetched = datetime.now()

    with patch(f"{MODULE_UNDER_TEST}.validate_ingest_job") as mock_validate_ingest_job:
        mock_validate_ingest_job.side_effect = KeyError("job_id")

        with pytest.raises(KeyError) as exc_info:
            process_message(job, ts_fetched)

        assert "job_id" in str(exc_info.value)
        mock_validate_ingest_job.assert_called_once_with(job)


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
def test_process_message_missing_job_payload(job_payload):
    """
    Test that process_message handles a job missing 'job_payload'.
    """
    job = json.loads(job_payload)
    job.pop("job_payload")
    ts_fetched = datetime.now()

    with pytest.raises(pydantic.ValidationError):
        process_message(job, ts_fetched)


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
def test_process_message_invalid_tasks(job_payload):
    """
    Test that process_message raises an exception when a task is invalid (missing 'type').
    """
    job = json.loads(job_payload)
    job["tasks"][0].pop("type")  # Make the first task invalid
    ts_fetched = datetime.now()

    with pytest.raises(Exception) as exc_info:
        process_message(job, ts_fetched)

    # Look for any validation or structure error message
    err_str = str(exc_info.value).lower()
    assert 'task must have a "type"' in err_str or "validation" in err_str


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
def test_process_message_with_tracing(job_payload):
    """
    Test that process_message adds tracing metadata when tracing options are enabled.
    """
    job = json.loads(job_payload)
    job["tracing_options"] = {
        "trace": True,
        "ts_send": int(datetime.now().timestamp() * 1e9),  # ts_send in nanoseconds
        "trace_id": "trace-123",
    }
    ts_fetched = datetime.now()

    MODULE_NAME = "message_broker_task_source"

    result = process_message(job, ts_fetched)
    assert isinstance(result, IngestControlMessage)

    # Check that tracing metadata were added
    assert result.get_metadata("config::add_trace_tagging") is True
    assert result.get_metadata("trace_id") == "trace-123"

    # Check timestamps
    assert result.get_timestamp(f"trace::entry::{MODULE_NAME}") is not None
    assert result.get_timestamp(f"trace::exit::{MODULE_NAME}") is not None
    assert result.get_timestamp("trace::entry::broker_source_network_in") is not None
    assert result.get_timestamp("trace::exit::broker_source_network_in") == ts_fetched
    assert result.get_timestamp("latency::ts_send") is not None


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
def test_process_message_exception_with_job_id(job_payload):
    """
    Test that process_message handles exceptions and sets metadata when 'job_id' is present.
    """
    job = json.loads(job_payload)
    ts_fetched = datetime.now()

    # Setting job_payload to None should raise a ValidationError from pydantic
    job["job_payload"] = None

    with pytest.raises(pydantic.ValidationError):
        process_message(job, ts_fetched)


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
def test_process_message_exception_without_job_id(job_payload):
    """
    Test that process_message raises an exception when 'job_id' is missing and an exception occurs.
    """
    job = json.loads(job_payload)
    job.pop("job_id")
    ts_fetched = datetime.now()

    # Setting job_payload to None should raise an exception
    job["job_payload"] = None

    with pytest.raises(Exception) as exc_info:
        process_message(job, ts_fetched)
    # Optionally check the exception details if desired
