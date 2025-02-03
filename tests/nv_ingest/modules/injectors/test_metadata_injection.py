# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from nv_ingest.primitives.ingest_control_message import IngestControlMessage
from nv_ingest.modules.injectors.metadata_injector import on_data


def test_on_data_no_update_required():
    """
    Test that on_data does not modify the payload when each row already contains valid metadata.

    This test creates a DataFrame where the 'metadata' column is pre-populated. It verifies that
    calling on_data returns the same DataFrame without any modifications.
    """
    df = pd.DataFrame(
        {
            "document_type": ["pdf"],
            "content": ["some content"],
            "source_id": ["id1"],
            "source_name": ["source1"],
            "metadata": [
                {
                    "content": "preexisting content",
                    "content_metadata": {"type": "preexisting"},
                    "error_metadata": None,
                    "image_metadata": None,
                    "source_metadata": {"source_id": "id1", "source_name": "source1", "source_type": "pdf"},
                    "text_metadata": {"text_type": "document"},
                }
            ],
        }
    )
    msg = IngestControlMessage()
    msg.payload(df)
    result = on_data(msg)
    updated_df = result.payload()
    pd.testing.assert_frame_equal(updated_df, df)


def test_on_data_injects_metadata():
    """
    Test that on_data injects metadata when it is missing from the payload.

    For a document of type "pdf", this test now expects that:
      - content_metadata.type is set to "structured" (according to the current mapping)
      - image_metadata remains None
      - text_metadata is set to None (since a pdf is not mapped to TEXT)
    The test verifies that the missing metadata is injected correctly.
    """
    df = pd.DataFrame(
        {
            "document_type": ["pdf"],
            "content": ["new content"],
            "source_id": ["id2"],
            "source_name": ["source2"],
        }
    )
    msg = IngestControlMessage()
    msg.payload(df)
    result = on_data(msg)
    updated_df = result.payload()
    assert "metadata" in updated_df.columns
    meta = updated_df.loc[0, "metadata"]
    expected_content_metadata = {"type": "structured"}
    expected_source_metadata = {"source_id": "id2", "source_name": "source2", "source_type": "pdf"}
    assert meta["content"] == "new content"
    assert meta["content_metadata"] == expected_content_metadata
    assert meta["error_metadata"] is None
    assert meta["image_metadata"] is None
    assert meta["source_metadata"] == expected_source_metadata
    assert meta["text_metadata"] is None


def test_on_data_multiple_rows():
    """
    Test that on_data correctly processes a DataFrame with multiple rows where some rows already have metadata
    and others do not.

    The first row already contains metadata and should remain unchanged. The second row is missing metadata;
    for a document of type "png", it is expected that:
      - content_metadata.type is set to "image"
      - image_metadata is set to {"image_type": "png"}
      - text_metadata is None
    The test verifies that the first row's metadata remains intact and that the second row's metadata is injected as
    expected.
    """
    df = pd.DataFrame(
        {
            "document_type": ["pdf", "png"],
            "content": ["existing content", "new content"],
            "source_id": ["id1", "id3"],
            "source_name": ["source1", "source3"],
            "metadata": [
                {
                    "content": "existing content",
                    "content_metadata": {"type": "preexisting"},
                    "error_metadata": None,
                    "image_metadata": None,
                    "source_metadata": {"source_id": "id1", "source_name": "source1", "source_type": "pdf"},
                    "text_metadata": {"text_type": "document"},
                },
                {},
            ],
        }
    )
    msg = IngestControlMessage()
    msg.payload(df)
    result = on_data(msg)
    updated_df = result.payload()
    meta0 = updated_df.loc[0, "metadata"]
    expected_meta0 = {
        "content": "existing content",
        "content_metadata": {"type": "preexisting"},
        "error_metadata": None,
        "image_metadata": None,
        "source_metadata": {"source_id": "id1", "source_name": "source1", "source_type": "pdf"},
        "text_metadata": {"text_type": "document"},
    }
    assert meta0 == expected_meta0

    meta1 = updated_df.loc[1, "metadata"]
    expected_meta1 = {
        "content": "new content",
        "content_metadata": {"type": "image"},
        "error_metadata": None,
        "image_metadata": {"image_type": "png"},
        "source_metadata": {"source_id": "id3", "source_name": "source3", "source_type": "png"},
        "text_metadata": None,
    }
    assert meta1 == expected_meta1


def test_on_data_empty_payload():
    """
    Test that on_data correctly handles an empty payload.

    This test provides an empty DataFrame to on_data and verifies that the output remains an empty DataFrame.
    """
    df = pd.DataFrame()
    msg = IngestControlMessage()
    msg.payload(df)
    result = on_data(msg)
    updated_df = result.payload()
    pd.testing.assert_frame_equal(updated_df, df)
