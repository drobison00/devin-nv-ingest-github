# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import mrc
import pandas as pd
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module

# Use our IngestControlMessage instead of the Morpheus ControlMessage and MessageMeta.
from nv_ingest.primitives.ingest_control_message import IngestControlMessage

from nv_ingest.schemas import MetadataInjectorSchema
from nv_ingest.schemas.ingest_job_schema import DocumentTypeEnum
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.util.converters.type_mappings import doc_type_to_content_type
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)

MODULE_NAME = "metadata_injection"
MODULE_NAMESPACE = "nv_ingest"

MetadataInjectorLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE)


def on_data(message: IngestControlMessage):
    """
    Retrieve the payload as a pandas DataFrame, inject metadata into rows that lack it,
    and update the message payload if modifications were made.

    Parameters
    ----------
    message : IngestControlMessage
        The control message whose payload will be processed.

    Returns
    -------
    IngestControlMessage
        The updated control message.
    """
    # Get the payload as a pandas DataFrame
    df = message.payload()

    update_required = False
    rows = []
    for _, row in df.iterrows():
        content_type = doc_type_to_content_type(DocumentTypeEnum(row["document_type"]))
        if "metadata" not in row or "content" not in row["metadata"]:
            update_required = True
            row["metadata"] = {
                "content": row["content"],
                "content_metadata": {
                    "type": content_type.name.lower(),
                },
                "error_metadata": None,
                "image_metadata": (
                    None if content_type != ContentTypeEnum.IMAGE else {"image_type": row["document_type"]}
                ),
                "source_metadata": {
                    "source_id": row["source_id"],
                    "source_name": row["source_name"],
                    "source_type": row["document_type"],
                },
                "text_metadata": (None if (content_type != ContentTypeEnum.TEXT) else {"text_type": "document"}),
            }
        rows.append(row)

    if update_required:
        docs = pd.DataFrame(rows)
        # Directly update the payload with the new pandas DataFrame
        message.payload(docs)

    return message


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _metadata_injection(builder: mrc.Builder):
    """
    A module for injecting metadata into messages. It retrieves the message payload,
    processes it to ensure required metadata is present, and updates the payload if necessary.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus pipeline builder object.
    """
    validated_config = fetch_and_validate_module_config(builder, MetadataInjectorSchema)

    @traceable(MODULE_NAME)
    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
    )
    def _on_data(message: IngestControlMessage):
        return on_data(message)

    node = builder.make_node("metadata_injector", _on_data)

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
