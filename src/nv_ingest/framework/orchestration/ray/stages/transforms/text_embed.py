# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any
import ray

# Assume these imports come from your project:
from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest_api.internal.primitives.ingest_control_message import remove_task_by_type
from nv_ingest_api.internal.schemas.transform.transform_text_embedding_schema import TextEmbeddingSchema
from nv_ingest_api.internal.transform.embed_text import transform_create_text_embeddings_internal
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_context_manager,
    unified_exception_handler,
)

logger = logging.getLogger(__name__)


@ray.remote
class TextEmbeddingTransformStage(RayActorStage):
    """
    A Ray actor stage that extracts text embeddings from a DataFrame payload.

    This stage uses the validated configuration (TextEmbeddingSchema) to process the DataFrame
    and generate text embeddings. The resulting DataFrame is set back on the message, and any
    trace or extraction metadata is added.
    """

    def __init__(self, config: TextEmbeddingSchema, progress_engine_count: int) -> None:
        super().__init__(config, progress_engine_count)
        try:
            self.validated_config = config
        except Exception as e:
            logger.exception("Error validating text embedding extractor config")
            raise e

    @filter_by_task(required_tasks=["embed"])
    @nv_ingest_node_failure_context_manager(annotation_id="text_embedding", raise_on_failure=False)
    @unified_exception_handler
    async def on_data(self, control_message: Any) -> Any:
        # Get the DataFrame payload.
        df_payload = control_message.payload()
        # Call the text embedding extraction function.
        task_config = remove_task_by_type(control_message, "embed")
        new_df, execution_trace_log = transform_create_text_embeddings_internal(
            df_payload, task_config=task_config, transform_config=self.validated_config
        )
        # Update the control message payload.
        control_message.payload(new_df)
        # Annotate the message metadata with trace info.
        control_message.set_metadata("text_embedding_trace", execution_trace_log)
        return control_message
