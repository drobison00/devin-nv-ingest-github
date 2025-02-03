# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import traceback

import mrc
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from mrc.core import operators as ops

from nv_ingest.primitives.ingest_control_message import IngestControlMessage
from nv_ingest.schemas.job_counter_schema import JobCounterSchema
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.telemetry.global_stats import GlobalStats
from nv_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)

MODULE_NAME = "job_counter"
MODULE_NAMESPACE = "nv_ingest"

JobCounterLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE)


def _count_jobs(message: IngestControlMessage, stat_name: str, stats: GlobalStats) -> IngestControlMessage:
    """
    Increment the specified statistic in the global statistics structure. If `completed_jobs`
    is being incremented and the message has `cm_failed` set, then increment `failed_jobs`
    instead.

    Parameters
    ----------
    message : IngestControlMessage
        The control message being processed.
    stat_name : str
        The name of the stat to increment.
    stats : GlobalStats
        The global statistics instance.

    Returns
    -------
    IngestControlMessage
        The original message after incrementing the relevant statistic.

    Raises
    ------
    ValueError
        If any error occurs while incrementing the statistics.
    """
    try:
        logger.debug(f"Performing job counter on stat '{stat_name}'")
        if stat_name == "completed_jobs":
            if message.has_metadata("cm_failed") and message.get_metadata("cm_failed"):
                stats.increment_stat("failed_jobs")
            else:
                stats.increment_stat("completed_jobs")
            return message

        stats.increment_stat(stat_name)
        return message
    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Failed to run job counter: {e}")


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _job_counter(builder: mrc.Builder) -> None:
    """
    Module for counting submitted jobs and updating the global statistics. This module sets up
    a job counter that increments a specified statistic in the global statistics structure for
    each processed message.

    Parameters
    ----------
    builder : mrc.Builder
        The module configuration builder.
    """
    validated_config = fetch_and_validate_module_config(builder, JobCounterSchema)
    stats = GlobalStats.get_instance()

    @traceable(MODULE_NAME)
    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
        skip_processing_if_failed=False,
    )
    def _wrapped_count_jobs(message: IngestControlMessage) -> IngestControlMessage:
        """
        Wraps the global `count_jobs` function to pass in the validated config
        and global stats. This allows for direct testing of `count_jobs` without
        the overhead of pipeline operators.
        """
        return _count_jobs(message, validated_config.name, stats)

    job_counter_node = builder.make_node(f"{validated_config.name}_counter", ops.map(_wrapped_count_jobs))

    builder.register_module_input("input", job_counter_node)
    builder.register_module_output("output", job_counter_node)
