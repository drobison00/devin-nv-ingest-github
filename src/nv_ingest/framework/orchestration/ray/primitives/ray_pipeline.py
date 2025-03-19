# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from nv_ingest.framework.orchestration.ray.edges.async_queue_edge import AsyncQueueEdge

from typing import Any, Dict, List
import ray


@dataclass
class StageInfo:
    """
    Information about a pipeline stage.

    Attributes
    ----------
    name : str
        Name of the stage.
    callable : Any
        A Ray remote actor class that implements the stage.
    config : Dict[str, Any]
        Configuration parameters for the stage.
    is_source : bool, optional
        Whether the stage is a source.
    is_sink : bool, optional
        Whether the stage is a sink.
    """

    name: str
    callable: Any  # Already a remote actor class
    config: Dict[str, Any]
    is_source: bool = False
    is_sink: bool = False


class RayPipeline:
    """
    A structured pipeline that supports source, intermediate, and sink stages.
    Stages are connected using AsyncQueueEdge actors.

    Attributes
    ----------
    stages : List[StageInfo]
        List of stage definitions.
    connections : Dict[str, List[tuple]]
        Mapping from stage name to a list of tuples (destination stage name, queue size).
    stage_actors : Dict[str, List[Any]]
        Mapping from stage name to a list of instantiated Ray actor handles.
    edge_queues : Dict[str, Any]
        Mapping from edge name to an AsyncQueueEdge actor.
    """

    def __init__(self) -> None:
        self.stages: List[StageInfo] = []
        self.connections: Dict[str, List[tuple]] = {}  # from_stage -> list of (to_stage, queue_size)
        self.stage_actors: Dict[str, List[Any]] = {}
        self.edge_queues: Dict[str, Any] = {}  # edge_name -> AsyncQueueEdge actor

    def add_source(self, name: str, source_actor: Any, **config: Any) -> "RayPipeline":
        self.stages.append(StageInfo(name=name, callable=source_actor, config=config, is_source=True))
        return self

    def add_stage(self, name: str, stage_actor: Any, **config: Any) -> "RayPipeline":
        self.stages.append(StageInfo(name=name, callable=stage_actor, config=config))
        return self

    def add_sink(self, name: str, sink_actor: Any, **config: Any) -> "RayPipeline":
        self.stages.append(StageInfo(name=name, callable=sink_actor, config=config, is_sink=True))
        return self

    def make_edge(self, from_stage: str, to_stage: str, queue_size: int = 100) -> "RayPipeline":
        if from_stage not in [s.name for s in self.stages]:
            raise ValueError(f"Stage {from_stage} not found")
        if to_stage not in [s.name for s in self.stages]:
            raise ValueError(f"Stage {to_stage} not found")
        self.connections.setdefault(from_stage, []).append((to_stage, queue_size))
        return self

    def build(self) -> Dict[str, List[Any]]:
        # Instantiate stage actors.
        for stage in self.stages:
            actor = stage.callable.options(name=stage.name).remote(**stage.config)
            self.stage_actors[stage.name] = [actor]

        # Wire up edges using AsyncQueueEdge actors.
        for from_stage, conns in self.connections.items():
            for to_stage, queue_size in conns:
                queue_name = f"{from_stage}_to_{to_stage}"
                # Create an AsyncQueueEdge actor with the specified queue_size.
                edge_actor = AsyncQueueEdge.options(name=queue_name).remote(
                    max_size=queue_size, multi_reader=True, multi_writer=True
                )
                self.edge_queues[queue_name] = edge_actor

                # Set the from_stage's output edge to this edge.
                for actor in self.stage_actors[from_stage]:
                    ray.get(actor.set_output_edge.remote(edge_actor))
                # Set the to_stage's input edge to this edge.
                for actor in self.stage_actors[to_stage]:
                    ray.get(actor.set_input_edge.remote(edge_actor))
        return self.stage_actors

    def start(self) -> None:
        """
        Start the pipeline by starting all stage actors.
        """
        for stage in self.stages:
            for actor in self.stage_actors.get(stage.name, []):
                if hasattr(actor, "start"):
                    ray.get(actor.start.remote())

    def stop(self) -> None:
        """
        Stop the pipeline by stopping all stage actors.
        """
        for stage in self.stages:
            for actor in self.stage_actors.get(stage.name, []):
                if hasattr(actor, "stop"):
                    ray.get(actor.stop.remote())
