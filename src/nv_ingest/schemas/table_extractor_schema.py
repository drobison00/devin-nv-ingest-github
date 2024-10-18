# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging

from typing import Optional, Tuple
from pydantic import BaseModel, root_validator

logger = logging.getLogger(__name__)


class TableExtractorConfigSchema(BaseModel):
    """
    Configuration schema for the table extraction stage settings.

    Parameters
    ----------
    auth_token : Optional[str], default=None
        Authentication token required for secure services.

    yolox_endpoints : Tuple[Optional[str], Optional[str]], default=(None, None)
        A tuple containing the gRPC and HTTP services for the yolox endpoint.
        Either the gRPC or HTTP service can be empty, but not both.

    Methods
    -------
    validate_endpoints(values)
        Validates that at least one of the gRPC or HTTP services is provided for the yolox endpoint.

    Raises
    ------
    ValueError
        If both gRPC and HTTP services are empty for the yolox endpoint.

    Config
    ------
    extra : str
        Pydantic config option to forbid extra fields.
    """

    auth_token: Optional[str] = None

    yolox_endpoints: Tuple[Optional[str], Optional[str]] = (None, None)

    @root_validator(pre=True)
    def validate_endpoints(cls, values):
        """
        Validates the gRPC and HTTP services for the yolox endpoint.

        Parameters
        ----------
        values : dict
            Dictionary containing the values of the attributes for the class.

        Returns
        -------
        dict
            The validated dictionary of values.

        Raises
        ------
        ValueError
            If both gRPC and HTTP services are empty for the yolox endpoint.
        """

        def clean_service(service):
            """Set service to None if it's an empty string or contains only spaces or quotes."""
            if service is None or not service.strip() or service.strip(" \"'") == "":
                return None
            return service

        grpc_service, http_service = values.get("yolox_endpoints", (None, None))
        grpc_service = clean_service(grpc_service)
        http_service = clean_service(http_service)

        if not grpc_service and not http_service:
            raise ValueError("Both gRPC and HTTP services cannot be empty for yolox_endpoints.")

        values["yolox_endpoints"] = (grpc_service, http_service)

        return values

    class Config:
        extra = "forbid"


class TableExtractorSchema(BaseModel):
    """
    Configuration schema for the table extraction processing settings.

    Parameters
    ----------
    max_queue_size : int, default=1
        The maximum number of items allowed in the processing queue.

    n_workers : int, default=2
        The number of worker threads to use for processing.

    raise_on_failure : bool, default=False
        A flag indicating whether to raise an exception if a failure occurs during table extraction.

    stage_config : Optional[TableExtractorConfigSchema], default=None
        Configuration for the table extraction stage, including yolox service endpoints.
    """

    max_queue_size: int = 1
    n_workers: int = 2
    raise_on_failure: bool = False

    stage_config: Optional[TableExtractorConfigSchema] = None

    class Config:
        extra = "forbid"