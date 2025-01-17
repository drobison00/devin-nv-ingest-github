# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import functools
import pandas as pd
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import tritonclient.grpc as grpcclient
from morpheus.config import Config
from nv_ingest.schemas.audio_extractor_schema import AudioExtractorSchema
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage

import sys
sys.path.append('../../..')

from nv_ingest.util.nim.helpers import call_audio_inference_model, create_inference_client
from nv_ingest.util.nim.helpers import get_version

logger = logging.getLogger(f"morpheus.{__name__}")


def _update_metadata(row: pd.Series, audio_client: Any, audio_version: Any, trace_info: Dict) -> Dict:
    """
    Modifies the metadata of a row if the conditions for table extraction are met.

    Parameters
    ----------
    row : pd.Series
        A row from the DataFrame containing metadata for the audio extraction.

    audio_client : Any
        The client used to call the audio inference model.

    trace_info : Dict
        Trace information used for logging or debugging.

    Returns
    -------
    Dict
        The modified metadata if conditions are met, otherwise the original metadata.

    Raises
    ------
    ValueError
        If critical information (such as metadata) is missing from the row.
    """


    metadata = row.get("metadata")
    
    if metadata is None:
        logger.error("Row does not contain 'metadata'.")
        raise ValueError("Row does not contain 'metadata'.")

    content_metadata = metadata.get("content_metadata", {})

    # Only modify if content type is audio
    if content_metadata.get("type") != "audio" :
        return metadata

    source_metadata = metadata.get("source_metadata")
    audio_id = source_metadata['source_id']
    
    content_metadata = metadata.get("content_metadata")
    content_metadata = content_metadata['content']
    audio_content = content_metadata['content']
    

    # Modify audio metadata with the result from the inference model
    try:
        audio_result = call_audio_inference_model(audio_client, audio_content, audio_id, trace_info=trace_info)
        print(audio_result)
        metadata['audio_metadata'] = {'content': audio_result}
    except Exception as e:
        logger.error(f"Unhandled error calling audio inference model: {e}", exc_info=True)
        raise
 
    return metadata


def _transcribe_audio(df: pd.DataFrame, task_props: Dict[str, Any],
                        validated_config: Any, trace_info: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Extracts audio data from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the content from which audio data is to be extracted.

    task_props : Dict[str, Any]
        Dictionary containing task properties and configurations.

    validated_config : Any
        The validated configuration object for audio extraction.

    trace_info : Optional[Dict], optional
        Optional trace information for debugging or logging. Defaults to None.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        A tuple containing the updated DataFrame and the trace information.

    Raises
    ------
    Exception
        If any error occurs during the audio data extraction process.
    """

    #port = 32783
    #audio_client = create_inference_client(
    #    (None, f'http://0.0.0.0:{port}/v1/transcribe'),
    #    None,
    #    "http"
    #)

   
    audio_client = create_inference_client(
        validated_config.stage_config.audio_endpoints,
        None,
        "http"
    )    

    if trace_info is None:
        trace_info = {}
        logger.debug("No trace_info provided. Initialized empty trace_info dictionary.")

    try:
        # Apply the _update_metadata function to each row in the DataFrame
        #audio_version = get_version(validated_config.stage_config.audio_endpoints[1])
        audio_version = get_version(f'http://audio:{port}')
        df["metadata"] = df.apply(_update_metadata, axis=1, args=(audio_client, audio_version, trace_info))
        
        return df, trace_info

    except Exception as e:
        logger.error("Error occurred while extracting audio data.", exc_info=True)
        raise


def generate_audio_extractor_stage(
        c: Config,
        stage_config: Dict[str, Any],
        task: str = "audio_data_extract",
        task_desc: str = "audio_data_extraction",
        pe_count: int = 1,
):
    """
    Generates a multiprocessing stage to perform audio data extraction.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object.

    stage_config : Dict[str, Any]
        Configuration parameters for the audio content extractor, passed as a dictionary
        validated against the `AudioExtractorSchema`.

    task : str, optional
        The task name for the stage worker function, defining the specific audio extraction process.
        Default is "audio_data_extract".

    task_desc : str, optional
        A descriptor used for latency tracing and logging during audio extraction.
        Default is "audio_data_extraction".

    pe_count : int, optional
        The number of process engines to use for audio data extraction. This value controls
        how many worker processes will run concurrently. Default is 1.

    Returns
    -------
    MultiProcessingBaseStage
        A configured Morpheus stage with an applied worker function that handles audio data extraction
        from PDF content.
    """

    validated_config = AudioExtractorSchema(**stage_config)
    _wrapped_process_fn = functools.partial(_transcribe_audio, validated_config=validated_config)

    return MultiProcessingBaseStage(
        c=c, 
        pe_count=pe_count, 
        task=task, 
        task_desc=task_desc, 
        process_fn=_wrapped_process_fn,
        document_type="regex:^(mp3|wav)$",
    )



if __name__ == "__main__":
    metadata = {
        "source_metadata": {
            "access_level": 1,
            "collection_id": "",
            "date_created": "2024-11-04T12:29:08",
            "last_modified": "2024-11-04T12:29:08",
            "partition_id": -1,
            "source_id": "https://audio.listennotes.com/e/p/3946bc3aba1f425f8b2e146f0b3f72fc/",
            "source_location": "",
            "source_type": "wav",
            "summary": ""
        },

        "content_metadata": {
            "description": "Audio wav file",
            "type": "audio",
            "content": ''
        }
    }


    metadata = {
        "source_metadata": {
            "access_level": 1,
            "collection_id": "",
            "date_created": "2024-11-04T12:29:08",
            "last_modified": "2024-11-04T12:29:08",
            "partition_id": -1,
            "source_id": "test.mp3",
            "source_location": "",
            "source_type": "mp3",
            "summary": ""
        },

        "content_metadata": {
            "description": "Audio wav file",
            "type": "audio",
            "content": 'some base64 string'
        }
    }
    
    

    data = [{"metadata": metadata}]
    df = pd.DataFrame(data)

    df.to_csv('test.csv', index=False)
    
    df_result, _ = _transcribe_audio(df)

    df_result.to_csv('result.csv', index=False)


        
    print("Done!")