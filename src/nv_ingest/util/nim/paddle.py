import json
import logging
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import packaging
import pandas as pd
from sklearn.cluster import DBSCAN

from nv_ingest.schemas.metadata_schema import TableFormatEnum
from nv_ingest.util.image_processing.transforms import base64_to_numpy
from nv_ingest.util.nim.helpers import ModelInterface
from nv_ingest.util.nim.helpers import preprocess_image_for_paddle

logger = logging.getLogger(__name__)


class PaddleOCRModelInterface(ModelInterface):
    def __init__(
        self,
        paddle_version: Optional[str] = None,
        table_format: Optional[TableFormatEnum] = TableFormatEnum.PSEUDO_MARKDOWN,
    ):
        self.paddle_version = paddle_version
        self.table_format = table_format

    def prepare_data_for_inference(self, data):
        # Expecting base64_image in data
        base64_image = data["base64_image"]
        image_array = base64_to_numpy(base64_image)
        data["image_array"] = image_array

        # Cache image dimensions for computing bounding boxes.
        self._width, self._height = image_array.shape[:2]

        return data

    def format_input(self, data, protocol: str, **kwargs):
        image_data = data["image_array"]
        if protocol == "grpc":
            logger.debug("Formatting input for gRPC PaddleOCR model")
            image_data = preprocess_image_for_paddle(image_data, self.paddle_version)
            image_data = image_data.astype(np.float32)
            image_data = np.expand_dims(image_data, axis=0)

            return image_data
        elif protocol == "http":
            logger.debug("Formatting input for HTTP PaddleOCR model")
            # For HTTP, preprocessing is not necessary
            base64_img = data["base64_image"]
            payload = self._prepare_paddle_payload(base64_img)

            return payload
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def parse_output(self, response, protocol: str):
        if protocol == "grpc":
            logger.debug("Parsing output from gRPC PaddleOCR model")
            return self._extract_content_from_paddle_grpc_response(response)
        elif protocol == "http":
            logger.debug("Parsing output from HTTP PaddleOCR model")
            return self._extract_content_from_paddle_http_response(response)
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def process_inference_results(self, output, **kwargs):
        # For PaddleOCR, the output is the table content as a string
        return output

    def _is_version_early_access_legacy_api(self):
        return self.paddle_version and (
            packaging.version.parse(self.paddle_version) < packaging.version.parse("0.2.1-rc2")
        )

    def _prepare_paddle_payload(self, base64_img: str) -> Dict[str, Any]:
        image_url = f"data:image/png;base64,{base64_img}"

        if self._is_version_early_access_legacy_api():
            image = {"type": "image_url", "image_url": {"url": image_url}}
            message = {"content": [image]}
            payload = {"messages": [message]}
        else:
            image = {"type": "image_url", "url": image_url}
            payload = {"input": [image]}

        return payload

    def _extract_content_from_paddle_http_response(self, json_response):
        if "data" not in json_response or not json_response["data"]:
            raise RuntimeError("Unexpected response format: 'data' key is missing or empty.")

        if self._is_version_early_access_legacy_api():
            content = json_response["data"][0]["content"]
        else:
            text_detections = json_response["data"][0]["text_detections"]

            text_predictions = []
            bounding_boxes = []
            for text_detection in text_detections:
                text_predictions.append(text_detection["text_prediction"]["text"])
                bounding_boxes.append([(point["x"], point["y"]) for point in text_detection["bounding_box"]["points"]])

            if self.table_format == TableFormatEnum.SIMPLE:
                content = " ".join(text_predictions)
            elif self.table_format == TableFormatEnum.PSEUDO_MARKDOWN:
                content = self._convert_paddle_response_to_psuedo_markdown(bounding_boxes, text_predictions)
            else:
                raise ValueError(f"Unexpected table format: {self.table_format}")

        return content

    def _extract_content_from_paddle_grpc_response(self, response):
        # Convert bytes output to string
        if not isinstance(response, np.ndarray):
            raise ValueError("Unexpected response format: response is not a NumPy array.")

        if self._is_version_early_access_legacy_api():
            if response.dtype.type is np.object_ or response.dtype.type is np.bytes_:
                output_strings = [
                    output[0].decode("utf-8") if isinstance(output, bytes) else str(output) for output in response
                ]
                content = " ".join(output_strings)
            else:
                output_bytes = response.tobytes()
                output_str = output_bytes.decode("utf-8")
                content = output_str
        else:
            bboxes_bytestr, texts_bytestr, _ = response
            bounding_boxes = json.loads(bboxes_bytestr.decode("utf8"))[0]
            text_predictions = json.loads(texts_bytestr.decode("utf8"))[0]

            if self.table_format == TableFormatEnum.SIMPLE:
                content = " ".join(texts_predictions)
            elif self.table_format == TableFormatEnum.PSEUDO_MARKDOWN:
                content = self._convert_paddle_response_to_psuedo_markdown(bounding_boxes, text_predictions)

        return content

    def _convert_paddle_response_to_psuedo_markdown(self, bounding_boxes, text_predictions):
        bboxes = []
        texts = []
        for box, txt in zip(bounding_boxes, text_predictions):
            if box == "nan":
                continue
            points = []
            for point in box:
                # The coordinates from Paddle are normlized. Convert them back to integers for DBSCAN.
                x = float(point[0]) * self._width
                y = float(point[1]) * self._height
                points.append([x, y])
            bboxes.append(points)
            texts.append(txt)

        if (not bboxes) or (not texts):
            return ""

        bboxes = np.array(bboxes).astype(int)
        bboxes = bboxes.reshape(-1, 8)[:, [0, 1, 2, -1]]

        preds_df = pd.DataFrame(
            {"x0": bboxes[:, 0], "y0": bboxes[:, 1], "x1": bboxes[:, 2], "y1": bboxes[:, 3], "text": texts}
        )
        preds_df = preds_df.sort_values("y0")

        dbscan = DBSCAN(eps=10, min_samples=1)
        dbscan.fit(preds_df["y0"].values[:, None])

        preds_df["cluster"] = dbscan.labels_
        preds_df = preds_df.sort_values(["cluster", "x0"])

        results = ""
        for _, dfg in preds_df.groupby("cluster"):
            results += "| " + " | ".join(dfg["text"].values.tolist()) + " |\n"

        return results
