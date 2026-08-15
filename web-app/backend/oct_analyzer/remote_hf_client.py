import os
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Hugging Face Space Endpoints
HF_SEGMENTATION_SPACE = os.getenv("HF_SEGMENTATION_SPACE", "NMundhra/OCT-Segmentation-Model")
HF_CLASSIFIER_SPACE = os.getenv("HF_CLASSIFIER_SPACE", "NMundhra/OCT-Image-Classifier-Model")
HF_TOKEN = os.getenv("HF_TOKEN")

class RemoteHFSpaceClient:
    _instance: Optional["RemoteHFSpaceClient"] = None

    def __init__(self):
        self.enabled = os.getenv("OCT_REMOTE_OFFLOAD", "false").lower() == "true"
        self._seg_client = None
        self._cls_client = None

    @classmethod
    def get_instance(cls) -> "RemoteHFSpaceClient":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_seg_client(self):
        if self._seg_client is None:
            from gradio_client import Client
            logger.info(f"Connecting to remote HF Segmentation Space: {HF_SEGMENTATION_SPACE}")
            self._seg_client = Client(HF_SEGMENTATION_SPACE, hf_token=HF_TOKEN)
        return self._seg_client

    def _get_cls_client(self):
        if self._cls_client is None:
            from gradio_client import Client
            logger.info(f"Connecting to remote HF Classifier Space: {HF_CLASSIFIER_SPACE}")
            self._cls_client = Client(HF_CLASSIFIER_SPACE, hf_token=HF_TOKEN)
        return self._cls_client

    def predict_segmentation(self, image_path: str, model_id: int = 1) -> Any:
        client = self._get_seg_client()
        api_name = f"/predict_model{model_id}"
        logger.info(f"Offloading segmentation model {model_id} to HF Space...")
        return client.predict(image_path, api_name=api_name)

    def predict_classification(self, image_path: str) -> dict[str, Any]:
        client = self._get_cls_client()
        logger.info("Offloading classification request to HF Space...")
        return client.predict(image_path, api_name="/predict_multi_head")
