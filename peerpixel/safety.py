"""Local output-image safety classification. No image leaves before this runs."""
from __future__ import annotations

import io
from PIL import Image

THRESHOLD = 0.65


class SafetyClassifier:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.classifier = None

    def warm(self):
        if self.classifier is not None:
            return
        from transformers import pipeline
        from . import model_cache

        path = self.model_path or model_cache.ensure_directory("nsfw-image-detection")
        self.classifier = pipeline("image-classification", model=path, device_map="auto")

    def classify(self, jpeg: bytes) -> dict:
        self.warm()
        results = self.classifier(Image.open(io.BytesIO(jpeg)).convert("RGB"))
        score = max((float(item["score"]) for item in results
                     if "nsfw" in str(item.get("label", "")).lower()), default=0.0)
        return {"label": "nsfw" if score > THRESHOLD else "normal", "nsfwScore": score}

    def unload(self):
        self.classifier = None
