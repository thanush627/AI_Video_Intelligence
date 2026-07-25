import json
import logging
import torch

from transformers import AutoProcessor, AutoModelForCausalLM

from .prompt_templates import build_user_prompt

logger = logging.getLogger(__name__)


class FlorenceVL:

    def __init__(
        self,
        model_name="microsoft/Florence-2-large",
        device=None
    ):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info(f"Loading {model_name}...")

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device)

        self.model.eval()

    def predict(self, image):

        prompt = build_user_prompt()

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        response = self.processor.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0]

        return response

    def predict_batch(self, images):

        results = []

        for image in images:
            try:
                results.append(self.predict(image))
            except Exception as e:
                logger.error(e)
                results.append(None)

        return results

    @staticmethod
    def parse_json(response):

        try:
            start = response.index("{")
            end = response.rindex("}") + 1
            return json.loads(response[start:end])

        except Exception:
            return None