import json
import logging
import torch

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig
)

from .prompt_templates import (
    SYSTEM_PROMPT,
    build_user_prompt
)

logger = logging.getLogger(__name__)


class QwenVL:

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        device=None
    ):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info(f"Loading {model_name} on {self.device}...")

        # Load Processor
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # 4-bit Quantization Configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Load Model (ONLY ONCE)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        self.model.eval()

        logger.info("Qwen2.5-VL loaded successfully.")

    def predict(self, image):

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {
                        "type": "text",
                        "text": build_user_prompt()
                    }
                ]
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt"
        )

        device = next(self.model.parameters()).device

        inputs = {
            k: v.to(device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        with torch.inference_mode():

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                use_cache=True
            )

        generated_ids = output_ids[
            :,
            inputs["input_ids"].shape[1]:
        ]

        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        print("\n" + "=" * 80)
        print("QWEN RESPONSE")
        print("=" * 80)
        print(response)
        print("=" * 80 + "\n")

        return response.strip()

    def predict_batch(self, images):

        responses = []

        for image in images:

            try:
                responses.append(self.predict(image))

            except Exception as e:

                logger.exception(e)
                responses.append(None)

        return responses

    @staticmethod
    def parse_json(response):

        if response is None:
            return None

        try:

            start = response.find("{")
            end = response.rfind("}") + 1

            if start == -1 or end == 0:
                return None

            return json.loads(response[start:end])

        except Exception:

            return None