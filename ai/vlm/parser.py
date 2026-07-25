import json
import re
import logging

logger = logging.getLogger(__name__)


class MetadataParser:

    @staticmethod
    def extract_json(response: str):

        if not response:
            return None

        try:
            start = response.find("{")
            end = response.rfind("}") + 1

            if start == -1 or end == 0:
                return None

            return json.loads(response[start:end])

        except Exception as e:
            logger.error(e)
            return None

    @staticmethod
    def repair_json(response: str):

        if not response:
            return None

        try:
            text = response.strip()

            text = re.sub(r"```json", "", text)
            text = re.sub(r"```", "", text)

            start = text.find("{")
            end = text.rfind("}") + 1

            if start == -1 or end == 0:
                return None

            text = text[start:end]

            text = re.sub(r",\s*}", "}", text)
            text = re.sub(r",\s*]", "]", text)

            return json.loads(text)

        except Exception as e:
            logger.error(e)
            return None

    @staticmethod
    def parse(response: str):

        data = MetadataParser.extract_json(response)

        if data is not None:
            return data

        return MetadataParser.repair_json(response)

    @staticmethod
    def parse_batch(responses):

        parsed = []

        for response in responses:
            parsed.append(MetadataParser.parse(response))

        return parsed