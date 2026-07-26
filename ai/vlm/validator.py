import logging

logger = logging.getLogger(__name__)


class MetadataValidator:

    REQUIRED_FIELDS = [
        "colors",
        "attributes",
        "action",
        "orientation",
        "visibility",
        "confidence"
    ]

    CONFIDENCE_FIELDS = [
        "colors",
        "attributes",
        "action"
    ]

    @classmethod
    def validate(cls, metadata):

        if metadata is None:
            return False

        for field in cls.REQUIRED_FIELDS:
            if field not in metadata:
                logger.warning(f"Missing field: {field}")
                return False

        if not isinstance(metadata["colors"], dict):
            return False

        if not isinstance(metadata["attributes"], list):
            return False

        if not isinstance(metadata["confidence"], dict):
            return False

        for field in cls.CONFIDENCE_FIELDS:
            if field not in metadata["confidence"]:
                return False

            value = metadata["confidence"][field]

            if not isinstance(value, (int, float)):
                return False

            if value < 0 or value > 1:
                return False

        return True

    @classmethod
    def filter_valid(cls, metadata_list):

        valid = []

        for metadata in metadata_list:
            if cls.validate(metadata):
                valid.append(metadata)

        return valid

    @classmethod
    def summary(cls, metadata_list):

        total = len(metadata_list)
        valid = len(cls.filter_valid(metadata_list))

        return {
            "total": total,
            "valid": valid,
            "invalid": total - valid
        }