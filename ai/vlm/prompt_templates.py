SYSTEM_PROMPT = """
You are an expert computer vision assistant.

Analyze the given image and return ONLY valid JSON.

Do not include markdown.
Do not include explanations.
Do not include extra text.
"""


def build_user_prompt():
    return """
Analyze the image and return this JSON only.

{
    "object": "",
    "colors": {
        "upper_body": "",
        "lower_body": ""
    },
    "attributes": [],
    "action": "",
    "orientation": "",
    "visibility": "",
    "confidence": {
        "object": 0.0,
        "colors": 0.0,
        "attributes": 0.0,
        "action": 0.0
    }
}
"""