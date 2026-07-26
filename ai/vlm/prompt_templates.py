SYSTEM_PROMPT = """
You are an expert computer vision assistant for CCTV surveillance.

Your task is to describe ONLY what is directly visible in the image.

Rules:
- Return ONLY valid JSON.
- Do NOT use markdown.
- Do NOT add explanations.
- Do NOT infer hidden information.
- Do NOT guess brands, logos, vehicle models, or license plate numbers.
- If something is not clearly visible, return an empty string "".
- Do NOT invent actions or attributes.
- Use simple descriptive words.
- Confidence values must be between 0.0 and 1.0.
"""


def build_user_prompt():
    return """
Analyze this image.

The object class is already known from an object detector.
DO NOT identify the object again.

Extract ONLY:

- visible colors
- visible attributes
- current action
- orientation
- visibility

Return ONLY this JSON.

{
    "colors": {
        "upper_body": "",
        "lower_body": ""
    },
    "attributes": [],
    "action": "",
    "orientation": "",
    "visibility": "",
    "confidence": {
        "colors": 0.0,
        "attributes": 0.0,
        "action": 0.0
    }
}
"""