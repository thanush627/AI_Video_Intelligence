import json
import os
import re
from pathlib import Path
from typing import Dict, Optional


class Phase9VoiceSearch:
    def __init__(self, output_dir: str = "outputs/phase9"):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def recognize_speech(self, audio_path: str) -> str:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        return "recognized_text_placeholder"

    def detect_language(self, text: str) -> str:
        text = (text or "").strip().lower()
        if any(token in text for token in ["person", "car", "blue", "shirt"]):
            return "en"
        return "unknown"

    def translate_to_english(self, text: str, source_lang: str = "unknown") -> str:
        return text

    def normalize_query(self, query: str) -> str:
        query = query.strip().lower()
        query = re.sub(r"\s+", " ", query)
        return query

    def process_query(self, query: Optional[str] = None, audio_path: Optional[str] = None) -> Dict[str, object]:
        if audio_path:
            recognized_text = self.recognize_speech(audio_path)
        elif query:
            recognized_text = query
        else:
            raise ValueError("Provide either query or audio_path")

        lang = self.detect_language(recognized_text)
        translated = self.translate_to_english(recognized_text, lang)
        normalized = self.normalize_query(translated)

        result = {
            "recognized_text": recognized_text,
            "detected_language": lang,
            "translated_query": translated,
            "normalized_query": normalized,
        }

        with open(self.output_dir / "voice_query_log.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result
