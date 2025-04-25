import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Iterator

import google.generativeai as genai
import ollama
import opencc
from dotenv import load_dotenv

load_dotenv()
logging.getLogger("httpx").setLevel(logging.WARNING)
os.environ["OLLAMA_TIMEOUT"] = "10"


class BaseTranslateEngine(ABC):
    def __init__(self, config: dict):
        self.src = config.get("source_lang", "English")
        self.dest = config.get("target_lang", "繁體中文")

        self.empty_timeout = float(config.get("empty_timeout", 5.0))
        self._last_non_empty = time.time()
        self._empty_emitted = False

    @abstractmethod
    def translate(self, text: str) -> str:
        pass

    def translate_stream(self, text_stream: Iterator[str]) -> Iterator[str]:
        for text in text_stream:
            now = time.time()
            if not text.strip():
                if (not self._empty_emitted) and (
                    (now - self._last_non_empty) >= self.empty_timeout
                ):
                    yield ""
                    self._empty_emitted = True
                continue
            self._last_non_empty = now
            self._empty_emitted = False
            yield self.translate(text)

class AITranslateEngine(BaseTranslateEngine):
    def __init__(self, config: dict):
        super().__init__(config)
        self.temperature = config.get("temperature", 0)
class GeminiTranslateEngine(AITranslateEngine):
    def __init__(self, config: dict):
        super().__init__(config)
        api_key = os.getenv("google_key")
        if not api_key:
            raise ValueError("Gemini API 金鑰未設定")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-lite")

    def translate(self, text: str) -> str:
        prompt = f"請將以下文字翻譯成 {self.dest}，並只給我翻譯內容回復，確保滿足以上條件再回覆：\n{text}"
        response = self.model.generate_content(prompt)
        return response.text.strip()


class OllamaTranslateEngine(AITranslateEngine):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model = config.get("model", "gemma3")

    def _build_prompt(self, text: str) -> str:
        return f"請將以下文字翻譯成 {self.dest}，並只給我翻譯內容回覆，確保滿足以上條件再回覆：\n{text}"

    def translate(self, text: str) -> str:
        prompt = self._build_prompt(text)
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature},
            )
            return response.message.content.strip()
        except Exception as e:
            raise RuntimeError(f"翻譯失敗: {e}")

class OpenCCTranslateEngine(BaseTranslateEngine):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model  = config.get("model ", "s2t")+".json"
        self.converter = opencc.OpenCC(self.model)

    def translate(self, text: str) -> str:
        try:
            return self.converter.convert(text)
        except Exception as e:
            raise RuntimeError(f"翻譯失敗: {e}")

class TranslateEngineFactory:
    @staticmethod
    def create(config: dict):
        engine_type = config.get("engine_type", "ollama")
        if engine_type == "gemini":
            return GeminiTranslateEngine(config)
        elif engine_type == "ollama":
            return OllamaTranslateEngine(config)
        elif engine_type == "opencc":
            return OpenCCTranslateEngine(config)
        else:
            raise ValueError(f"未知的翻譯引擎類型: {engine_type}")
