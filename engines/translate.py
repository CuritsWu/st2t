import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Iterator

import google.generativeai as genai
import ollama
import opencc
import torch
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          M2M100ForConditionalGeneration, M2M100Tokenizer)

logging.getLogger("httpx").setLevel(logging.WARNING)


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
            yield text + "\n" + self.translate(text)


class AITranslateEngine(BaseTranslateEngine):
    def __init__(self, config: dict):
        super().__init__(config)
        self.temperature = config.get("temperature", 0)


class OllamaTranslateEngine(AITranslateEngine):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model = config.get("model", "gemma3")
        os.environ["OLLAMA_TIMEOUT"] = "10"
        self.topic: str = ""

    def translate(self, text: str) -> str:
        prompt = f"""將以下STT來源的文本翻譯成【目標語言】，如有需要請補足語意並修正錯誤，使語句自然通順。根據【主題（可選）】調整語氣。僅輸出翻譯結果，不附加任何說明。

【目標語言】：{self.dest}
【主題（可選）】：{self.topic}
【文本】：{text}"""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature},
            )
            return response.message.content.strip()
        except Exception as e:
            raise RuntimeError(f"翻譯失敗: {e}")


class NLLBTranslateEngine(AITranslateEngine):
    _LANG_CODE_MAP = {
        # 自行擴充 200 種 FLoRes 語言
        "英文": "eng_Latn",
        "日文": "jpn_Jpan",
        "繁體中文": "zho_Hant",
    }

    def _to_code(self, lang_name: str) -> str:
        try:
            return self._LANG_CODE_MAP[lang_name]
        except KeyError:
            raise ValueError(f"未定義語言代碼：{lang_name}")

    def __init__(self, config: dict):
        super().__init__(config)
        self.model_name = config.get("model", "facebook/nllb-200-distilled-600M")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.src_code = self._to_code(self.src)
        self.dest_code = self._to_code(self.dest)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, src_lang=self.src_code
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, device_map="auto"
        ).to(self.device)

        self.bos_id = self.tokenizer.convert_tokens_to_ids(self._to_code(self.dest))

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        sampling = self.temperature > 0
        gen_kwargs = {
            "max_length": 512,
            "forced_bos_token_id": self.bos_id,
            "do_sample": sampling,
            "temperature": self.temperature if sampling else None,
        }
        if not sampling:
            gen_kwargs["num_beams"] = 4
        ids = self.model.generate(**inputs, **gen_kwargs)

        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()


class M2MTranslateEngine(AITranslateEngine):
    _LANG_CODE_MAP = {
        "英文": "en",
        "日文": "ja",
        "繁體中文": "zh",
        "简体中文": "zh",
    }

    def _to_code(self, lang_name: str) -> str:
        try:
            return self._LANG_CODE_MAP[lang_name]
        except KeyError:
            raise ValueError(f"未定義語言代碼：{lang_name}")

    def __init__(self, config):
        super().__init__(config)
        self.model_name = config.get("model", "facebook/m2m100_418M")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.src_code = self._to_code(self.src)
        self.dest_code = self._to_code(self.dest)

        self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(self.model_name).to(
            self.device
        )
        self.bos_id = self.tokenizer.get_lang_id(self.dest_code)

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        self.tokenizer.src_lang = self.src_code

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        sampling = self.temperature > 0
        gen_kwargs = {
            "max_length": 512,
            "forced_bos_token_id": self.bos_id,
            "do_sample": sampling,
            "temperature": self.temperature if sampling else None,
        }
        if not sampling:
            gen_kwargs["num_beams"] = 4
        generated = self.model.generate(**inputs, **gen_kwargs)

        return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()


class OpenCCTranslateEngine(BaseTranslateEngine):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model = config.get("model ", "s2t") + ".json"
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
        elif engine_type == "nllb":
            return NLLBTranslateEngine(config)
        elif engine_type == "m2m":
            return M2MTranslateEngine(config)
        elif engine_type == "opencc":
            return OpenCCTranslateEngine(config)
        else:
            raise ValueError(f"未知的翻譯引擎類型: {engine_type}")
