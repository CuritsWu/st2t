import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from faster_whisper.tokenizer import Tokenizer

# 將 faster_whisper 的詳盡 debug 訊息關掉，保持輸出乾淨
logging.getLogger("faster_whisper").setLevel(logging.WARNING)


class BaseTranscribeEngine(ABC):
    def __init__(self, config: dict):
        # ----- 模型設定 -----
        self.model_size = config.get("model_size", "large-v3")
        self.compute_type = config.get("compute_type", "auto")
        self.model = WhisperModel(
            self.model_size, device="cuda", compute_type=self.compute_type
        )

        # ----- 音訊與語言 -----
        self.sample_rate = config.get("sample_rate", 16000)
        lang = config.get("language", None)
        self.language = None if lang == "auto" else lang
        self.task = config.get("task", "transcribe")
        self.init_prompt = config.get("init_prompt", "")

        # ----- 緩衝設定（依 GPU 容量 / 延遲需求調整） -----
        self.max_buffer_sec = config.get("max_buffer_sec", 12.0)  # 推薦 12–15 秒
        self._buffer: deque[np.ndarray] = deque()
        self._total_samples = 0
        self._max_samples = int(self.max_buffer_sec * self.sample_rate)

        # ----- 解碼與抑制參數（Real‑time + 高準度 + 抗幻覺） -----
        self.beam_size = config.get("beam_size", 5)
        temp = map(
            lambda t: float(t.strip()),
            config.get("temperature", "0.0, 0.2, 0.4").split(","),
        )
        self.temperature = [*temp]
        self.vad_threshold = config.get("vad_threshold", 0.7)
        self.no_speech_threshold = config.get("no_speech_threshold", 0.7)
        self.compression_ratio_threshold = config.get(
            "compression_ratio_threshold", 2.2
        )
        self.log_prob_threshold = config.get("log_prob_threshold", -1.0)
        self.hallucination_silence_threshold = config.get(
            "hallucination_silence_threshold", 1.0
        )
        self.repetition_penalty = config.get("repetition_penalty", 1.1)
        self.no_repeat_ngram_size = config.get("no_repeat_ngram_size", 3)
        self.condition_on_previous_text = config.get(
            "condition_on_previous_text", False
        )
        self.prompt_reset_on_temperature = config.get(
            "prompt_reset_on_temperature", 0.3
        )

        # ── 句 / 字級時間碼 ─────────────────────────────
        self.without_timestamps = config.get("without_timestamps", False)
        self.word_timestamps = config.get("word_timestamps", False)

        # ----- 抑制口號與空白 -----
        self.suppress = config.get("suppress", True)
        self.init_suppress_tokens()

        # ----- 暖機 -----
        self.warm_up = config.get("warm_up", True)
        self.do_warm_up()

    def do_warm_up(self):
        if not self.warm_up:
            return

        lang = self.language or "en"
        lang_map = {
            "ja": "ja.wav",
            "en": "en.wav",
            "zh": "zh.wav",
        }
        warmup_file = lang_map.get(lang, "ja.wav")
        warmup_path = Path("warmup").joinpath(warmup_file)

        if not warmup_path.is_file():
            logging.warning(f"找不到 warmup 音訊檔案 {warmup_path}，跳過暖機")
            return

        # 讀入暖機音訊（不輸出逐字稿）
        warmup_audio, _ = sf.read(warmup_path, dtype="float32")
        warmup_audio = self.process_audio_chunk(warmup_audio)
        self._buffer.append(warmup_audio)
        self._total_samples += len(warmup_audio)

        try:
            data = np.concatenate(self._buffer)
            self.model.transcribe(
                data,
                language=self.language,
                task=self.task,
            )
        except Exception as e:
            logging.error(f"warm_up transcribe 發生錯誤：{e}")

        self.reset_buffer()
        self.full_silence()

    def full_silence(self):
        # 填充一段靜音以確保解碼圖被初始化
        silence_samples = int(self.max_buffer_sec * self.sample_rate)
        silence = np.zeros(silence_samples, dtype=np.float32)
        self._buffer.append(silence)
        self._total_samples += len(silence)

    def init_suppress_tokens(self):
        if not self.suppress:
            self.suppress_tokens = None
            return

        tok = Tokenizer(
            self.model.hf_tokenizer, multilingual=True, task="transcribe", language="en"
        )

        ban_phrases = [
            # ====== 英文 ======
            " thanks for watching",
            " thank you for watching",
            " thanks for listening",
            " thank you for listening",
            " thanks for tuning in",
            " thanks for joining us",
            " don't forget to like and subscribe",
            " please like and subscribe",
            " remember to like and subscribe",
            " make sure to subscribe",
            " hit the bell",
            " leave a comment",
            " see you next time",
            " see you in the next video",
            " bye bye",
            " goodbye everyone",
            " have a great day",
            # ==== 中文 (繁/簡) ====
            " 感謝觀看",
            "感谢观看",
            " 感謝收聽",
            "感谢收听",
            " 記得訂閱",
            "记得订阅",
            " 別忘了訂閱",
            "别忘了订阅",
            " 按讚",
            "点赞",
            " 留言",
            " 下次再見",
            "下一次见",
            " 拜拜",
            # ==== Japanese ====
            " ご視聴ありがとうございました",
            "ご視聴ありがとうございました",
            " チャンネル登録よろしくお願いします",
            " 高評価お願いします",
            " コメントお願いします",
            " また次の動画でお会いしましょう",
            " バイバイ",
            # ==== Korean ====
            " 시청해 주셔서 감사합니다",
            "구독과 좋아요 부탁드립니다",
            " 댓글 남겨주세요",
            " 다음 영상에서 만나요",
            " 안녕히 계세요",
            # ==== Spanish ====
            " gracias por ver",
            " gracias por ver este video",
            " gracias por escuchar",
            " no olvides suscribirte",
            " recuerda suscribirte",
            " deja un comentario",
            " nos vemos en el próximo video",
            " hasta la próxima",
            " adiós",
            # ==== French ====
            " merci d'avoir regardé",
            " merci pour votre attention",
            " n'oubliez pas de vous abonner",
            " pensez à vous abonner",
            " laissez un commentaire",
            " à la prochaine",
            " au revoir",
            # ==== German ====
            " danke fürs zuschauen",
            " danke fürs zuhören",
            " vergiss nicht zu abonnieren",
            " abonniere meinen kanal",
            " lass einen kommentar",
            " bis zum nächsten mal",
            " tschüss",
            # ==== Portuguese ====
            " obrigado por assistir",
            " obrigado por escutar",
            " não esqueça de se inscrever",
            " deixe um comentário",
            " até a próxima",
            " tchau",
            # ==== Italian ====
            " grazie per aver guardato",
            " non dimenticare di iscriverti",
            " iscriviti al canale",
            " lascia un commento",
            " ci vediamo nel prossimo video",
            " ciao",
            # ==== Russian ====
            " спасибо за просмотр",
            " не забудьте подписаться",
            " ставьте лайк",
            " оставьте комментарий",
            " до следующего раза",
            " пока",
            # ==== Hindi ====
            " देखने के लिए धन्यवाद",
            " सुनने के लिए धन्यवाद",
            " सब्सक्राइब करना न भूलें",
            " लाइक और शेयर करें",
            " कमेंट करें",
            " मिलते हैं अगली बार",
            " अलविदा",
        ]

        ban_ids = set()
        for phrase in ban_phrases:
            ban_ids.update(tok.encode(phrase))

        # -1 為 Whisper 的非語音特殊 token
        self.suppress_tokens = [-1, *sorted(ban_ids)]

    @abstractmethod
    def transcribe_stream(self, audio_stream):
        """子類別實作串流轉錄的主要邏輯"""

    def process_audio_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """將雙聲道轉單聲道，或其他前處理"""
        if chunk.ndim == 2:
            chunk = np.mean(chunk, axis=1)
        return chunk

    def reset_buffer(self, full_silence=False):
        """清空緩衝區"""
        self._buffer.clear()
        self._total_samples = 0
        if full_silence:
            self.full_silence()


class OverlapTranscribeEngine(BaseTranscribeEngine):
    def __init__(self, config: dict):
        super().__init__(config)
        self.overlap_sec = config.get("overlap_sec", 1.0)
        self._overlap_samples = int(self.sample_rate * self.overlap_sec)

    def transcribe_stream(self, audio_stream):
        for chunk in audio_stream:
            chunk = self.process_audio_chunk(chunk)
            self._buffer.append(chunk)
            self._total_samples += len(chunk)

            if self._total_samples >= self._max_samples:
                data = np.concatenate(self._buffer)

                segments, _ = self.model.transcribe(
                    data,
                    language=self.language,
                    multilingual=self.language is None,
                    task=self.task,
                    initial_prompt=self.init_prompt or None,
                    beam_size=self.beam_size,
                    temperature=self.temperature,
                    compression_ratio_threshold=self.compression_ratio_threshold,
                    log_prob_threshold=self.log_prob_threshold,
                    hallucination_silence_threshold=self.hallucination_silence_threshold,
                    repetition_penalty=self.repetition_penalty,
                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                    condition_on_previous_text=self.condition_on_previous_text,
                    prompt_reset_on_temperature=self.prompt_reset_on_temperature,
                    vad_filter=True,
                    vad_parameters={"threshold": self.vad_threshold},
                    no_speech_threshold=self.no_speech_threshold,
                    suppress_tokens=self.suppress_tokens,
                    suppress_blank=self.suppress,
                )

                yield "".join(seg.text for seg in segments)

                # 保留尾段重疊以保持上下文連續
                overlap_data = data[-self._overlap_samples :]
                self._buffer = deque([overlap_data])
                self._total_samples = len(overlap_data)


class SlidingWindowTranscribeEngine(BaseTranscribeEngine):
    def __init__(self, config: dict):
        super().__init__(config)
        self.interval_sec = config.get("interval_sec", 3.0)
        self._interval_samples = int(self.sample_rate * self.interval_sec)

    def transcribe_stream(self, audio_stream):
        last_end_time = 0.0
        new_samples = 0
        for chunk in audio_stream:
            chunk = self.process_audio_chunk(chunk)

            self._buffer.append(chunk)
            self._total_samples += len(chunk)
            new_samples += len(chunk)

            # 若 buffer 過長，丟棄最舊資料
            while self._total_samples > self._max_samples:
                left = self._buffer.popleft()
                self._total_samples -= len(left)

            # 每收滿 interval_sec 就解碼一次
            if new_samples >= self._interval_samples:
                data = np.concatenate(self._buffer)
                start_time = time.time()
                segments, _ = self.model.transcribe(
                    data,
                    language=self.language,
                    multilingual=self.language is None,
                    task=self.task,
                    initial_prompt=self.init_prompt or None,
                    beam_size=self.beam_size,
                    temperature=self.temperature,
                    compression_ratio_threshold=self.compression_ratio_threshold,
                    log_prob_threshold=self.log_prob_threshold,
                    hallucination_silence_threshold=self.hallucination_silence_threshold,
                    repetition_penalty=self.repetition_penalty,
                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                    condition_on_previous_text=self.condition_on_previous_text,
                    prompt_reset_on_temperature=self.prompt_reset_on_temperature,
                    vad_filter=True,
                    vad_parameters={"threshold": self.vad_threshold},
                    no_speech_threshold=self.no_speech_threshold,
                    suppress_tokens=self.suppress_tokens,
                    suppress_blank=self.suppress,
                )
                transcribe_time = time.time() - start_time
                target_interval = max(self.interval_sec, transcribe_time * 1.5)
                self._interval_samples = int(target_interval * self.sample_rate)

                new_samples = 0

                new_segs = [s for s in segments if s.end > last_end_time]
                if self.suppress:
                    new_segs = [
                        seg
                        for seg in new_segs
                        if seg.avg_logprob >= -1.0
                        or (seg.end - seg.start) / len(seg.text) >= 0.07
                    ]

                if not new_segs:
                    last_end_time = 0
                    self.reset_buffer(full_silence=True)
                    yield ""
                else:
                    last_end_time = new_segs[-1].end
                    yield "".join(s.text for s in new_segs).strip()


class TranscribeEngineFactory:
    @staticmethod
    def create(config: dict):
        engine_type = config.get("engine_type", "overlap")
        if engine_type == "overlap":
            return OverlapTranscribeEngine(config)
        elif engine_type == "sliding":
            return SlidingWindowTranscribeEngine(config)
        else:
            raise ValueError(f"未知的引擎類型: {engine_type}")
