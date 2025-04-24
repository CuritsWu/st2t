import logging
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from faster_whisper.tokenizer import Tokenizer

logging.getLogger("faster_whisper").setLevel(logging.WARNING)


class BaseTranscribeEngine(ABC):
    def __init__(self, config: dict):
        self.model_size = config.get("model_size", "large-v3")
        self.compute_type = config.get("compute_type", "auto")
        self.model = WhisperModel(
            self.model_size, device="cuda", compute_type=self.compute_type
        )
        self.sample_rate = config.get("sample_rate", 16000)
        lang = config.get("language", None)
        self.language = None if lang == "auto" else lang
        self.task = config.get("task", "transcribe")

        self.max_buffer_sec = config.get("max_buffer_sec", 5.0)
        self._buffer = deque()
        self._total_samples = 0
        self._max_samples = int(self.max_buffer_sec * self.sample_rate)
        self.beam_size = config.get("beam_size", 10)
        self.vad_threshold = config.get("vad_threshold", 0.7)
        self.temperature = config.get("temperature", 0)
        self.no_speech_threshold = config.get("no_speech_threshold", 0.7)
        self.condition_on_previous_text = config.get(
            "condition_on_previous_text", False
        )
        self.suppress = config.get("suppress", True)
        self.init_suppress_tokens()
        self.warm_up = config.get("warm_up", True)
        self.do_warm_up()

    def do_warm_up(self):
        import time

        if not self.warm_up:
            return
        # 根據語言載入對應檔案
        lang = self.language or "en"
        lang_map = {
            "ja": "ja.wav",
            "en": "en.wav",
            "zh": "zh.wav",
            "zh-TW": "zh.wav",
            "zh-CN": "zh.wav",
        }
        warmup_file = lang_map.get(lang, "ja.wav")
        warmup_path = Path("warmup").joinpath(warmup_file)

        if not warmup_path.is_file:
            logging.warning(f"找不到 warmup 音訊檔案 {warmup_path}，跳過暖機")
            return

        # 播放暖機音訊（但不送出 transcript）
        warmup_audio, _ = sf.read(warmup_path, dtype="float32")
        warmup_audio = self.process_audio_chunk(warmup_audio)
        self._buffer.append(warmup_audio)
        self._total_samples += len(warmup_audio)

        # 執行一次 transcribe，但不產生輸出（丟棄）
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

        # 再加入 max_buffer_sec 長度的靜音，觸發模型辨識
        silence_samples = int(self.max_buffer_sec * self.sample_rate)
        silence = np.zeros(silence_samples, dtype=np.float32)
        self._buffer.append(silence)
        self._total_samples += len(silence)

    def init_suppress_tokens(self):
        if not self.suppress:
            return
        tok = Tokenizer(
            self.model.hf_tokenizer, multilingual=True, task="transcribe", language="en"
        )

        # ② 列出所有想封鎖的結尾／口號（大小寫/標點差異照樣列入）
        ban_phrases = [
            # ==== English ====
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

        # ③ 轉成 token ID、去重
        ban_ids = set()
        for phrase in ban_phrases:
            ban_ids.update(tok.encode(phrase))

        # ④ 加入 -1（Whisper 預設的非語音特殊 token）
        self.suppress_tokens = [-1, *sorted(ban_ids)]

    @abstractmethod
    def transcribe_stream(self, audio_stream):
        pass

    def process_audio_chunk(self, chunk):
        """預處理音訊片段（例如轉換為單聲道等）"""
        if chunk.ndim == 2:
            chunk = np.mean(chunk, axis=1)  # convert stereo to mono
        return chunk

    def reset_buffer(self):
        """清空緩衝區"""
        self._buffer.clear()
        self._total_samples = 0


class OverlapTranscribeEngine(BaseTranscribeEngine):
    def __init__(self, config: dict):
        super().__init__(config)
        self.overlap_sec = config.get("overlap_sec", 3.0)
        self._overlap_samples = int(self.overlap_sec * self.sample_rate)

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
                    task=self.task,
                    beam_size=self.beam_size,
                    vad_filter=True,
                    vad_parameters={"threshold": self.vad_threshold},
                    temperature=self.temperature,
                    no_speech_threshold=self.no_speech_threshold,
                    condition_on_previous_text=self.condition_on_previous_text,
                    suppress_tokens=self.suppress_tokens if self.suppress else None,
                    suppress_blank=self.suppress,
                )

                yield "".join([seg.text for seg in segments])

                # 保留 overlap 的尾段
                overlap_data = data[-self._overlap_samples :]
                self._buffer = deque([overlap_data])
                self._total_samples = len(overlap_data)


class SlidingWindowTranscribeEngine(BaseTranscribeEngine):
    def __init__(self, config: dict):
        super().__init__(config)
        self.interval_sec = config.get("interval_sec", 2.0)
        self._interval_samples = int(self.interval_sec * self.sample_rate)
        self._new_samples = 0

    def transcribe_stream(self, audio_stream):
        for chunk in audio_stream:
            chunk = self.process_audio_chunk(chunk)

            # 加入新 chunk
            self._buffer.append(chunk)
            self._total_samples += len(chunk)
            self._new_samples += len(chunk)

            # 如果 buffer 超過 max_buffer_sec，就從左邊丟棄最舊的片段
            while self._total_samples > self._max_samples:
                left = self._buffer.popleft()
                self._total_samples -= len(left)

            # 當前視窗足夠長，且自上次輸出以來累積了 interval_sec
            if self._new_samples >= self._interval_samples:
                data = np.concatenate(self._buffer)
                segments, _ = self.model.transcribe(
                    data,
                    language=self.language,
                    task=self.task,
                    beam_size=self.beam_size,
                    vad_filter=True,
                    vad_parameters={"threshold": self.vad_threshold},
                    temperature=self.temperature,
                    no_speech_threshold=self.no_speech_threshold,
                    # condition_on_previous_text=self.condition_on_previous_text,
                    suppress_tokens=self.suppress_tokens if self.suppress else None,
                    suppress_blank=self.suppress,
                )

                # 輸出
                yield "".join(seg.text for seg in segments)

                # 重設新資料計數
                self._new_samples = 0


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
