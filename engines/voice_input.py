import logging
import threading
import time

import soundcard as sc

from adapters.recorder_adapter import ListenerRecorderAdapter
from utils.simple import SimpleThreadDeque

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RecorderWorker(threading.Thread):
    def __init__(
        self, recorder, block_size, audio_queue: SimpleThreadDeque, stop_event
    ):
        super().__init__(daemon=True)
        self.recorder = recorder
        self.block_size = block_size
        self.audio_queue = audio_queue
        self.stop_event = stop_event

    def run(self):
        try:
            with self.recorder as rec:
                while not self.stop_event.is_set():
                    try:
                        data = rec.record(self.block_size)  # float32
                        # --- 遇到雙聲道時 down-mix ---
                        if data.ndim == 2:
                            data = data.mean(axis=1, keepdims=True)
                        self.audio_queue.append(data)
                    except Exception as e:
                        logger.error(f"錄音過程中發生錯誤: {e}")
        except Exception as e:
            logger.error(f"RecorderWorker 啟動失敗: {e}")


class AudioInputStream:
    def __init__(self, recorder, sample_rate: int, chunk_sec=0.03, max_latency=1.5):
        # chunk_sec = 每幀 n ms, # max_latency = 允許隊列最多積 n s 的音
        maxlen = int(max_latency / chunk_sec)
        self.audio_queue = SimpleThreadDeque(maxlen=maxlen)
        self.stop_event = threading.Event()
        self.block_size = int(sample_rate * chunk_sec)
        self.recorder = recorder
        self.worker = RecorderWorker(
            recorder, self.block_size, self.audio_queue, self.stop_event
        )

    def start(self):
        self.worker.start()

    def stream(self):
        while not self.stop_event.is_set() or self.audio_queue:
            if self.audio_queue:
                yield self.audio_queue.popleft()  # float32 ndarray
            else:
                time.sleep(0.005)

    def stop(self):
        self.stop_event.set()
        self.worker.join(timeout=3)
        logger.info("AudioInputStream 已停止並釋放資源")


class BaseInputEngine:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.streamer: AudioInputStream = None

    def start(self):
        raise NotImplementedError

    def stream_audio(self):
        yield from self.streamer.stream()

    def stop(self):
        if self.streamer:
            self.streamer.stop()


class MicrophoneInputEngine(BaseInputEngine):
    def __init__(self, config: dict):
        super().__init__(config.get("sample_rate", 16000))
        available = [d.name for d in sc.all_microphones(include_loopback=False)]
        name = config.get("device_name", sc.default_microphone().name)
        self.device_name = name if name in available else None
        logger.info(f"使用麥克風: {self.device_name or '預設'}")

    def start(self):
        try:
            mic = sc.get_microphone(self.device_name)
        except Exception as e:
            logger.warning(f"找不到指定麥克風 '{self.device_name}'，改為使用預設: {e}")
            mic = sc.default_microphone(include_loopback=False)

        recorder = mic.recorder(samplerate=self.sample_rate, channels=1)
        self.streamer = AudioInputStream(recorder, self.sample_rate)
        self.streamer.start()


class SystemAudioInputEngine(BaseInputEngine):
    def __init__(self, config: dict):
        super().__init__(config.get("sample_rate", 16000))
        available = [m.name for m in sc.all_speakers()]
        name = config.get("device_name", sc.default_speaker().name)
        self.device_name = name if name in available else None
        logger.info(f"使用系統音效裝置: {self.device_name or '預設'}")

    def start(self):
        try:
            sys_mic = sc.get_microphone(self.device_name, include_loopback=True)
        except Exception as e:
            logger.warning(
                f"找不到指定系統音效裝置 '{self.device_name}'，改用預設：{e}"
            )
            sys_mic = sc.get_microphone(
                sc.default_speaker().name, include_loopback=True
            )

        recorder = sys_mic.recorder(samplerate=self.sample_rate, channels=2)
        self.streamer = AudioInputStream(recorder, self.sample_rate)
        self.streamer.start()


class SocketInputEngine(BaseInputEngine):
    def __init__(self, config: dict):
        super().__init__(config.get("sample_rate", 16000))
        self.address = ("localhost", 6000)

    def start(self):
        recorder = ListenerRecorderAdapter(address=self.address)
        self.streamer = AudioInputStream(recorder, self.sample_rate)
        self.streamer.start()


class VoiceInputEngineFactory:
    @staticmethod
    def create(config: dict):
        engine_type = config.get("engine_type", "microphone")
        if engine_type == "microphone":
            return MicrophoneInputEngine(config)
        elif engine_type == "system":
            return SystemAudioInputEngine(config)
        elif engine_type == "socket":
            return SocketInputEngine(config)
        else:
            raise ValueError(f"未知的輸入引擎類型: {engine_type}")
