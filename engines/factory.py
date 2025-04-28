# engines/factory.py
from engines.output import OutputEngineFactory as _OEFactory
from engines.transcribe import TranscribeEngineFactory
from engines.translate import TranslateEngineFactory
from engines.voice_input import VoiceInputEngineFactory as _IEFactory
from engines.websocket_io import WebSocketInputEngine, WebSocketOutputEngine

__all__ = [
    "VoiceInputEngineFactory",
    "TranscribeEngineFactory",
    "TranslateEngineFactory",
    "OutputEngineFactory",
]


# -------------------------------
# 重新包一層，注入 websocket 版本
# -------------------------------
class VoiceInputEngineFactory(_IEFactory):
    @staticmethod
    def create(config: dict):
        eng_type = config.get("engine_type", "microphone")
        if eng_type == "websocket":
            queue = config["extra"]["queue"]  # 由 server 注入
            return WebSocketInputEngine(config, queue)
        return _IEFactory.create(config)


class OutputEngineFactory(_OEFactory):
    @staticmethod
    def create(config: dict):
        eng_type = config.get("engine_type", "system")
        if eng_type == "websocket":
            clients = config["extra"]["clients"]  # 由 server 注入
            return WebSocketOutputEngine(config, clients)
        return _OEFactory.create(config)
