# engines/websocket_io.py
import asyncio
import logging
import queue
from typing import List

from engines.output import BaseOutputEngine
from engines.voice_input import BaseInputEngine

logger = logging.getLogger(__name__)


# ------------------------------
# WebSocket → InputEngine
# ------------------------------
class WebSocketInputEngine(BaseInputEngine):
    """
    從 FastAPI 注入的 asyncio.Queue 取得 float32 PCM 音訊。
    """

    def __init__(self, config: dict, queue_: queue.Queue):
        super().__init__(config.get("sample_rate", 16000))
        self._queue: queue.Queue = queue_
        self._stopped = False

    def start(self):
        pass

    def stream_audio(self):
        while not self._stopped:
            chunk = self._queue.get()  # blocking 取資料
            yield chunk

    def stop(self):
        self._stopped = True


# ------------------------------
# OutputEngine → WebSocket
# ------------------------------
class WebSocketOutputEngine(BaseOutputEngine):
    """
    把字幕廣播給所有已掛載的 WebSocket 客戶端。
    """

    def __init__(self, config: dict, clients: List):
        self._clients: List = clients
        self._lock = asyncio.Lock()

    async def start(self):
        pass

    async def display(self, text: str):
        async with self._lock:
            for ws in list(self._clients):
                try:
                    await ws.send_text(text)
                except Exception:
                    # 連線中斷就移除
                    self._clients.remove(ws)

    def stop(self): ...
