from abc import ABC, abstractmethod
from multiprocessing.connection import Listener

import numpy as np


class RecorderAdapter(ABC):
    """
    所有 Recorder 的抽象基底類別
    必須支援 with 語法、record(block_size)
    """

    @abstractmethod
    def __enter__(self):
        """初始化資源（ex: socket, stream）"""
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """清理資源"""
        pass

    @abstractmethod
    def record(self, block_size: int) -> np.ndarray:
        """
        回傳 shape 為 (block_size, 1) 的 float32 音訊資料
        """
        pass


class ListenerRecorderAdapter(RecorderAdapter):
    def __init__(self, address, authkey=None, timeout=0.02, fake_value=0.0):
        self.address = address
        self.authkey = authkey
        self.timeout = timeout
        self.fake_value = fake_value
        self.listener = None
        self.conn = None
        self.buffer = np.empty((0, 1), dtype=np.float32)

    def __enter__(self):
        self.listener = Listener(self.address, authkey=self.authkey)
        self.conn = self.listener.accept()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
        if self.listener:
            self.listener.close()

    def record(self, block_size: int) -> np.ndarray:
        while len(self.buffer) < block_size:
            if self.conn.poll(self.timeout):
                try:
                    data = self.conn.recv_bytes()
                    data = self._to_array(data)
                    self.buffer = np.vstack([self.buffer, data])
                except Exception:
                    self._fake_buffer(block_size)
            else:
                self._fake_buffer(block_size)

        out = self.buffer[:block_size]
        self.buffer = self.buffer[block_size:]
        return out

    def _fake_buffer(self, block_size):
        need_size = block_size - len(self.buffer)
        if need_size > 0:
            self.buffer = np.vstack(
                [
                    self.buffer,
                    self._generate_fake_data(need_size),
                ]
            )

    def _generate_fake_data(self, size):
        return np.full((size, 1), self.fake_value, dtype=np.float32)

    def _to_array(self, data) -> np.ndarray:
        data = np.frombuffer(data, dtype=np.float32)
        return data[:, np.newaxis] if data.ndim == 1 else data
