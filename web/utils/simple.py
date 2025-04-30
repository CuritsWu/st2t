import threading
import time
from multiprocessing.connection import Client, Connection


class ReconnectableClient:
    def __init__(self, address, authkey=None, retry_interval=3):
        self.address = address
        self.authkey = authkey
        self.retry_interval = retry_interval
        self.conn: Connection | None = None
        self.running = True
        self.lock = threading.Lock()
        self._connect_thread = threading.Thread(target=self._connect_loop, daemon=True)
        self._connect_thread.start()

    def _connect_loop(self):
        while self.running:
            if self.conn is None:
                try:
                    conn = Client(self.address, authkey=self.authkey)
                    with self.lock:
                        self.conn = conn
                    print(f"[ReconnectableClient] 已連線到 {self.address}")
                except Exception as e:
                    print(
                        f"[ReconnectableClient] 連線失敗：{e} {self.retry_interval} 秒後重試…"
                    )
                    time.sleep(self.retry_interval)
            else:
                time.sleep(1)  # 已連線時定期檢查

    def send_bytes(self, data):
        with self.lock:
            if self.conn:
                try:
                    self.conn.send_bytes(data)
                except Exception:
                    print("[ReconnectableClient] 發送失敗，重設連線")
                    self.conn = None

    def poll(self, timeout=None):
        with self.lock:
            if self.conn:
                try:
                    return self.conn.poll(timeout)
                except Exception:
                    print("[ReconnectableClient] poll 失敗，重設連線")
                    self.conn = None
        return False

    def recv(self):
        with self.lock:
            if self.conn:
                try:
                    return self.conn.recv()
                except Exception:
                    print("[ReconnectableClient] recv 失敗，重設連線")
                    self.conn = None
        return None

    def close(self):
        self.running = False
        with self.lock:
            if self.conn:
                self.conn.close()
                self.conn = None
