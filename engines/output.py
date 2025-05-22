# engines/output.py

import logging
import queue
import threading
import time
import tkinter as tk
from abc import ABC, abstractmethod
from multiprocessing.connection import Listener

logger = logging.getLogger(__name__)


class BaseOutputEngine(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def display(self, text: str):
        pass

    @abstractmethod
    def stop(self):
        pass


class WindowOutputEngine(BaseOutputEngine):
    def __init__(self, config: dict):
        self.font_size = config.get("font_size", 24)
        self.font_color = config.get("font_color", "#00FF99")
        self.transparent_bg = config.get("transparent_bg", True)
        self.wrap_length = config.get("wrap_length", 800)
        self.text = ""
        self.root = None
        self.label = None
        self.queue = queue.Queue()

    def _start_move(self, event):
        self._x_offset = event.x
        self._y_offset = event.y

    def _on_move(self, event):
        x = self.root.winfo_pointerx() - self._x_offset
        y = self.root.winfo_pointery() - self._y_offset
        self.root.geometry(f"+{x}+{y}")

    def _run_window(self):
        self.root.title("字幕翻譯")
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x = (screen_width) // 2
        y = screen_height * 2 // 3
        self.root.geometry(f"+{x}+{y}")
        self.label = tk.Label(
            self.root,
            text=self.text,
            font=("Helvetica", self.font_size),
            fg=self.font_color,
            bg="black",
            wraplength=self.wrap_length,
            justify="left",
        )
        self.label.pack()
        self.label.bind("<Button-1>", self._start_move)
        self.label.bind("<B1-Motion>", self._on_move)

        if self.transparent_bg:
            self.root.attributes("-alpha", 0.8)
        self.root.mainloop()

    def start(self):
        self.root = tk.Tk()
        self._poll_queue()
        self._run_window()

    def _poll_queue(self):
        try:
            while True:
                text = self.queue.get_nowait()
                self.label.config(text=text)  # 只在 Tk 執行緒動 GUI
        except queue.Empty:
            pass
        self.root.after(30, self._poll_queue)  # 30ms 更新一次

    def display(self, text: str):
        # run in other thread to put text for main thread
        self.queue.put(text)

    def stop(self):
        if self.root:
            self.root.quit()


class SocketOutputEngine(BaseOutputEngine):
    def __init__(self, config: dict):
        self.address = ("localhost", 6001)
        self.listener = None
        self.conn = None
        self.authkey = None

    def start(self):
        self.listener = Listener(self.address, authkey=self.authkey)
        self.conn = self.listener.accept()
        while True:
            time.sleep(1)

    def display(self, text):
        self.conn.send(text)

    def stop(self):
        if self.conn:
            self.conn.close()
        if self.listener:
            self.listener.close()


class OutputEngineFactory:
    @staticmethod
    def create(config: dict) -> BaseOutputEngine:
        engine_type = config.get("engine_type", "window")
        if engine_type == "window":
            return WindowOutputEngine(config)
        elif engine_type == "socket":
            return SocketOutputEngine(config)
        else:
            raise ValueError(f"未知的輸出引擎類型: {engine_type}")
