# web/server.py
import asyncio
import queue
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from engines.factory import TranscribeEngineFactory
from engines.translate import TranslateEngineFactory
from engines.websocket_io import WebSocketInputEngine, WebSocketOutputEngine

# ------------------------------
# 基本參數
# ------------------------------
SAMPLE_RATE = 16000  # 與瀏覽器端相同
AUDIO_FRAME_MS = 20  # 20 ms ＝ 320 sample
audio_queue = queue.Queue()  # Float32 PCM buffer
caption_clients = []  # 連線中的字幕 WebSocket

app = FastAPI()
html_path = Path(__file__).parent / "static" / "index.html"
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ------------------------------
# 路由 – 首頁
# ------------------------------
@app.get("/")
async def get_root():
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ------------------------------
# 音訊 WebSocket
# ------------------------------
@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    print("[audio] connected")
    try:
        while True:
            data = await ws.receive_bytes()  # ArrayBuffer → bytes
            pcm = np.frombuffer(data, dtype=np.float32)
            audio_queue.put(pcm)
    except WebSocketDisconnect:
        print("[audio] disconnected")


# ------------------------------
# 字幕 WebSocket
# ------------------------------
@app.websocket("/ws/caption")
async def ws_caption(ws: WebSocket):
    await ws.accept()
    caption_clients.append(ws)
    try:
        while True:
            await ws.receive_text()  # keep-alive
    except WebSocketDisconnect:
        caption_clients.remove(ws)


# ------------------------------
# 背景 STT 工作
# ------------------------------
import asyncio
import threading


async def stt_worker():
    # ---------- 建立引擎 ----------
    in_cfg = {
        "engine_type": "websocket",
        "sample_rate": SAMPLE_RATE,
        "extra": {"queue": audio_queue},
    }
    out_cfg = {
        "engine_type": "websocket",
        "extra": {"clients": caption_clients},
    }

    input_engine = WebSocketInputEngine(in_cfg, audio_queue)
    output_engine = WebSocketOutputEngine(out_cfg, caption_clients)
    stt_engine = TranscribeEngineFactory.create(
        {
            "engine_type": "sliding",
            "model_size": "medium",
            "compute_type": "float16",
            "sample_rate": 16000,
            "language": "zh",
            "task": "transcribe",
            "init_prompt": "這是一段繁體中文語音",
            "max_buffer_sec": 30,
            "overlap_sec": 1.0,
            "interval_sec": 1.0,
            "vad_threshold": 0.4,
            "beam_size": 5,
            "temperature": "0.2, 0.4, 0.6",
            "compression_ratio_threshold": 2.2,
            "log_prob_threshold": -1.0,
            "hallucination_silence_threshold": 0.8,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3,
            "prompt_reset_on_temperature": 0.3,
            "condition_on_previous_text": 1,
            "no_speech_threshold": 0.4,
            "without_timestamps": 0,
            "word_timestamps": False,
            "suppress": 0,
            "warm_up": True,
        }
    )
    translator = TranslateEngineFactory.create(
        {
            "engine_type": "ollama",
            "model": "gemma3",
            "target_lang": "英文",
            "temperature": 0.0,
            "enabled": 1,
        }
    )

    input_engine.start()  # 啟動 OK（同步，內部開 recorder thread）
    await output_engine.start()  # 目前是空實作

    loop = asyncio.get_running_loop()

    # ---------- 把重活丟到執行緒 ----------
    def transcribe_job():
        # for text in stt_engine.transcribe_stream(input_engine.stream_audio()):
        for text in translator.translate_stream(
            stt_engine.transcribe_stream(input_engine.stream_audio())
        ):
            # 把 async display 推回 event-loop 執行
            fut = asyncio.run_coroutine_threadsafe(output_engine.display(text), loop)
            try:
                fut.result()  # 同步等字幕送完，可換成 .done() 無視例外
            except Exception as e:
                print("display error:", e)

    threading.Thread(target=transcribe_job, daemon=True).start()


@app.on_event("startup")
async def startup():
    asyncio.create_task(stt_worker())


# ------------------------------
# main
# ------------------------------
if __name__ == "__main__":
    # uvicorn.run("web.server:app", host="0.0.0.0", port=8000, reload=False)
    uvicorn.run(
        "web.server:app",
        host="0.0.0.0",
        port=8443,
        reload=False,
        ssl_certfile="fastapi-cert.pem",
        ssl_keyfile="fastapi-key.pem",
    )
