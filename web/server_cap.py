# web/server.py
import asyncio
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from web.utils.simple import ReconnectableClient

# ------------------------------
# 設定參數
# ------------------------------
SAMPLE_RATE = 16000  # 與瀏覽器端相同
AUDIO_FRAME_MS = 20  # 每 20ms 一幀音訊
caption_websockets = []  # 所有字幕 WebSocket 連線

app = FastAPI()
html_path = Path(__file__).parent / "static" / "index_cap.html"
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ------------------------------
# 主頁面路由
# ------------------------------
@app.get("/")
async def get_root():
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.on_event("startup")
async def on_startup():
    global text_conn, caption_thread_running
    text_conn = ReconnectableClient(("localhost", 6001))

    loop = asyncio.get_running_loop()
    caption_thread_running = True

    def caption_listener():
        while caption_thread_running:
            if text_conn.poll(timeout=1):
                text = text_conn.recv()
                future = asyncio.run_coroutine_threadsafe(broadcast_caption(text), loop)
                try:
                    future.result()
                except Exception as e:
                    print("字幕傳送失敗：", e)

    threading.Thread(target=caption_listener, daemon=True).start()


@app.on_event("shutdown")
async def on_shutdown():
    global caption_thread_running
    caption_thread_running = False  # 停止背景 thread
    text_conn.close()
    print("[Server] 清理完成，準備關閉")


# ------------------------------
# 廣播字幕到所有 WebSocket Client
# ------------------------------
async def broadcast_caption(text: str):
    disconnected = []
    for ws in caption_websockets:
        try:
            await ws.send_text(text)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        caption_websockets.remove(ws)

# ------------------------------
# 推送字幕資料的 WebSocket
# ------------------------------
@app.websocket("/ws/caption")
async def websocket_caption(ws: WebSocket):
    await ws.accept()
    caption_websockets.append(ws)
    print("[Caption WebSocket] 已連線")
    try:
        while True:
            # 收到 client 傳來的資料僅為 keep-alive，忽略內容
            await ws.receive_text()
    except WebSocketDisconnect:
        caption_websockets.remove(ws)
        print("[Caption WebSocket] 已斷線")
    except Exception as e:
        caption_websockets.remove(ws)
        print(f"[Caption WebSocket] 錯誤：{e}")


# ------------------------------
# 啟動服務
# ------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "web.server_cap:app",
        host="0.0.0.0",
        port=8445,
        reload=False
    )
