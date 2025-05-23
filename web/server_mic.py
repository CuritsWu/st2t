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
html_path = Path(__file__).parent / "static" / "index_mic.html"
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
    global audio_conn, caption_thread_running
    audio_conn = ReconnectableClient(("localhost", 6000))


@app.on_event("shutdown")
async def on_shutdown():
    audio_conn.close()
    print("[Server] 清理完成，準備關閉")

# ------------------------------
# 接收音訊資料的 WebSocket
# ------------------------------
@app.websocket("/ws/audio")
async def websocket_audio(ws: WebSocket):
    await ws.accept()
    print("[Audio WebSocket] 已連線")
    try:
        while True:
            audio_data = await ws.receive_bytes()
            audio_conn.send_bytes(audio_data)
    except WebSocketDisconnect:
        print("[Audio WebSocket] 已斷線")
    except Exception as e:
        print(f"[Audio WebSocket] 錯誤：{e}")


# ------------------------------
# 啟動服務
# ------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "web.server_mic:app",
        host="0.0.0.0",
        port=8443,
        reload=False,
        ssl_certfile="cert.pem",
        ssl_keyfile="key.pem",
    )
