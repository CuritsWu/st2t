import argparse
import json
import signal
import sys
from copy import deepcopy
from pathlib import Path

from config.path import DEFAULT_CFG_PATH, USER_CFG_PATH
from engines.factory import (OutputEngineFactory, TranscribeEngineFactory,
                             TranslateEngineFactory, VoiceInputEngineFactory)


# ---------- 共用 ---------- #
def deep_update(base: dict, patch: dict) -> dict:
    """遞迴把 patch 值覆蓋到 base；對 list / tuple 直接替換"""
    merged = deepcopy(base)
    for k, v in patch.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_update(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(path: str | Path | None) -> dict:
    default_cfg = json.loads(DEFAULT_CFG_PATH.read_text(encoding="utf-8"))
    user_cfg = {}
    # 優先用 CLI 指定的檔；沒有就找 user_config.json
    if path:
        user_cfg_path = Path(path)
    else:
        user_cfg_path = Path(USER_CFG_PATH)
    if user_cfg_path.is_file():
        user_cfg = json.loads(user_cfg_path.read_text(encoding="utf-8"))
    return deep_update(default_cfg, user_cfg)


# ---------- 參數解析 ---------- #
parser = argparse.ArgumentParser(description="STT + 翻譯 + 輸出")
parser.add_argument(
    "-c", "--config", help="自訂設定檔 (json)，預設為 user_config.json", default=None
)
args = parser.parse_args()

config = load_config(args.config)
# ---------- Ctrl-C 處理 ---------- #
input_engine = None


def signal_handler(sig, frame):
    print("\n🛑 偵測到 Ctrl+C，中止...\n")
    if input_engine:
        input_engine.stop()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

# ---------- 執行 ---------- #
print("🎧 STT + 翻譯 + 輸出")
print("====================================")

# === 輸入 ===
input_engine = VoiceInputEngineFactory.create(config["input_config"])

# === 翻譯器（可選） ===
trans_cfg = config.get("translate_config", {})
if trans_cfg.get("enabled", False):
    translator = TranslateEngineFactory.create(trans_cfg)
else:
    translator = None

# === 執行流程 ===
print("\n📡 開始錄音中，請說話...（Ctrl+C 可中止）")
stt_engine = TranscribeEngineFactory.create(config["transcribe_config"])
input_engine.start()


def create_stream():
    raw_stream = stt_engine.transcribe_stream(input_engine.stream_audio())
    return translator.translate_stream(raw_stream) if translator else raw_stream


output_engine = OutputEngineFactory.create(config["output_config"])
output_engine.start()

stream = create_stream()
for text in stream:
    output_engine.display(text)
