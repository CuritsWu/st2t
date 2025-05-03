# Whisper 引擎參數設定說明與推薦

## 🎯 Whisper 核心參數精準度 / 效能影響分析表

| 參數 | 範圍 / 類型 | 精準度 ↑ | 速度 ↓ | 說明與建議 |
|------|--------------|----------|--------|------------|
| `model_size` | `"tiny"` → `"large-v3"` | ✅✅✅✅✅ | ❌❌❌❌❌ | 模型越大越準但越慢；串流建議用 `small` / `medium`，離線處理可用 `large-v3` |
| `beam_size` | `1 ~ 10` | ✅✅（到 5） | ❌❌（線性變慢） | 控制搜尋精度；建議 `beam_size=5` 是準確與速度的折衷點，串流可設為 1~3 |
| `temperature` | `0.0 ~ 1.0+` | ✅（低） / ❌（高） | ⏱️ 幾乎無影響 | 控制隨機性，`0.0` 最穩定；高溫會跑出更多變化內容，可能不準 |
| `best_of` | `1 ~ 5+` | ✅✅ | ❌❌❌ | 多次生成挑最佳結果，非串流用，需搭配 temperature |
| `no_repeat_ngram_size` | `2 ~ 4` | ✅（抑制重複） | ⏱️ 幾乎無影響 | 提高語句自然性，建議設為 `3` |
| `repetition_penalty` | `1.0 ~ 1.5` | ✅（抑制口吃） | ⏱️ 幾乎無影響 | 防止重複輸出，建議設 `1.1 ~ 1.2` |
| `compression_ratio_threshold` | `1.5 ~ 3.0` | ✅（過濾異常） | ⏱️ 極小影響 | 過高代表內容不自然，建議 `2.2` |
| `log_prob_threshold` | `None`, `-1.0 ~ 0.0` | ✅（濾掉低信心句） | ⏱️ 極小影響 | 越接近 0 越嚴格，建議 `-0.3` |
| `condition_on_previous_text` | `True / False` | ✅（上下文） | ⏱️ 有效能負擔 | 串流建議開啟，需記憶空間 |
| `prompt_reset_on_temperature` | `0.0 ~ 1.0` | ✅（避免語境亂跑） | ⏱️ 無明顯影響 | 建議設為 `0.3` |
| `no_speech_threshold` | `0.3 ~ 0.8` | ✅（防幻聽） | ⏱️ 幾乎無影響 | 建議 `0.7`，避免亂聽靜音段 |
| `vad_threshold` | `0.3 ~ 0.8` | ✅（判斷有聲段） | ⏱️ 小影響 | 靈敏度，建議 `0.6 ~ 0.8` |

---

## 💨 串流速度優先設定建議

```python
model_size = "small"
beam_size = 1
temperature = [0.0]
no_repeat_ngram_size = 2
repetition_penalty = 1.0
condition_on_previous_text = True
```
## 🎯 精準度優先設定建議（離線處理）
```python
model_size = "large-v3"
beam_size = 5
temperature = [0.0, 0.2, 0.4]
best_of = 3
no_repeat_ngram_size = 3
repetition_penalty = 1.2
compression_ratio_threshold = 2.2
log_prob_threshold = -0.3
```