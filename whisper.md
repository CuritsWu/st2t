以下內容將「WhisperModel.transcribe 各參數」按 **「適用情境 → 調整後效果 → 實務範例」** 整合成一張可立即對照的操作手冊，並特別把「Real-time + 高準度 + 抑制幻覺」的強化建議與 4070 Ti SUPER 上的緩衝 / 視窗最佳值放在最前面，方便直接採用。

---

## A. 推薦的「即時高準度」參數組

| 參數 | 推薦值 | 為何要這樣設？ | 立刻能得到的好處 |
|------|--------|----------------|------------------|
| `beam_size` | **5** | 在 Large-v3 + FP16 下可把解碼時間砍半，仍保留搜尋效果 | 延遲↓≈2×、WER 只小幅增加 |
| `temperature` | **[0 → 0.2 → 0.4]** | 失敗條件觸發才升溫 | 抗噪 ↑、避免死循環 |
| `compression_ratio_threshold` | **2.2** | gzip 比值稍嚴格 | 更快抓到亂語重試 |
| `log_prob_threshold` | **-1.0** | 平均對數機率過低就重試 | 同上 |
| `hallucination_silence_threshold` | **1.0 s** | 字級戳記時長靜音即斷句 | 大幅減少「靜音胡言」 |
| `repetition_penalty` | **1.1** | 對已輸出 token 施壓 | 口頭禪、重覆句下降 |
| `no_repeat_ngram_size` | **3** | 禁 3-gram 重覆 | 同上 |
| `condition_on_previous_text` | **False** | 改用視窗重疊而非跨窗 prompt | 幻覺 ↓、延遲 ↓ |
| `prompt_reset_on_temperature` | **0.3** | 一旦升溫即清空 prompt | 阻斷錯誤訊息擴散 |
| `vad_filter` | **True** | 先過 VAD 再丟模型 | 靜音切除、速度再提 5–10 % |
| 其餘保持預設 | — | 如 `suppress_blank=True`、`suppress_tokens` | 空行少、特殊符號剔除 |

> **典型呼叫範例**
> ```python
> segments, _ = model.transcribe(
>     audio,
>     language="ja",
>     task="translate",
>     beam_size=5,
>     temperature=[0, 0.2, 0.4],
>     compression_ratio_threshold=2.2,
>     log_prob_threshold=-1.0,
>     hallucination_silence_threshold=1.0,
>     repetition_penalty=1.1,
>     no_repeat_ngram_size=3,
>     condition_on_previous_text=False,
>     prompt_reset_on_temperature=0.3,
>     vad_filter=True,
>     vad_parameters={"threshold": 0.65},
>     suppress_blank=True,
> )
> ```

---

## B. 4070 Ti SUPER 推薦緩衝 / 視窗設定

| 參數 | 建議值 | 原因與效果 |
|------|--------|------------|
| `max_buffer_sec` | **12–15 s** | • 仍在單張 16 GB GPU 可 RT<1.0<br>• 片段長度足以保持上下文、防幻覺 |
| `overlap_sec` | **1.0 s** (語速快 0.8) | 一個完整單字保險；後處理簡單 |
| `interval_sec` | **3 s** (Sliding) | 增量字幕平均延遲 ≈ 1.5 s |
| `vad_threshold` | **0.62–0.7** | 視環境噪音微調，先求不漏字 |

> *若仍需再壓延遲*：把 `beam_size` ↓ 3、`max_buffer_sec` ↓ 10 s，再觀察 WER 取捨。

---

## C. 全參數對照表（依功能群組）

| 群組 | 主要參數 | 適用情境 | 調整後效果 | 實務範例 |
|------|----------|----------|------------|----------|
| **1. 基本輸入** | `audio` (必填) | — | — | `"/data/interview.wav"`<br>`language="ja"` 保日文、跳偵測<br>`task="translate"` 直接出英譯 |
| **2. 搜尋策略** | `beam_size`、`best_of`、`patience`、`length_penalty`、`repetition_penalty`、`no_repeat_ngram_size` | 高精 & 容忍延遲 | ↑ 值 = 準度↑/速度↓；懲罰重覆 | 法院逐字稿 `beam_size=10` |
| **3. 取樣 & 重試** | `temperature`(序列) `max_new_tokens` `compression_ratio_threshold` `log_prob_threshold` `no_speech_threshold` | 噪聲多、需重試 | 失敗即升溫；嚴格門檻更快重試 | 街訪 audio `temperature=[0,0.4,0.8]` |
| **4. 連貫 & Prompt** | `condition_on_previous_text` `prompt_reset_on_temperature` `initial_prompt` `prefix` | 長會議、專業詞 | 保上下文或提前餵詞庫 | 2 hr 會議逐字 |
| **5. 抑制 & Hotwords** | `suppress_blank` `suppress_tokens` `hotwords` | 想封鎖符號、強化專有名詞 | ↑ Hotword 機率；↓ 空行/符號 | 品酒會 `hotwords=["Cabernet"]` |
| **6. 時間戳 & 對齊** | `without_timestamps` `max_initial_timestamp` `word_timestamps` `prepend_/append_punctuations` | 字幕、K-Lite、Karaoke | 句級 vs 字級 時間碼、標點歸屬 | `word_timestamps=True` 卡拉 OK |
| **7. 多語 & 偵測** | `multilingual` `language_detection_threshold` `segments` | 段落混語 | 每段重偵測語言 | 國際研討會 |
| **8. VAD & 切片** | `vad_filter` `vad_parameters` `chunk_length` `clip_timestamps` `hallucination_silence_threshold` | 大量靜音、GPU 受限 | 靜音切除、平行化、小窗 | RTX 3060 `chunk_length=15` |
| **9. 其他** | `max_new_tokens` | 行動裝置硬上限 | 防「跑飛」佔滿 GPU | 手機即譯 2 s 分片 |

---

## D. 選擇 **without_timestamps / word_timestamps / hallucination_silence_threshold**

| 方案 | `without_timestamps` | `word_timestamps` | `hallucination_silence_threshold` | 用途 |
|------|---------------------|-------------------|-----------------------------------|------|
| **速度優先** | **True** | False | — | 純文字即時翻譯 |
| **抗幻覺 + 精對齊** | False | **True** | **1.0** | 字幕疊圖、Karaoke |
| **折衷 (預設)** | False | False | 1.0 (其實被忽略) | 句級時間碼即可 |

---

## E. 三種實戰工作流

| 目標 | 關鍵設置 | 呼叫範例摘要 |
|------|----------|--------------|
| **1. 快速轉錄** (YouTube 字幕) | `beam_size=1` `temperature=[0]` `vad_filter=True` | 速度第一，品質 OK |
| **2. 高精逐字** (研究訪談) | `beam=8` `best_of=5` `temperature=[0,0.3,0.6]` `condition_on_previous_text=True` `word_timestamps=True` | 犧牲時間換 WER↓ |
| **3. 多語翻譯** (國際研討會) | `task="translate"` `multilingual=True` `language_detection_segments=2` `no_repeat_ngram_size=4` | 自動偵測語言、翻成英文 |

---

### 使用說明小結

1. **先套用「即時高準度」建議值 (表 A)。**
2. 觀察延遲與錯字：
   * 若延遲仍高 → `beam_size` ↓ 3、`max_buffer_sec` ↓ 10 s。
   * 若幻覺仍多 → 開啟 `word_timestamps=True` + `hallucination_silence_threshold=1.0`。
3. 視 GPU/應用場景在 **速度 ↔ 準確度 ↔ 記憶體** 之間微調即可。

---

希望這份整合手冊能讓你在不同噪音、人力或運算限制下，迅速選出最划算的配置。需要更細部的腳本範例或想把表格直接匯成 Markdown / Excel 告訴我就行！