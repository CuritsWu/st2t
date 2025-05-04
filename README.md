# 即時語音辨識與翻譯工具

[![AI Generated](https://img.shields.io/badge/AI%20Written-90%25-blue.svg)](.)

這是一個基於 Python 的即時語音轉文字 (STT) 與翻譯工具，提供圖形化介面 (GUI) 進行設定，並支援多種輸入/輸出模式，包含網頁介面。

**注意：約 90% 的程式碼由 AI (例如 GPT-4) 協助編寫。**

## 主要功能

*   **即時語音辨識:**
    *   支援 Faster-Whisper 模型 (多種尺寸：tiny, base, small, medium, large-v3)
    *   支援 FunASR 模型 (目前**僅支援中文**)
    *   多種轉錄策略 (Overlap, Sliding Window)
*   **即時翻譯 (可選):**
    *   支援 Ollama (需本地運行 Ollama 服務)
    *   支援 NLLB (Facebook)
    *   支援 M2M100 (Facebook)
    *   支援 OpenCC (簡繁轉換)
*   **多種輸入來源:**
    *   麥克風 (`microphone`)
    *   系統音訊 (`system`, 例如喇叭輸出的聲音)
    *   Socket (`socket`, 可接收外部傳來的音訊流)
*   **多種輸出目標:**
    *   桌面懸浮窗 (`window`)
    *   Socket (`socket`, 可將結果傳送到外部應用)
*   **圖形化設定介面 (GUI):**
    *   使用 `gui.py` 方便調整所有參數。
    *   動態顯示/隱藏與所選引擎相關的設定。
    *   可儲存使用者設定至 `config/user_config.json`。
    *   可還原預設設定。
*   **網頁介面 (可選):**
    *   透過瀏覽器使用麥克風進行即時辨識與翻譯。
    *   使用 WebSocket 進行音訊傳輸和字幕顯示。
    *   **Web Server 的 Input/Output 皆使用 Socket 與主程式 (`main.py`) 通訊。**

## 測試環境

*   **GPU:** NVIDIA GeForce RTX 4070 Ti Super
*   **CUDA:** 12.1 或更高版本 (配合 PyTorch `cu128`)
*   **OS:** Windows (但也應能在 Linux/MacOS 上運行，需確認 `soundcard` 庫的相容性)

## 環境準備

1.  **Python:** 建議使用 Python 3.10 或 3.11。
2.  **CUDA Toolkit:** 若要使用 GPU 加速 (強烈建議)，請先安裝與您顯卡驅動相容的 NVIDIA CUDA Toolkit (例如 12.1 或更新版本)。 [CUDA 下載](https://developer.nvidia.com/cuda-downloads)
3.  **Git:** 用於克隆本專案。
4.  **(Web Server 選用) SSL 憑證:** 若要使用網頁介面 (`https://`)，瀏覽器通常需要 HTTPS 才能存取麥克風。您需要產生自我簽署或有效的 SSL 憑證 (`cert.pem`, `key.pem`) 並放在專案根目錄。可以使用 `openssl` 等工具生成：
    ```bash
    openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365 -nodes -subj "/CN=localhost"
    ```

## 安裝說明

1.  **克隆專案:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **建立虛擬環境 (建議):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux / MacOS
    source venv/bin/activate
    ```

3.  **安裝依賴:**
    ```bash
    # 注意 --extra-index-url 是為了安裝 CUDA 版本的 PyTorch
    pip install -r requirements.txt
    ```
    *如果安裝 `torch` 時遇到 CUDA 版本問題，請參考 [PyTorch 官網](https://pytorch.org/get-started/locally/) 的說明，選擇適合您 CUDA 版本的安裝指令。*
    *安裝 `soundcard` 可能需要在系統上安裝 PortAudio (Linux) 或其他音訊庫。*

## 使用說明

有三種主要的使用方式：

**1. 使用 GUI 設定並執行:**

*   這是最推薦的方式，方便調整所有設定。
*   執行 GUI：
    ```bash
    python gui.py
    ```
*   在 GUI 中：
    *   選擇各個引擎 (Input, Transcribe, Translate, Output) 的類型和參數。
    *   設定會自動儲存到 `config/user_config.json`。
    *   點擊 **▶ 執行** 按鈕啟動 `main.py`。
    *   點擊 **⛔ 停止** 按鈕中止 `main.py`。
    *   點擊 **🔄 還原...預設** 按鈕可還原設定。

**2. 直接使用命令列執行:**

*   您可以先透過 GUI 設定好 `config/user_config.json`，或手動編輯該檔案。
*   執行主程式：
    ```bash
    python main.py -c config/user_config.json
    # 或省略 -c 使用預設的 config/user_config.json
    # python main.py
    ```
*   程式會根據設定檔開始運作。按 `Ctrl+C` 可中止程式。

**3. 使用 Web Server 介面:**

*   **重要:** Web Server 模式需要 `main.py` 在背景運行，並且 **Input 和 Output 引擎都設定為 `socket` 模式**。
*   **步驟一：設定並啟動後端 `main.py`**
    *   開啟 `python gui.py`。
    *   將 `input_config` 的 `engine_type` 設為 `socket`。
    *   將 `output_config` 的 `engine_type` 設為 `socket`。
    *   (可選) 根據需要設定 `transcribe_config` 和 `translate_config`。
    *   點擊 **▶ 執行** 按鈕啟動後端。 **保持此程式運行。**
    *   *(或者，手動修改 `config/user_config.json` 將 input/output engine_type 改為 `socket`，然後執行 `python main.py`)*
*   **步驟二：啟動 Web Server**
    *   開啟 **新的** 終端機或命令提示字元。
    *   確認您已產生 `cert.pem` 和 `key.pem` 並放在專案根目錄 (參見 [環境準備](#環境準備))。
    *   執行 Web Server (使用 Python 模組模式)：
        ```bash
        python -m web.server
        ```
    *   Server 會監聽在 `https://0.0.0.0:8443`。
*   **步驟三：開啟瀏覽器**
    *   開啟支援 WebRTC 和 WebSocket 的現代瀏覽器 (如 Chrome, Firefox, Edge)。
    *   訪問 `https://localhost:8443` (或您伺服器的 IP 位址)。
    *   瀏覽器可能會警告憑證不受信任 (因為是自我簽署)，請選擇接受並繼續。
    *   瀏覽器會要求**麥克風權限**，請允許。
    *   開始說話，辨識和翻譯結果 (如果啟用翻譯) 會顯示在網頁上。

## 設定說明 (`config/user_config.json`)

設定檔主要包含四個部分：

*   `input_config`: 設定音訊輸入來源。
    *   `engine_type`: `microphone`, `system`, `socket`
    *   `device_name`: 選擇具體的麥克風或音效裝置 (GUI 會列出可用選項)。`socket` 模式下此項無效。
    *   `sample_rate`: 取樣率 (需與模型匹配，通常是 16000)。
*   `transcribe_config`: 設定語音辨識引擎。
    *   `engine_type`: `overlap`, `sliding` (基於 Faster-Whisper), `funasr` (僅中文)。
    *   `model_size` (Whisper): 模型大小。
    *   `compute_type` (Whisper): 計算精度 (`float16`, `int8` 等)。
    *   `language` (Whisper): 辨識語言 (`zh`, `en`, `ja`, `auto` 等)。 FunASR 固定為中文。
    *   `task` (Whisper): `transcribe` (轉錄) 或 `translate` (直接翻譯成英文)。
    *   其他參數用於調整 VAD (語音活動偵測)、解碼策略等，可參考 Faster-Whisper 文件。
*   `translate_config`: 設定翻譯引擎 (可選)。
    *   `enabled`: `true` / `false` 是否啟用翻譯。
    *   `engine_type`: `ollama`, `nllb`, `m2m`, `opencc`。
    *   `model`: 選擇具體的翻譯模型 (例如 Ollama 的模型名稱，NLLB/M2M 的 Hugging Face 路徑，OpenCC 的轉換模式)。
    *   `source_lang`, `target_lang`: 來源與目標語言。
    *   `temperature`: 控制生成文本的隨機性 (僅 AI 模型)。
*   `output_config`: 設定結果輸出方式。
    *   `engine_type`: `window` (懸浮窗), `socket`。
    *   `transparent_bg`, `font_size`, `font_color`, `wrap_length`: 懸浮窗樣式 (僅 `window` 模式)。

## 已知限制與注意事項

*   **FunASR 僅支援中文 (普通話)。**
*   效能高度依賴硬體 (特別是 GPU) 和所選的模型大小。
*   即時辨識和翻譯的準確度可能受口音、語速、背景噪音等因素影響。
*   Web Server 模式下，瀏覽器通常需要 HTTPS 連線才能安全地存取麥克風。
*   Socket 模式需要您有對應的客戶端或伺服器來接收或發送資料。
*   `soundcard` 庫在不同作業系統上的裝置名稱可能不同，GUI 會嘗試列出可用的裝置。