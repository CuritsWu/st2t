const TARGET_RATE = 16000;

class PCMWriter extends AudioWorkletProcessor {
    constructor() {
        super();
        this.inputRate = sampleRate;                // 通常 48000
        this.buf = [];
        this.ratio = this.inputRate / TARGET_RATE;  // 3  (48k ➜ 16k)
    }

    process(inputs) {
        const input = inputs[0][0];
        if (!input) return true;

        // ---------- 簡單下採樣 (drop samples) ----------
        for (let i = 0; i < input.length; i += this.ratio) {
            this.buf.push(input[i]);
            if (this.buf.length >= TARGET_RATE * 0.02) {  // 20 ms = 320 個 16k 取樣
                const chunk = new Float32Array(this.buf.splice(0, TARGET_RATE * 0.02));
                this.port.postMessage(chunk);           // 傳回主執行緒
            }
        }
        return true;
    }
}
registerProcessor('pcm-writer', PCMWriter);
