import time
from multiprocessing import Process, Queue

from faster_whisper import WhisperModel


class BackgroundDecoder(Process):
    def __init__(
        self, model_size, compute_type, decode_options, input_q: Queue, output_q: Queue
    ):
        super().__init__(daemon=True)
        self.model_size = model_size
        self.compute_type = compute_type
        self.decode_options = decode_options
        self.input_q = input_q
        self.output_q = output_q

    def run(self):
        model = WhisperModel(self.model_size, compute_type=self.compute_type)

        while True:
            wav_data = self.input_q.get()
            if wav_data is None:
                break  # 結束訊號
            start = time.time()
            try:
                segments, _ = model.transcribe(wav_data, **self.decode_options)
                segments = list(segments)
                text = "".join(s.text for s in segments).strip()
                duration = time.time() - start
                self.output_q.put((text, duration))
            except Exception as e:
                self.output_q.put((f"[ERROR] {e}", 0))
