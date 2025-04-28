import json
import signal
import subprocess
import sys
import tkinter as tk
from copy import deepcopy
from tkinter import messagebox, ttk

import soundcard as sc

from config.path import (CHOICES_PATH, DEFAULT_CFG_PATH, TRANSLATE_MODEL_PATH,
                         USER_CFG_PATH)
from utils.common import deep_update


def deep_set(dic: dict, dot_path: str, value):
    parts = dot_path.split(".")
    cur = dic
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def ensure_files():
    if not DEFAULT_CFG_PATH.is_file():
        messagebox.showerror("錯誤", "找不到 default_config.json")
        sys.exit(1)
    if not CHOICES_PATH.is_file():
        CHOICES_PATH.write_text("{}", encoding="utf-8")
    if not USER_CFG_PATH.is_file():
        USER_CFG_PATH.write_text("{}", encoding="utf-8")
    if not TRANSLATE_MODEL_PATH.is_file():
        TRANSLATE_MODEL_PATH.write_text("{}", encoding="utf-8")

    default_cfg = json.loads(DEFAULT_CFG_PATH.read_text(encoding="utf-8"))
    user_cfg = json.loads(USER_CFG_PATH.read_text(encoding="utf-8"))
    merged = deep_update(default_cfg, user_cfg)
    if merged != user_cfg:
        USER_CFG_PATH.write_text(
            json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
        )


# ---------------------------------------------
# GUI
# ---------------------------------------------
class ConfigGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("STT 設定")
        self.resizable(False, False)

        ensure_files()
        self.default_cfg = json.loads(DEFAULT_CFG_PATH.read_text(encoding="utf-8"))
        self.choices = json.loads(CHOICES_PATH.read_text(encoding="utf-8"))
        self.user_cfg = json.loads(USER_CFG_PATH.read_text(encoding="utf-8"))

        # 動態 widgets & rows
        self.widgets = {sec: {} for sec in self.default_cfg}
        self.rows = {sec: {} for sec in self.default_cfg}
        # 載入聲卡裝置列表
        self.mic_list = [m.name for m in sc.all_microphones(include_loopback=False)]
        self.spk_list = [s.name for s in sc.all_speakers()]
        # 載入翻譯模型列表
        self.translate_model_dict = json.loads(
            TRANSLATE_MODEL_PATH.read_text(encoding="utf-8")
        )

        # Notebook
        notebook = ttk.Notebook(self)
        notebook.pack(padx=10, pady=10)

        # 各 section tab
        for section in self.default_cfg:
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=section)
            self._build_section(section, frame)

        # 按鈕區
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=8)
        self.run_btn = ttk.Button(btn_frame, text="▶ 執行", command=self._on_run)
        self.run_btn.pack(side="left", padx=5)
        self.stop_btn = ttk.Button(
            btn_frame, text="⛔ 停止", command=self._on_stop, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=5)
        self.proc = None
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.reset_all_btn = ttk.Button(
            btn_frame, text="🔄 還原全部預設", command=self._restore_all
        )
        self.reset_all_btn.pack(side="left", padx=5)

    def _restore_section(self, section: str):
        # 1) 覆蓋 user_cfg 的該節
        defaults = self.default_cfg[section]
        self.user_cfg[section] = deepcopy(defaults)
        # 2) 寫回檔案
        USER_CFG_PATH.write_text(
            json.dumps(self.user_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # 3) 只更新這一節的 widget
        self._refresh_section(section)

    def _restore_all(self):
        # 1) 整個覆蓋
        self.user_cfg = deepcopy(self.default_cfg)
        # 2) 寫回檔案
        USER_CFG_PATH.write_text(
            json.dumps(self.user_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # 3) 更新所有節的 widget
        for sec in self.default_cfg:
            self._refresh_section(sec)

    def _refresh_section(self, section: str):
        # 讀出 user_cfg 裡這節的值
        section_cfg = self.user_cfg.get(section, {})
        for key, widget in self.widgets[section].items():
            # 嘗試用 textvariable
            if "textvariable" in widget.keys():
                var_name = widget.cget("textvariable")
                # 直接透過 self.setvar 設定變數的值
                self.setvar(var_name, section_cfg.get(key))
            # 再嘗試用 variable（Checkbutton）
            elif "variable" in widget.keys():
                var_name = widget.cget("variable")
                self.setvar(var_name, section_cfg.get(key))
        # 更新可見性
        self._update_visibility(section)

    def _build_section(self, section: str, parent: ttk.Frame):
        cfg = self.default_cfg[section]

        for idx, (key, default) in enumerate(cfg.items()):
            dot_path = f"{section}.{key}"
            row = ttk.Frame(parent)
            row.grid(row=idx, column=0, sticky="w", pady=2)
            self.rows[section][key] = row
            ttk.Label(row, text=key, width=25).pack(side="left")
            choices = self.choices.get(dot_path)
            if choices is not None:
                var = tk.StringVar(value=str(self._get_user_cfg(section, key, default)))
                if dot_path == "input_config.device_name":
                    engine_type = self.user_cfg["input_config"].get(
                        "engine_type", self.default_cfg["input_config"]["engine_type"]
                    )
                    choices = self._get_device_list(engine_type)
                widget = ttk.Combobox(
                    row, textvariable=var, values=choices, state="readonly", width=18
                )
            elif isinstance(default, bool):
                var = tk.BooleanVar(value=self._get_user_cfg(section, key, default))
                widget = ttk.Checkbutton(row, variable=var)
            elif isinstance(default, float):
                var = tk.StringVar(value=str(self._get_user_cfg(section, key, default)))
                widget = tk.Spinbox(
                    row, textvariable=var, from_=-1e9, to=1e9, increment=0.1, width=19
                )
            elif isinstance(default, int):
                var = tk.StringVar(value=str(self._get_user_cfg(section, key, default)))
                widget = tk.Spinbox(
                    row, textvariable=var, from_=-1e9, to=1e9, increment=1, width=19
                )
            else:
                var = tk.StringVar(value=str(self._get_user_cfg(section, key, default)))
                widget = ttk.Entry(row, textvariable=var, width=21)

            widget.pack(side="left")
            var.trace_add(
                "write", lambda *_, p=dot_path, v=var: self._write_cfg(p, v.get())
            )
            self.widgets[section][key] = widget

            if dot_path == "input_config.engine_type":
                var.trace_add(
                    "write", lambda *_, v=var: self._on_input_engine_change(v)
                )
            elif dot_path == "translate_config.engine_type":
                var.trace_add(
                    "write", lambda *_, v=var: self._on_translate_engine_change(v)
                )

        self._update_visibility(section)

        restore_btn = ttk.Button(
            parent,
            text="🔄 還原本節預設",
            command=lambda sec=section: self._restore_section(sec),
        )
        restore_btn.grid(row=len(cfg) + 1, column=0, sticky="e", pady=5)

    def _get_device_list(self, eng):
        return self.spk_list if eng == "system" else self.mic_list

    def _on_translate_engine_change(self, var: tk.StringVar):
        val = var.get()
        self._write_cfg("translate_config.engine_type", val)
        cb = self.widgets["translate_config"]["model"]
        cb["values"] = self.translate_model_dict.get(val)
        cb.set("")

    def _on_input_engine_change(self, var: tk.StringVar):
        val = var.get()
        self._write_cfg("input_config.engine_type", val)
        cb = self.widgets["input_config"]["device_name"]
        cb["values"] = self._get_device_list(val)
        cb.set("")

    def _get_user_cfg(self, section, key, default):
        return self.user_cfg.get(section, {}).get(key, default)

    def _write_cfg(self, dot_path: str, raw):
        if isinstance(raw, str) and raw.lower() in ("true", "false"):
            val = raw.lower() == "true"
        else:
            try:
                val = int(raw)
            except:
                try:
                    val = float(raw)
                except:
                    val = raw
        deep_set(self.user_cfg, dot_path, val)
        USER_CFG_PATH.write_text(
            json.dumps(self.user_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if dot_path.endswith("engine_type"):
            section = dot_path.split(".")[0]
            self._update_visibility(section)

    def _update_visibility(self, section):
        cfg = self.default_cfg[section]
        eng = self.user_cfg.get(section, {}).get("engine_type", cfg.get("engine_type"))
        for key, row in self.rows[section].items():
            if section == "transcribe_config":
                if key == "overlap_sec":
                    row.grid() if eng == "overlap" else row.grid_remove()
                elif key == "interval_sec":
                    row.grid() if eng == "sliding" else row.grid_remove()
                else:
                    row.grid()
            elif section == "translate_config":
                if key == "target_lang":
                    row.grid_remove() if eng == "opencc" else row.grid()
                elif key == "source_lang":
                    (
                        row.grid_remove()
                        if eng in ["gemini", "ollama", "opencc"]
                        else row.grid()
                    )
                elif key == "temperature":
                    row.grid_remove() if eng == "opencc" else row.grid()
                else:
                    row.grid()
            else:
                row.grid()

    def _on_run(self):
        self._on_stop()
        USER_CFG_PATH.write_text(
            json.dumps(self.user_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        cmd = [sys.executable, "main.py", "-c", str(USER_CFG_PATH)]
        self.proc = subprocess.Popen(
            cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

    def _on_stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
                self.proc.wait(5)
            except:
                pass
        self.proc = None
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def _on_close(self):
        self._on_stop()
        self.destroy()


if __name__ == "__main__":
    app = ConfigGUI()
    app.mainloop()
