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


def deep_set(config: dict, path: str, value):
    parts = path.split(".")
    for part in parts[:-1]:
        config = config.setdefault(part, {})
    config[parts[-1]] = value


def parse_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except (ValueError, TypeError):
            continue
    return value


def ensure_config_files():
    required_paths = {
        DEFAULT_CFG_PATH: None,
        CHOICES_PATH: "{}",
        USER_CFG_PATH: "{}",
        TRANSLATE_MODEL_PATH: "{}",
    }
    for path, default_content in required_paths.items():
        if not path.is_file():
            if default_content is None:
                messagebox.showerror("錯誤", f"找不到 {path.name}")
                sys.exit(1)
            path.write_text(default_content, encoding="utf-8")

    default_config = json.loads(DEFAULT_CFG_PATH.read_text(encoding="utf-8"))
    user_config = json.loads(USER_CFG_PATH.read_text(encoding="utf-8"))
    merged_config = deep_update(default_config, user_config)

    if merged_config != user_config:
        USER_CFG_PATH.write_text(
            json.dumps(merged_config, indent=2, ensure_ascii=False), encoding="utf-8"
        )


class ConfigGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("STT 設定")
        self.resizable(False, False)

        ensure_config_files()

        self.default_config = json.loads(DEFAULT_CFG_PATH.read_text(encoding="utf-8"))
        self.choices = json.loads(CHOICES_PATH.read_text(encoding="utf-8"))
        self.user_config = json.loads(USER_CFG_PATH.read_text(encoding="utf-8"))

        self.widgets = {section: {} for section in self.default_config}
        self.section_rows = {section: {} for section in self.default_config}

        self.mic_list = [m.name for m in sc.all_microphones(include_loopback=False)]
        self.spk_list = [s.name for s in sc.all_speakers()]
        self.translate_model_dict = json.loads(
            TRANSLATE_MODEL_PATH.read_text(encoding="utf-8")
        )

        notebook = ttk.Notebook(self)
        notebook.pack(padx=10, pady=10)

        for section in self.default_config:
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=section)
            self._build_section(section, frame)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=8)
        self.run_btn = ttk.Button(btn_frame, text="▶ 執行", command=self._on_run)
        self.run_btn.pack(side="left", padx=5)
        self.stop_btn = ttk.Button(
            btn_frame, text="⛔ 停止", command=self._on_stop, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=5)
        self.reset_all_btn = ttk.Button(
            btn_frame, text="🔄 還原全部預設", command=self._restore_all
        )
        self.reset_all_btn.pack(side="left", padx=5)

        self.proc = None
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_section(self, section: str, parent: ttk.Frame):
        config = self.default_config[section]

        for idx, (key, default_value) in enumerate(config.items()):
            path = f"{section}.{key}"
            row = ttk.Frame(parent)
            row.grid(row=idx, column=0, sticky="w", pady=2)
            self.section_rows[section][key] = row

            ttk.Label(row, text=key, width=25).pack(side="left")

            choices = self.choices.get(path)
            current_value = self._get_config_value(section, key, default_value)

            if path == "input_config.device_name":
                engine_type = self.user_config["input_config"].get(
                    "engine_type", self.default_config["input_config"]["engine_type"]
                )
                choices = self._get_device_list(engine_type)

            if choices is not None:
                var = tk.StringVar(value=str(current_value))
                widget = ttk.Combobox(
                    row, textvariable=var, values=choices, state="readonly", width=18
                )
            elif isinstance(default_value, bool):
                var = tk.BooleanVar(value=current_value)
                widget = ttk.Checkbutton(row, variable=var)
            elif isinstance(default_value, (int, float)):
                var = tk.StringVar(value=str(current_value))
                increment = 1 if isinstance(default_value, int) else 0.1
                widget = tk.Spinbox(
                    row,
                    textvariable=var,
                    from_=-1e9,
                    to=1e9,
                    increment=increment,
                    width=19,
                )
            else:
                var = tk.StringVar(value=str(current_value))
                widget = ttk.Entry(row, textvariable=var, width=21)

            widget.pack(side="left")
            var.trace_add(
                "write", lambda *_, p=path, v=var: self._write_config(p, v.get())
            )

            self.widgets[section][key] = widget

            if path == "input_config.engine_type":
                var.trace_add(
                    "write", lambda *_, v=var: self._on_input_engine_change(v)
                )
            elif path == "translate_config.engine_type":
                var.trace_add(
                    "write", lambda *_, v=var: self._on_translate_engine_change(v)
                )

        self._update_section_visibility(section)

        restore_btn = ttk.Button(
            parent,
            text="🔄 還原本節預設",
            command=lambda sec=section: self._restore_section(sec),
        )
        restore_btn.grid(row=len(config) + 1, column=0, sticky="e", pady=5)

    def _get_device_list(self, engine_type):
        return self.spk_list if engine_type == "system" else self.mic_list

    def _on_translate_engine_change(self, var):
        val = var.get()
        self._write_config("translate_config.engine_type", val)
        cb = self.widgets["translate_config"]["model"]
        cb["values"] = self.translate_model_dict.get(val)
        cb.set("")

    def _on_input_engine_change(self, var):
        val = var.get()
        self._write_config("input_config.engine_type", val)
        cb = self.widgets["input_config"]["device_name"]
        cb["values"] = self._get_device_list(val)
        cb.set("")

    def _get_config_value(self, section, key, default):
        return self.user_config.get(section, {}).get(key, default)

    def _write_config(self, path: str, raw_value):
        value = parse_value(raw_value)
        deep_set(self.user_config, path, value)
        USER_CFG_PATH.write_text(
            json.dumps(self.user_config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if path.endswith("engine_type"):
            section = path.split(".")[0]
            self._update_section_visibility(section)

    def _update_section_visibility(self, section):
        engine_type = self.user_config.get(section, {}).get(
            "engine_type", self.default_config[section].get("engine_type")
        )
        for key, row in self.section_rows[section].items():
            visible = True
            if section == "transcribe_config":
                if key == "overlap_sec":
                    visible = engine_type == "overlap"
                elif key == "interval_sec":
                    visible = engine_type == "sliding"
                elif key != "engine_type":
                    visible = engine_type != "funasr"
            elif section == "translate_config":
                if key == "target_lang":
                    visible = engine_type != "opencc"
                elif key == "source_lang":
                    visible = engine_type not in ["gemini", "ollama", "opencc"]
                elif key == "temperature":
                    visible = engine_type != "opencc"
            elif section == "input_config" and key == "device_name":
                visible = engine_type != "socket"
            elif section == "output_config" and key in (
                "transparent_bg",
                "font_size",
                "font_color",
                "wrap_length",
            ):
                visible = engine_type != "socket"

            row.grid() if visible else row.grid_remove()

    def _restore_section(self, section: str):
        self.user_config[section] = deepcopy(self.default_config[section])
        USER_CFG_PATH.write_text(
            json.dumps(self.user_config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._refresh_section(section)

    def _restore_all(self):
        self.user_config = deepcopy(self.default_config)
        USER_CFG_PATH.write_text(
            json.dumps(self.user_config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        for section in self.default_config:
            self._refresh_section(section)

    def _refresh_section(self, section: str):
        section_config = self.user_config.get(section, {})
        for key, widget in self.widgets[section].items():
            if "textvariable" in widget.keys():
                var_name = widget.cget("textvariable")
                self.setvar(var_name, section_config.get(key))
            elif "variable" in widget.keys():
                var_name = widget.cget("variable")
                self.setvar(var_name, section_config.get(key))
        self._update_section_visibility(section)

    def _on_run(self):
        self._on_stop()
        USER_CFG_PATH.write_text(
            json.dumps(self.user_config, indent=2, ensure_ascii=False), encoding="utf-8"
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
                self.proc.wait()
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
