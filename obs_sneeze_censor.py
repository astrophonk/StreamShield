from __future__ import annotations

import argparse
import json
import os
import queue
import random
import threading
import time
from pathlib import Path

import sounddevice as sd
import vosk  # Speech‑to‑text (offline)
from obsws_python import ReqClient, events  # OBS WebSocket v5 SDK

DEFAULT_SWEAR_WORDS = {
    "fuck", "shit", "bitch", "asshole", "bastard", "damn", "crap",
    "dick", "piss", "bollocks", "bugger", "bloody", "hell",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live swear‑word mute & cat‑video overlay for OBS")
    p.add_argument("--obs-host", default="localhost")
    p.add_argument("--obs-port", type=int, default=4455)
    p.add_argument("--obs-password", default="")
    p.add_argument("--mic-source-name", required=True, help="Name of the microphone input in OBS")
    p.add_argument("--cat-source-name", required=True, help="Name of the Media Source for sneeze videos")
    p.add_argument("--cat-folder", type=Path, required=True, help="Folder containing sneeze‑cat video files")
    p.add_argument("--model", type=Path, default=Path("model"), help="Path to Vosk model folder")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--swear-list", type=Path, help="Optional path to custom newline‑separated swear words file")
    p.add_argument("--cat-visible-seconds", type=int, default=4,
                   help="How long (seconds) the cat video stays visible")
    return p.parse_args()


class CensorBot:
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self.silence = threading.Event()

        # ---- Connect to OBS v5 ----
        self.ws = ReqClient(host=cfg.obs_host, port=cfg.obs_port, password=cfg.obs_password)

        # ---- Speech recogniser ----
        self.vosk_model = vosk.Model(str(cfg.model))
        self.recognizer = vosk.KaldiRecognizer(self.vosk_model, cfg.sample_rate)

        # ---- Other state ----
        self.audio_queue: queue.Queue[bytes] = queue.Queue()
        self.swear_words = self._load_swear_words(cfg)
        self._prepare_scene_items()
        print("[OK] CensorBot made by AstroPhonx 🧑‍🚀")
        print("[OK] Connected and ready 🐱")

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _load_swear_words(self, cfg: argparse.Namespace) -> set[str]:
        words = set(DEFAULT_SWEAR_WORDS)
        if cfg.swear_list and cfg.swear_list.is_file():
            with cfg.swear_list.open("r", encoding="utf-8") as f:
                words.update({w.strip().lower() for w in f if w.strip()})
        return words

    def _prepare_scene_items(self) -> None:
        # Check inputs exist
        inputs = {s["inputName"] for s in self.ws.get_input_list().inputs}
        if self.cfg.mic_source_name not in inputs:
            raise SystemExit(f"[ERR] Mic source '{self.cfg.mic_source_name}' not found in OBS")
        if self.cfg.cat_source_name not in inputs:
            raise SystemExit(f"[ERR] Cat source '{self.cfg.cat_source_name}' not found in OBS")

        # Get current program scene & the scene‑item ID of the cat source
        self.scene_name = self.ws.get_current_program_scene().current_program_scene_name
        try:
            self.cat_item_id = self.ws.get_scene_item_id(
                scene_name=self.scene_name,
                source_name=self.cfg.cat_source_name,
            ).scene_item_id
        except Exception as e:
            raise SystemExit(
                f"[ERR] Source '{self.cfg.cat_source_name}' is not in the current scene "
                f"('{self.scene_name}'). Add it and hide it. → {e}")

    # ------------------------------------------------------------------
    # Audio capture
    # ------------------------------------------------------------------
    def _audio_callback(self, indata, frames, time_info, status):  # noqa: D401
        if status:
            print(status, flush=True)
        self.audio_queue.put(bytes(indata))

    # ------------------------------------------------------------------
    # Main loops
    # ------------------------------------------------------------------
    def start(self):
        self.stop_event = threading.Event()
        threading.Thread(target=self._recognition_loop, daemon=True).start()
        with sd.RawInputStream(
            samplerate=self.cfg.sample_rate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
        ):
            print("[🎙️] Listening for swear words…")
            try:
                while not self.stop_event.is_set():
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("[CTRL‑C] Exiting")

    def _recognition_loop(self):
        while not self.stop_event.is_set():
            try:
                data = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                print(result)  # Debug: print the recognition result
                text = result.get("text", "").lower()
                if text:
                    bad = set(text.split()) & self.swear_words
                    if bad:
                        print(f"[⚠️] Detected profanity: {bad}")
                        self._trigger_censor()
                    

    # ------------------------------------------------------------------
    # OBS actions
    # ------------------------------------------------------------------
    def _trigger_censor(self):
        if self.silence.is_set():
            return  # Already censoring
        self.silence.set()
        try:
            # 1. Mute mic
            self.ws.set_input_mute(self.cfg.mic_source_name, True)
            # 2. Pick random cat video & load it into source
            video_path = self._pick_cat_video()
            self._set_cat_source_file(video_path)
            # 3. Show cat overlay
            self.ws.set_scene_item_enabled(self.scene_name, self.cat_item_id, True)
            print(f"[🎞️] Sneezing cat → {Path(video_path).name}")
            time.sleep(self.cfg.cat_visible_seconds)
        finally:
            # Hide overlay & unmute
            self.ws.set_scene_item_enabled(self.scene_name, self.cat_item_id, False)
            self.ws.set_input_mute(self.cfg.mic_source_name, False)

            self.silence.clear()
            print("[✔] Mic unmuted")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _pick_cat_video(self) -> str:
        vids = [str(p) for p in self.cfg.cat_folder.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}]
        if not vids:
            raise SystemExit("[ERR] No video files found in cat folder")
        return random.choice(vids)

    def _set_cat_source_file(self, path: str) -> None:
        self.ws.set_input_settings(self.cfg.cat_source_name, {"local_file": path}, False)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    arguments = parse_args()
    bot = CensorBot(arguments)
    bot.start()
