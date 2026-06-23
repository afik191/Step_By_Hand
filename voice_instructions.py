"""Reusable spoken-instruction helper for the MultiHand project.

The newest instruction replaces any older instruction immediately.
Uses the built-in Windows System.Speech engine.
"""

import queue
import subprocess
import threading


class VoiceInstructions:
    def __init__(self, enabled=True, rate=-1, volume=95):
        self.enabled = enabled
        self.rate = max(-10, min(10, int(rate)))
        self.volume = max(0, min(100, int(volume)))
        self._queue = queue.Queue()
        self._last_key = None
        self._generation = 0
        self._speaking = threading.Event()
        self._stopped = threading.Event()
        self._process_lock = threading.Lock()
        self._current_process = None
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    @property
    def is_speaking(self):
        return self._speaking.is_set()

    def announce(self, key, text, force=False):
        if not self.enabled or self._stopped.is_set() or not text:
            return
        text = str(text).strip()
        if not text:
            return
        if not force and key == self._last_key:
            return
        self._last_key = key
        self._generation += 1
        generation = self._generation
        self._cancel_current_speech()
        self._clear_pending()
        self._queue.put((generation, text))

    def repeat(self, text):
        if not self.enabled or self._stopped.is_set() or not text:
            return
        text = str(text).strip()
        if not text:
            return
        self._generation += 1
        generation = self._generation
        self._cancel_current_speech()
        self._clear_pending()
        self._queue.put((generation, text))

    def reset(self):
        self._last_key = None

    def stop(self):
        if self._stopped.is_set():
            return
        self._stopped.set()
        self._generation += 1
        self._cancel_current_speech()
        self._clear_pending()
        self._queue.put(None)

    def _clear_pending(self):
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _cancel_current_speech(self):
        with self._process_lock:
            process = self._current_process
            self._current_process = None
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=0.8)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            generation, text = item
            if generation != self._generation or self._stopped.is_set():
                continue
            self._speak_windows(generation, text)

    def _speak_windows(self, generation, text):
        if generation != self._generation or self._stopped.is_set():
            return
        safe_text = text.replace("'", "''")
        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "try { $speaker.SelectVoiceByHints("
            "[System.Speech.Synthesis.VoiceGender]::Female,"
            "[System.Speech.Synthesis.VoiceAge]::Adult); } catch {}; "
            f"$speaker.Rate = {self.rate}; "
            f"$speaker.Volume = {self.volume}; "
            "$prompt = New-Object System.Speech.Synthesis.PromptBuilder; "
            "$prompt.AppendBreak([System.TimeSpan]::FromMilliseconds(350)); "
            f"$prompt.AppendText('{safe_text}'); "
            "$speaker.Speak($prompt);"
        )
        self._speaking.set()
        process = None
        try:
            process = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", command],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with self._process_lock:
                if generation != self._generation or self._stopped.is_set():
                    process.terminate()
                    return
                self._current_process = process
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        except Exception as error:
            print(f"Voice instruction error: {error}")
        finally:
            with self._process_lock:
                if self._current_process is process:
                    self._current_process = None
            self._speaking.clear()
