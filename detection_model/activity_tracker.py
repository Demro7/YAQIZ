"""System activity monitoring - Global Hook Fix."""

from __future__ import annotations

import ctypes
import threading
import time
from collections import deque
from typing import Deque, Dict

from pynput import keyboard, mouse

# Global variables to ensure hooks remain active outside class scope
_mouse_listener = None
_keyboard_listener = None

class SystemActivityTracker:
    """
    Tracks user activity using separated Global Listeners to fix focus issues.
    """

    def __init__(self, idle_threshold: float = 5.0) -> None:
        self.idle_threshold = idle_threshold
        self.sample_window = 2.0  
        
        self._current_typing_cpm = 0.0
        self._current_scroll_spm = 0.0
        self.smoothing_factor = 0.15

        # Shared resources
        self._typing_timestamps: Deque[float] = deque()
        self._scroll_timestamps: Deque[float] = deque()
        self._lock = threading.Lock()
        self._last_input_time = time.time()
        self._active_window_title = "Unknown"
        self._stop_event = threading.Event()

        # Start Global Listeners (Static Start)
        self._start_listeners()

        # Active Window Monitor Thread
        self._active_window_thread = threading.Thread(
            target=self._active_window_monitor, name="ActiveWindowMonitor", daemon=True
        )
        self._active_window_thread.start()

    def _start_listeners(self):
        global _mouse_listener, _keyboard_listener
        
        # Stop if already running to prevent duplicates
        if _mouse_listener and _mouse_listener.running:
            _mouse_listener.stop()
        if _keyboard_listener and _keyboard_listener.running:
            _keyboard_listener.stop()

        _mouse_listener = mouse.Listener(
            on_scroll=self._on_scroll,
            on_move=self._on_move,
            on_click=self._on_click
        )
        _mouse_listener.start()

        _keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
        _keyboard_listener.start()

    def _on_key_press(self, key) -> None:
        try:
            now = time.time()
            with self._lock:
                self._typing_timestamps.append(now)
                self._last_input_time = now
        except: pass

    def _on_scroll(self, x, y, dx, dy) -> None:
        try:
            now = time.time()
            with self._lock:
                self._scroll_timestamps.append(now)
                self._last_input_time = now
        except: pass

    def _on_move(self, x, y):
        self._last_input_time = time.time()

    def _on_click(self, x, y, button, pressed):
        if pressed:
            self._last_input_time = time.time()

    def _active_window_monitor(self) -> None:
        while not self._stop_event.is_set():
            try:
                title = self._get_active_window_title()
                if title:
                    with self._lock:
                        self._active_window_title = title
            except: pass
            time.sleep(0.5)

    @staticmethod
    def _get_active_window_title() -> str:
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if not hwnd: return "Unknown"
            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0: return "Unknown"
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            return buffer.value if buffer.value else "Unknown"
        except: return "Unknown"

    def _clean_old_events(self, now: float) -> None:
        cutoff = now - self.sample_window
        while self._typing_timestamps and self._typing_timestamps[0] < cutoff:
            self._typing_timestamps.popleft()
        while self._scroll_timestamps and self._scroll_timestamps[0] < cutoff:
            self._scroll_timestamps.popleft()

    def get_activity_stats(self) -> Dict[str, object]:
        now = time.time()
        with self._lock:
            self._clean_old_events(now)
            
            multiplier = 60.0 / self.sample_window
            
            target_typing = len(self._typing_timestamps) * multiplier
            target_scroll = len(self._scroll_timestamps) * multiplier
            
            self._current_typing_cpm += (target_typing - self._current_typing_cpm) * self.smoothing_factor
            self._current_scroll_spm += (target_scroll - self._current_scroll_spm) * self.smoothing_factor
            
            d_type = int(self._current_typing_cpm)
            d_scroll = int(self._current_scroll_spm)
            
            if d_type < 5: d_type = 0
            if d_scroll < 5: d_scroll = 0
            
            is_idle = (now - self._last_input_time) > self.idle_threshold
            window_title = self._active_window_title

        return {
            "active_window": window_title,
            "typing_cpm": d_type,
            "scroll_spm": d_scroll,
            "is_idle": is_idle,
        }

    def cleanup(self) -> None:
        self._stop_event.set()
        global _mouse_listener, _keyboard_listener
        if _mouse_listener: _mouse_listener.stop()
        if _keyboard_listener: _keyboard_listener.stop()
        self._active_window_thread.join(timeout=1.0)
