"""Optional owned mpv process and nonblocking JSON IPC for annotation review.

The GUI polls every 50ms. No audio framework, worker thread, or production
capture component participates. See https://mpv.io/manual/stable/#json-ipc.
"""

import json
import math
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path


class MpvPlayback:
    def __init__(
        self,
        connection: socket.socket,
        process: subprocess.Popen[bytes] | None = None,
        directory: tempfile.TemporaryDirectory[str] | None = None,
    ) -> None:
        self.connection = connection
        self.process = process
        self.directory = directory
        self.connection.setblocking(False)
        self.position: float | None = None
        self.paused = True
        self.start_seconds = 0.0
        self.end_seconds = 0.0
        self._received = bytearray()
        self._outgoing = bytearray()
        self._request_id = 0
        self._pending_seek: int | None = None
        self._desired_seek = 0.0
        self._closed = False
        self.command(["observe_property", 1, "time-pos"])
        self.command(["observe_property", 2, "pause"])

    @classmethod
    def start(cls, audio: Path) -> MpvPlayback | None:
        executable = shutil.which("mpv")
        if executable is None:
            return None
        directory = tempfile.TemporaryDirectory(prefix="spectra-mpv-")
        ipc = Path(directory.name) / "ipc.sock"
        process: subprocess.Popen[bytes] | None = None
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            process = subprocess.Popen(
                [
                    executable,
                    "--no-config",
                    "--no-video",
                    "--force-window=no",
                    "--audio-display=no",
                    "--terminal=no",
                    "--input-terminal=no",
                    "--idle=yes",
                    "--keep-open=yes",
                    "--pause=yes",
                    f"--input-ipc-server={ipc}",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            deadline = time.monotonic() + 3
            while True:
                if process.poll() is not None:
                    raise RuntimeError("mpv exited before opening its IPC socket.")
                try:
                    connection.connect(str(ipc))
                    break
                except FileNotFoundError, ConnectionRefusedError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("mpv IPC startup timed out.") from None
                    time.sleep(0.02)
            player = cls(connection, process, directory)
            player.command(["loadfile", str(audio.resolve()), "replace"])
            return player
        except BaseException:
            connection.close()
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
            directory.cleanup()
            raise

    def command(self, values: list[str | float | int | bool]) -> int:
        self._request_id += 1
        message = json.dumps({"command": values, "request_id": self._request_id}, allow_nan=False) + "\n"
        self._outgoing.extend(message.encode())
        if len(self._outgoing) > 65_536:
            raise RuntimeError("mpv IPC is not consuming commands.")
        self._flush()
        return self._request_id

    def _flush(self) -> None:
        while self._outgoing:
            try:
                sent = self.connection.send(self._outgoing)
            except BlockingIOError:
                return
            if sent == 0:
                raise RuntimeError("mpv IPC connection closed.")
            del self._outgoing[:sent]

    def pause(self) -> None:
        self.command(["set_property", "pause", True])
        self.paused = True

    def seek(self, timestamp: float) -> None:
        if not math.isfinite(timestamp):
            raise ValueError("Playback seeks require finite timestamps.")
        timestamp = max(self.start_seconds, min(timestamp, self.end_seconds))
        self._desired_seek = timestamp
        self.position = None
        self._pending_seek = self.command(["seek", timestamp, "absolute+exact"])

    def interval(self, start: float, end: float) -> None:
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
            raise ValueError("Playback intervals must be finite and increasing.")
        self.start_seconds, self.end_seconds = start, end
        self.pause()
        self.seek(start)

    def toggle(self) -> None:
        if self.paused and self.position is not None and self.position >= self.end_seconds:
            self.seek(self.start_seconds)
        self.command(["set_property", "pause", not self.paused])
        self.paused = not self.paused

    def poll(self) -> float | None:
        self._flush()
        while True:
            try:
                data = self.connection.recv(65_536)
            except BlockingIOError:
                break
            if not data:
                raise RuntimeError("mpv IPC disconnected.")
            self._received.extend(data)
            if len(self._received) > 1_048_576:
                raise RuntimeError("Invalid oversized mpv IPC message.")
            while b"\n" in self._received:
                line, _, remaining = self._received.partition(b"\n")
                self._received = bytearray(remaining)
                message = json.loads(line)
                if message.get("event") == "file-loaded":
                    self.pause()
                    self.seek(self._desired_seek)
                if message.get("request_id") == self._pending_seek and self._pending_seek is not None:
                    self._pending_seek = None
                if message.get("event") == "property-change":
                    value = message.get("data")
                    if message.get("name") == "time-pos" and type(value) in (int, float) and math.isfinite(value):
                        if self._pending_seek is None:
                            self.position = float(value)
                    elif message.get("name") == "pause" and isinstance(value, bool):
                        self.paused = value
        if self.position is not None and self.position >= self.end_seconds and not self.paused:
            self.pause()
            self.seek(self.end_seconds)
        return self.position

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            try:
                self.command(["quit"])
            except OSError, RuntimeError:
                pass
        finally:
            self.connection.close()
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=1)
            if self.directory is not None:
                self.directory.cleanup()
