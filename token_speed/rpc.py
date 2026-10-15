import json
import os
import queue
import signal
import subprocess
import threading
import time

from . import __version__


class ProtocolError(RuntimeError):
    pass


class AppServer:
    def __init__(self, command, cwd, timeout, cancel_event):
        self.command, self.cwd, self.timeout = command, cwd, timeout
        self.cancel_event = cancel_event
        self.messages = queue.Queue()
        self.pending = []
        self.serial = 0

    def __enter__(self):
        self.proc = subprocess.Popen(
            self.command, cwd=self.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            # Do not persist stderr: it may include tokens, URLs, or private prompts.
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
            start_new_session=(os.name == "posix"), bufsize=1,
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            self.info = self.call("initialize", {
                "clientInfo": {"name": "token_speed", "title": "Token Speed", "version": __version__},
                "capabilities": {"experimentalApi": True},
            })
            self.send({"method": "initialized"})
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _read(self):
        try:
            for line in self.proc.stdout:
                received = time.perf_counter()
                self.messages.put((received, json.loads(line)))
        except Exception as exc:
            self.messages.put((time.perf_counter(), exc))
        finally:
            self.messages.put((time.perf_counter(), EOFError("app-server disconnected")))

    def send(self, message):
        self.proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def request(self, method, params):
        self.serial += 1
        self.send({"id": self.serial, "method": method, "params": params})
        return self.serial

    def receive(self, deadline):
        while True:
            if self.cancel_event.is_set():
                raise InterruptedError('benchmark cancelled')
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError('app-server deadline exceeded')
            try:
                stamp, message = self.messages.get(timeout=min(remaining, .1))
                break
            except queue.Empty:
                continue
        if isinstance(message, Exception):
            raise message
        if "id" in message and "method" in message:
            self.send({"id": message["id"], "error": {
                "code": -32601, "message": "Benchmark cannot answer interactive requests"}})
            raise ProtocolError("interactive request: " + message["method"])
        return stamp, message

    def call(self, method, params):
        return self.response(self.request(method, params),
                             time.perf_counter() + self.timeout, self.pending)

    def response(self, request_id, deadline, events):
        while True:
            stamp, message = self.receive(deadline)
            if message.get("id") == request_id:
                if "error" in message:
                    # Provider error text can contain credentials.
                    raise ProtocolError(f"RPC rejected (code={message['error']['code']})")
                return message["result"]
            if "method" in message:
                events.append((stamp, message))

    def __exit__(self, *_):
        if os.name == "posix":
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif self.proc.poll() is None:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(self.proc.pid, signal.SIGKILL)
            else:
                self.proc.kill()
            self.proc.wait()
        if os.name == "posix":
            # Also reap tools left behind if the app-server exited first.
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for stream in (self.proc.stdin, self.proc.stdout):
            stream.close()
        self.reader.join(timeout=1)
