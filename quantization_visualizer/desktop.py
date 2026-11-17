"""Native WebView2 desktop window with a bounded, local Python event bridge."""
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]


class RuntimeBridge:
    def __init__(self):
        self._lock = threading.RLock()
        self._events = deque()
        self._bytes = 0
        self._seq = 0
        self._run_id = 0
        self._proc = None
        self._closing = False
        self._artifacts = set()
        self._stopped = False

    def _append(self, event, run_id):
        with self._lock:
            if self._closing or run_id != self._run_id:
                return
            self._seq += 1
            event = dict(event, seq=self._seq)
            size = len(json.dumps(event))
            self._events.append((event, size))
            self._bytes += size
            while len(self._events) > 1200 or self._bytes > 12_000_000:
                _, removed = self._events.popleft()
                self._bytes -= removed
            if event.get('type') == 'artifact':
                self._artifacts.add(str(Path(event['path']).resolve()))

    def start(self, mode='full'):
        if mode not in ('full', 'quick', 'demo'):
            return {'ok': False, 'error': 'Choose Full, Quick, or Demo.'}
        with self._lock:
            if self._closing or (self._proc and self._proc.poll() is None):
                return {'ok': False, 'error': 'A program is already running.'}
            self._run_id += 1
            run_id = self._run_id
            self._events.clear()
            self._bytes = 0
            self._artifacts.clear()
            self._stopped = False
            try:
                self._proc = subprocess.Popen(
                    [sys.executable, '-u', '-m', 'quantization_visualizer.runtime', '--mode', mode],
                    cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding='utf-8', errors='replace',
                    env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8'),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            except OSError as exc:
                return {'ok': False, 'error': str(exc)}
            threading.Thread(target=self._read, args=(self._proc, run_id), daemon=True).start()
            return {'ok': True, 'run_id': run_id, 'cursor': self._seq}

    def _read(self, proc, run_id):
        terminal = False
        try:
            for line in proc.stdout:
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise ValueError('Not an event')
                except ValueError:
                    event = {'type': 'log', 'message': line.strip()[:4000]}
                with self._lock:
                    if self._stopped or run_id != self._run_id:
                        continue
                    terminal = terminal or (event.get('type') == 'status' and event.get('status') in ('completed', 'error'))
                    self._append(event, run_id)
        finally:
            proc.stdout.close()
            code = proc.wait()
            with self._lock:
                if not terminal and not self._stopped:
                    self._append({'type': 'status', 'status': 'error',
                                  'message': f'Worker exited unexpectedly ({code}). See program messages.'}, run_id)

    def poll(self, after_seq=0):
        try:
            cursor = max(0, int(after_seq))
        except (ValueError, TypeError):
            cursor = 0
        with self._lock:
            batch = []
            size = 0
            for event, event_size in self._events:
                if event['seq'] <= cursor:
                    continue
                if batch and (len(batch) >= 80 or size + event_size > 2_000_000):
                    break
                batch.append(event)
                size += event_size
            return {'events': batch, 'latest': batch[-1]['seq'] if batch else cursor,
                    'running': bool(self._proc and self._proc.poll() is None),
                    'dropped': bool(self._events and cursor < self._events[0][0]['seq'] - 1)}

    def stop(self):
        with self._lock:
            self._stopped = True
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
            self._append({'type': 'status', 'status': 'stopped', 'message': 'Training stopped. Start begins a fresh run.'}, self._run_id)
        return {'ok': True}

    def open_artifact(self, path):
        candidate = Path(str(path)).resolve()
        with self._lock:
            allowed = str(candidate) in self._artifacts
        if not allowed or not candidate.is_file() or candidate.suffix.lower() not in ('.png', '.csv'):
            return {'ok': False, 'error': 'Choose a generated PNG or CSV from this run.'}
        try:
            os.startfile(str(candidate))
        except OSError as exc:
            return {'ok': False, 'error': str(exc)}
        return {'ok': True}

    def _close(self):
        self.stop()
        with self._lock:
            self._closing = True


def main():
    import webview
    bridge = RuntimeBridge()
    window = webview.create_window('Quantization Lab — A journey through learning',
        str(ROOT / 'quantization_visualizer' / 'web' / 'index.html'),
        js_api=bridge, width=1440, height=940, min_size=(1000, 700),
        background_color='#10101a')
    window.events.closed += bridge._close
    try:
        webview.start(gui='edgechromium', debug=False)
    finally:
        bridge._close()


if __name__ == '__main__':
    main()
