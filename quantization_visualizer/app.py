"""Classroom desktop dashboard for the instrumented quantization script."""
from __future__ import annotations

import json
from collections import deque
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

from .animations import AnimationPanel

ROOT = Path(__file__).resolve().parent.parent
BG = '#101827'
PANEL = '#19253a'
TEXT = '#eef4ff'
MUTED = '#adbed6'
CYAN = '#47d9f2'


class QuantizationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Quantization Lab · Live neural network explorer')
        self.geometry('1380x960')
        self.minsize(1060, 780)
        self.configure(bg=BG)
        self.proc = None
        self.events = queue.Queue(maxsize=4000)
        self.run_id = 0
        self.terminal_status = False
        self.user_stopped = False
        self.closing = False
        self.tables = []
        self.artifacts = []
        self.history = []
        self.animation_events = deque(maxlen=12)
        self.next_animation = 0.0
        self._style()
        self._build()
        self.protocol('WM_DELETE_WINDOW', self._close)
        self.after(40, self._drain)

    def _style(self):
        s = ttk.Style(self)
        s.theme_use('clam')
        s.configure('.', font=('Segoe UI', 10), background=PANEL, foreground=TEXT)
        s.configure('TFrame', background=BG)
        s.configure('TLabel', background=BG, foreground=TEXT)
        s.configure('TButton', background='#2b466a', foreground=TEXT, padding=(14, 8))
        s.map('TButton', background=[('active', '#3d6090'), ('disabled', '#263348')])
        s.configure('TCombobox', fieldbackground='#23344d', foreground=TEXT, padding=5)
        s.map('TCombobox', fieldbackground=[('readonly', '#23344d')], foreground=[('readonly', TEXT)])
        s.configure('TNotebook', background=BG, borderwidth=0)
        s.configure('TNotebook.Tab', background='#22334e', foreground=MUTED, padding=(15, 7))
        s.map('TNotebook.Tab', background=[('selected', '#35537b')], foreground=[('selected', TEXT)])
        s.configure('Treeview', background=PANEL, fieldbackground=PANEL, foreground=TEXT, rowheight=27)
        s.configure('Treeview.Heading', background='#2b4160', foreground=TEXT)
        s.configure('Horizontal.TProgressbar', troughcolor=PANEL, background=CYAN)

    def _build(self):
        header = ttk.Frame(self, padding=(22, 15, 22, 5))
        header.pack(fill='x')
        ttk.Label(header, text='QUANTIZATION LAB', font=('Segoe UI', 22, 'bold'), foreground=CYAN).pack(side='left')
        ttk.Label(header, text='LIVE • LEARN • COMPARE', foreground=MUTED).pack(side='right')
        ttk.Label(self, text='Follow data through quantized convolutions, predictions, gradients, and weight updates.',
                  foreground=MUTED, padding=(24, 0, 24, 10)).pack(anchor='w')
        controls = ttk.Frame(self, padding=(22, 0, 22, 8))
        controls.pack(fill='x')
        self.mode = tk.StringVar(value='Full experiment')
        self.mode_box = ttk.Combobox(controls, textvariable=self.mode, state='readonly', width=27,
                                  values=['Full experiment', 'Quick real-data run', 'Illustrative demo'])
        self.mode_box.pack(side='left', padx=(0, 12))
        self.start_button = ttk.Button(controls, text='▶ Start program', command=self.start)
        self.start_button.pack(side='left', padx=(0, 8))
        self.stop_button = ttk.Button(controls, text='■ Stop', command=self.stop, state='disabled')
        self.stop_button.pack(side='left')
        self.state_text = tk.StringVar(value='Ready · choose a mode and start')
        ttk.Label(controls, textvariable=self.state_text, foreground='#ffd166').pack(side='right')
        self.mode_note = tk.StringVar()
        ttk.Label(self, textvariable=self.mode_note, foreground=MUTED, padding=(24, 0, 24, 8)).pack(anchor='w')
        self.mode_box.bind('<<ComboboxSelected>>', lambda _: self._mode_note())
        self._mode_note()

        self.stage_text = tk.StringVar(value='Waiting for the program')
        ttk.Label(self, textvariable=self.stage_text, font=('Segoe UI', 12, 'bold'),
                  padding=(24, 4, 24, 6)).pack(anchor='w')
        self.animation = AnimationPanel(self)
        self.animation.pack(fill='both', expand=True, padx=20, pady=(0, 6))

        metrics = ttk.Frame(self, padding=(22, 3, 22, 5))
        metrics.pack(fill='x')
        self.metric_vars = {}
        for i, (name, color) in enumerate([('Dataset', CYAN), ('Precision', '#bc97ff'),
                ('Epoch', '#ffcf66'), ('Batch', '#62b8ff'), ('Loss', '#ff8d9b'), ('Accuracy', '#66e4b1')]):
            cell = tk.Frame(metrics, bg=PANEL, padx=14, pady=7)
            cell.grid(row=0, column=i, sticky='ew', padx=4)
            metrics.columnconfigure(i, weight=1)
            tk.Label(cell, text=name.upper(), fg=MUTED, bg=PANEL, font=('Segoe UI', 8)).pack(anchor='w')
            value = tk.StringVar(value='—')
            tk.Label(cell, textvariable=value, fg=color, bg=PANEL, font=('Segoe UI', 14, 'bold')).pack(anchor='w')
            self.metric_vars[name] = value
        self.progress = ttk.Progressbar(self, mode='determinate')
        self.progress.pack(fill='x', padx=26, pady=(3, 6))
        self.progress_note = tk.StringVar(value='Batch progress within the current training or evaluation pass')
        ttk.Label(self, textvariable=self.progress_note, foreground=MUTED, padding=(24, 0, 24, 7)).pack(anchor='w')

        notebook = ttk.Notebook(self, height=195)
        notebook.pack(fill='both', padx=24, pady=(0, 14))
        results = ttk.Frame(notebook)
        notebook.add(results, text='Results & saved outputs')
        self.table_picker = ttk.Combobox(results, state='readonly')
        self.table_picker.pack(fill='x', pady=(4, 3))
        self.table_picker.bind('<<ComboboxSelected>>', self._select_table)
        table_frame = ttk.Frame(results)
        table_frame.pack(fill='both', expand=True)
        self.tree = ttk.Treeview(table_frame, show='headings', height=4)
        ybar = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        xbar = ttk.Scrollbar(table_frame, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        self.tree.grid(row=0, column=0, sticky='nsew')
        ybar.grid(row=0, column=1, sticky='ns')
        xbar.grid(row=1, column=0, sticky='ew')
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        outputs = ttk.Frame(results)
        outputs.pack(fill='x')
        self.file_picker = ttk.Combobox(outputs, state='readonly')
        self.file_picker.pack(side='left', fill='x', expand=True)
        ttk.Button(outputs, text='Open saved output', command=self._open_artifact).pack(side='right')

        curves = ttk.Frame(notebook)
        notebook.add(curves, text='Learning curves')
        self.curve = tk.Canvas(curves, bg=PANEL, highlightthickness=0, height=165)
        self.curve.pack(fill='both', expand=True)
        self.curve.bind('<Configure>', lambda _: self._draw_history())
        logs = ttk.Frame(notebook)
        notebook.add(logs, text='Program messages')
        self.log = tk.Text(logs, bg=PANEL, fg=MUTED, insertbackground=TEXT, wrap='word',
                           font=('Consolas', 9), relief='flat', height=8, state='disabled')
        scrollbar = ttk.Scrollbar(logs, command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.log.pack(fill='both', expand=True)
        guide = ttk.Frame(notebook, padding=12)
        notebook.add(guide, text='How to read this view')
        ttk.Label(guide, justify='left', wraplength=1150, text=(
            'Cyan = dataset access • Violet = weight quantization • Blue = forward prediction • '
            'Coral = backward gradients • Gold = optimizer update • Green = evaluation and results.\n\n'
            'Animations summarize observed program events. The network picture is a teaching diagram, not every neuron. '
            'Activations are computed outputs; weights are learned parameters. During training, gradients update the stored '
            'full-precision weights. Convolution uses their quantized versions; the fully connected layers stay full precision.\n\n'
            'Full experiment runs the original workflow. Quick mode reduces the workload using real data. '
            'Illustrative demo uses generated events and is not a training result. Batch progress resets for each pass. '
            'Reported model-size estimates from the source are theoretical, not measured disk or memory savings.'
        )).pack(anchor='nw')

    def _mode_note(self):
        notes = {'Full experiment': 'Runs the full source workflow; dataset downloads and training can take considerable time.',
                 'Quick real-data run': 'Shortened workload with real training data, for a classroom walkthrough.',
                 'Illustrative demo': 'ILLUSTRATIVE DEMO · generated events only; no actual neural network is trained.'}
        self.mode_note.set(notes[self.mode.get()])

    def start(self):
        if self.proc and self.proc.poll() is None:
            return
        self.run_id += 1
        self.terminal_status = False
        self.user_stopped = False
        self.history.clear()
        self.animation_events.clear()
        self.next_animation = 0.0
        self.tables.clear()
        self.artifacts.clear()
        self.table_picker.configure(values=[])
        self.table_picker.set('Tables will appear as the program produces them')
        self.file_picker.configure(values=[])
        self.file_picker.set('No saved outputs yet')
        self.tree.delete(*self.tree.get_children())
        for var in self.metric_vars.values():
            var.set('—')
        self.progress['value'] = 0
        self._draw_history()
        self.animation.handle_event({'type': 'status', 'status': 'starting'})
        mode = {'Full experiment': 'full', 'Quick real-data run': 'quick', 'Illustrative demo': 'demo'}[self.mode.get()]
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
        try:
            self.proc = subprocess.Popen([sys.executable, '-u', '-m', 'quantization_visualizer.runtime', '--mode', mode],
                    cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding='utf-8', errors='replace',
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except OSError as exc:
            self.state_text.set('Could not start program')
            self._log(str(exc))
            return
        self.state_text.set('Running' + (' · ILLUSTRATIVE DEMO' if mode == 'demo' else ''))
        self.stage_text.set('Starting Python and loading dependencies…')
        self.start_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.mode_box.configure(state='disabled')
        self._log('\n── New run: ' + self.mode.get() + ' ──')
        threading.Thread(target=self._read_process, args=(self.proc, self.run_id), daemon=True).start()

    def _read_process(self, proc, run_id):
        for line in proc.stdout:
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    event = {'type': 'log', 'message': line.rstrip()}
            except ValueError:
                event = {'type': 'log', 'message': line.rstrip()}
            while not self.closing:
                try:
                    self.events.put((run_id, event), timeout=0.2)
                    break
                except queue.Full:
                    pass
        proc.stdout.close()
        code = proc.wait()
        while not self.closing:
            try:
                self.events.put((run_id, {'type': '_exit', 'code': code}), timeout=0.2)
                break
            except queue.Full:
                pass

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.user_stopped = True
            self.animation_events.clear()
            self.terminal_status = True
            self.state_text.set('Stopped by user')
            self.animation.handle_event({'type': 'status', 'status': 'stopped'})
            self.stop_button.configure(state='disabled')

    def _drain(self):
        for _ in range(250):
            try:
                run_id, event = self.events.get_nowait()
            except queue.Empty:
                break
            if run_id == self.run_id:
                try:
                    self._event(event)
                except Exception as exc:
                    self._log(f'Display event error: {exc}')
        if not self.closing:
            if self.animation_events and time.monotonic() >= self.next_animation and not self.user_stopped:
                self.animation.handle_event(self.animation_events.popleft())
                self.next_animation = time.monotonic() + .20
            self.after(40, self._drain)

    def _event(self, event):
        kind = event.get('type', 'log')
        if self.user_stopped and kind != '_exit':
            return
        if kind == '_exit':
            self.animation_events.clear()
            self.start_button.configure(state='normal')
            self.stop_button.configure(state='disabled')
            self.mode_box.configure(state='readonly')
            if not self.terminal_status:
                self.state_text.set('Finished' if event['code'] == 0 else f'Program failed (exit {event["code"]}) · see messages')
            self.animation.handle_event({'type': 'status', 'status': 'finished' if event['code'] == 0 else 'stopped'})
            return
        if kind == 'activity':
            # Replay a small recent sample at a readable pace; metrics stay live.
            # Discard older activity instead of building a growing training lag.
            self.animation_events.append(event.copy())
        elif kind not in ('batch', 'epoch', 'log'):
            if kind == 'stage':
                self.animation_events.clear()
            self.animation.handle_event(event)
        if kind == 'stage':
            self.stage_text.set(event.get('message') or str(event.get('stage', '')))
        elif kind in ('batch', 'epoch', 'result'):
            for key, label in [('dataset', 'Dataset'), ('bits', 'Precision'), ('epoch', 'Epoch')]:
                if event.get(key) is not None:
                    val = str(event[key])
                    self.metric_vars[label].set(val + (' bit' if key == 'bits' else ''))
            for key, label in [('loss', 'Loss'), ('accuracy', 'Accuracy')]:
                if isinstance(event.get(key), (int, float)):
                    self.metric_vars[label].set(f'{event[key]:.3f}' if key == 'loss' else f'{event[key]:.2f}%')
            if kind == 'batch':
                batch, total = event.get('batch', 0), event.get('total_batches', 0)
                self.metric_vars['Batch'].set(f'{batch} / {total}' if total else str(batch))
                self.progress['value'] = min(100, 100 * batch / total) if total else 0
                self.progress_note.set(f'{event.get("phase", "Current pass").title()} · batch {batch}' + (f' of {total}' if total else ''))
            elif kind == 'epoch':
                self.history.append(event.copy())
                self.history = self.history[-200:]
                self._draw_history()
        elif kind == 'table':
            self.tables.append(event)
            self.tables = self.tables[-25:]
            self.table_picker.configure(values=[f'{i+1}. {x.get("title", "Results")}' for i, x in enumerate(self.tables)])
            self.table_picker.current(len(self.tables) - 1)
            self._select_table()
        elif kind == 'artifact':
            path = str(event.get('path', ''))
            if path and path not in self.artifacts:
                self.artifacts.append(path)
                self.file_picker.configure(values=self.artifacts)
                self.file_picker.current(len(self.artifacts) - 1)
        elif kind == 'status':
            status = str(event.get('status', ''))
            self.state_text.set(event.get('message') or status.title())
            if status in ('complete', 'completed', 'finished', 'error', 'failed', 'stopped'):
                self.terminal_status = True
                self.animation_events.clear()
        if event.get('message') and kind in ('stage', 'log', 'status'):
            self._log(str(event['message']))

    def _select_table(self, _=None):
        index = self.table_picker.current()
        if not 0 <= index < len(self.tables):
            return
        event = self.tables[index]
        columns = [str(x) for x in event.get('columns', [])]
        self.tree.delete(*self.tree.get_children())
        ids = [str(i) for i in range(len(columns))]
        self.tree.configure(columns=ids)
        for key, name in zip(ids, columns):
            self.tree.heading(key, text=name)
            self.tree.column(key, width=max(110, min(220, len(name) * 8)), stretch=True)
        for row in event.get('rows', [])[:200]:
            values = [row.get(c, '') for c in columns] if isinstance(row, dict) else row
            self.tree.insert('', 'end', values=[str(v) for v in values])

    def _draw_history(self):
        c = self.curve
        c.delete('all')
        w, h = max(c.winfo_width(), 400), max(c.winfo_height(), 120)
        left, right, top, bottom = 42, w - 260, 24, h - 28
        right = max(left + 80, right)
        demo = self.mode.get() == 'Illustrative demo'
        c.create_text(left, 11, anchor='w', text=('SIMULATED accuracy · not training measurements' if demo else 'Accuracy by epoch · dataset / bit width / phase'), fill=MUTED)
        for pct in (0, 50, 100):
            y = bottom - pct / 100 * (bottom - top)
            c.create_line(left, y, right, y, fill='#30415b')
            c.create_text(left - 6, y, anchor='e', text=str(pct), fill=MUTED)
        groups = {}
        for e in self.history:
            if isinstance(e.get('accuracy'), (float, int)):
                key = (e.get('dataset', '?'), e.get('bits', '?'), e.get('phase', ''), e.get('experiment', ''))
                groups.setdefault(key, []).append(e)
        colors = ['#47d9f2', '#bc97ff', '#ffcf66', '#66e4b1', '#ff8d9b', '#62b8ff', '#f8a45e', '#e587dd']
        if not groups:
            c.create_text((left + right) / 2, (top + bottom) / 2, text='Illustrative values appear here during the demo.' if demo else 'Real epoch measurements appear here during a run.', fill=MUTED)
        shown = list(groups.items())[-8:]
        max_epoch = max((int(e.get('epoch', 1)) for _, entries in shown for e in entries), default=1)
        for epoch in range(1, max_epoch + 1):
            if max_epoch > 15 and epoch not in (1, max_epoch):
                continue
            x = left + (epoch - 1) / max(1, max_epoch - 1) * (right - left)
            c.create_text(x, bottom + 11, text=str(epoch), fill=MUTED, font=('Segoe UI', 8))
        for i, (key, entries) in enumerate(shown):
            color = colors[i % len(colors)]
            pts = []
            for j, e in enumerate(entries):
                x = left + (int(e.get('epoch', j + 1)) - 1) / max(1, max_epoch - 1) * (right - left)
                y = bottom - max(0, min(100, e['accuracy'])) / 100 * (bottom - top)
                pts.extend([x, y])
                c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline='')
            if len(pts) >= 4:
                c.create_line(*pts, fill=color, width=2)
            c.create_text(right + 14, top + i * 17, anchor='w', text=f'{key[0]} · {key[1]} bit · {key[2]} {key[3]}', fill=color, font=('Segoe UI', 8))
        c.create_text((left + right) / 2, h - 6, text='Epoch → (latest 8 series shown)', fill=MUTED, font=('Segoe UI', 8))

    def _open_artifact(self):
        raw = self.file_picker.get()
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            try:
                os.startfile(str(path))
            except OSError as exc:
                self._log(f'Could not open output: {exc}')
        else:
            self._log('No existing output selected.')

    def _log(self, message):
        self.log.configure(state='normal')
        self.log.insert('end', message + '\n')
        line_count = int(self.log.index('end-1c').split('.')[0])
        if line_count > 800:
            self.log.delete('1.0', f'{line_count - 800}.0')
        self.log.see('end')
        self.log.configure(state='disabled')

    def _close(self):
        self.closing = True
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.animation.stop()
        self.destroy()


def main():
    QuantizationApp().mainloop()


if __name__ == '__main__':
    main()
