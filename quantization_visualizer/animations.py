"""Event-driven, schematic classroom animation; no training dependencies."""

import math
import time
import tkinter as tk


BG = "#101b30"
PANEL = "#18263e"
TEXT = "#e7eefb"
MUTED = "#a5b5d0"
COLORS = {
    "data": "#35d9ed", "quantize": "#bd93ff", "forward": "#66a9ff",
    "backward": "#ff8492", "update": "#ffcf69", "evaluate": "#64e2aa",
    "results": "#64e2aa",
}
EXPLANATIONS = {
    "data": "Reading image batches from the dataset and preparing tensors. These are dataset files, not database queries.",
    "quantize": "Making temporary low-bit convolution weights for this forward pass. The trainable master weights remain full precision.",
    "forward": "Images move through convolutions, ReLU, pooling and dense layers to produce 10 class scores. The diagram groups many neurons into each block.",
    "backward": "Gradients carry the loss signal backward. The straight-through estimator passes gradients through the weight quantization step.",
    "update": "Adam uses gradients to update full-precision master weights. The next forward pass quantizes those updated convolution weights again.",
    "evaluate": "Testing predictions on held-out images without changing any weights. Accuracy measures how often the predicted class matches the label.",
    "results": "Collecting measured accuracy and experiment outputs to compare bit widths. The fully connected layers still use full precision.",
}


class AnimationPanel(tk.Frame):
    """Consume runtime event dictionaries on the Tk main thread.

    Particles indicate the latest observed operation, not individual tensors or
    measured transit times. Numeric weight cells appear only if values arrive.
    """

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("bg", BG)
        super().__init__(parent, **kwargs)
        self._job = None
        self._component = None
        self._last_activity = 0.0
        self._phase = ""
        self._layer = ""
        self._bits = None
        self._values = []
        self._terminal = False
        self._dirty = True
        self._alive = True
        self.heading = tk.Label(self, text="A network, in motion", bg=BG,
                                fg=TEXT, font=("Segoe UI", 15, "bold"), anchor="w")
        self.heading.pack(fill="x", padx=16, pady=(10, 2))
        self.subtitle = tk.Label(
            self, text="Recent observed activity playback • metrics stay live • convolution weights quantized; dense layers FP32",
            bg=BG, fg=MUTED, font=("Segoe UI", 9), anchor="w")
        self.subtitle.pack(fill="x", padx=16)
        self.canvas = tk.Canvas(self, bg=BG, height=245, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(4, 0))
        self.caption = tk.Label(
            self, text="Start a run to see the actual program's activity. Colors identify operations; moving dots are illustrative.",
            bg=PANEL, fg=TEXT, font=("Segoe UI", 10), justify="left", anchor="w",
            padx=12, pady=9, wraplength=850)
        self.caption.pack(fill="x", padx=14, pady=(0, 5))
        self.legend = tk.Frame(self, bg=BG)
        self.legend.pack(fill="x", padx=14, pady=(0, 8))
        for component, label in [("data", "Data"), ("quantize", "Quantize"),
                                 ("forward", "Forward"), ("backward", "Gradients"),
                                 ("update", "Update"), ("evaluate", "Evaluate / results")]:
            tk.Label(self.legend, text="● " + label, fg=COLORS[component], bg=BG,
                     font=("Segoe UI", 9)).pack(side="left", padx=(0, 12))
        self.canvas.bind("<Configure>", self._resize)
        self.bind("<Destroy>", self._destroy, add="+")
        self._schedule()

    def _resize(self, event):
        self._dirty = True
        self.caption.configure(wraplength=max(180, event.width - 40))
        self._schedule()

    def _destroy(self, event):
        if event.widget is self:
            self.stop()
            self._alive = False

    def _schedule(self):
        if self._alive and self._job is None:
            self._job = self.after(40, self._tick)

    def stop(self):
        """Stop animation timers; safe to call repeatedly or before a new run."""
        if self._job is not None:
            try:
                self.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None
        self._last_activity = 0
        self._terminal = True
        self._dirty = True
        if self._alive:
            try:
                self._draw(time.monotonic())
            except tk.TclError:
                pass

    def handle_event(self, event):
        if not self._alive or not isinstance(event, dict):
            return
        kind = event.get("type", "")
        if kind == "status":
            status = str(event.get("status", "")).lower()
            if status in {"done", "complete", "completed", "finished", "error", "failed", "stopped", "cancelled", "idle"}:
                self.stop()
                self.caption.configure(text=event.get("message") or "Run ended. The diagram shows the last observed operation.")
                return
            if status in {"running", "starting"}:
                self._terminal = False
                self._component = None
                self._values = []
                self._bits = None
                self.caption.configure(text=event.get("message") or "Starting the program…")
        component = None
        if kind == "activity":
            component = event.get("component")
        elif kind == "batch":
            # Batch summaries do not prove a particular layer is executing.
            component = "evaluate" if event.get("phase") in {"test", "eval", "evaluate", "evaluation"} else None
        elif kind in {"result", "table", "artifact"}:
            component = "results"
        elif kind == "stage":
            self.caption.configure(text=str(event.get("message") or event.get("stage") or "Preparing the next stage."))
        if component in COLORS:
            self._terminal = False
            self._component = component
            self._last_activity = time.monotonic()
            self._phase = str(event.get("phase", ""))
            self._layer = str(event.get("layer", ""))
            self._bits = event.get("bits", self._bits)
            supplied = event.get("values")
            if isinstance(supplied, (list, tuple)):
                vals = []
                for value in supplied[:12]:
                    try:
                        number = float(value)
                        if math.isfinite(number):
                            vals.append(number)
                    except (ValueError, TypeError):
                        pass
                self._values = vals
            context = (" [" + self._layer + "]") if self._layer else ""
            explanation = EXPLANATIONS[component]
            if component == "quantize" and self._bits == 32:
                explanation = "32-bit baseline: convolution uses the full-precision weights unchanged. This gives a reference accuracy for the lower-bit experiments."
            self.caption.configure(text=explanation + context)
        self._dirty = True
        self._schedule()

    def _tick(self):
        self._job = None
        if not self._alive:
            return
        now = time.monotonic()
        active = not self._terminal and now - self._last_activity < 1.5
        if self._dirty or active:
            self._draw(now)
            self._dirty = active  # Redraw once when activity expires.
        if active:
            self._schedule()

    def _draw(self, now):
        c = self.canvas
        c.delete("all")
        width = max(c.winfo_width(), 320)
        height = max(c.winfo_height(), 160)
        active = not self._terminal and now - self._last_activity < 1.5
        comp = self._component if active else None
        color = COLORS.get(comp, MUTED)
        # At narrow sizes use two rows so labels remain readable.
        narrow = width < 720
        labels = [("Images", "3 × 32 × 32"), ("Conv 1", "32 channels"),
                  ("ReLU / pool", "16 × 16"), ("Conv 2", "64 channels"),
                  ("ReLU / pool", "8 × 8"), ("Flatten", "4,096 values"),
                  ("Dense 1", "128 · FP32"), ("Class scores", "10 · FP32")]
        cols = 4 if narrow else 8
        gap = (width - 32) / cols
        box_w = min(gap - 10, 112)
        y1 = max(88, height * (0.38 if narrow else 0.49))
        y2 = max(y1 + 64, height * 0.72)
        positions = []
        for i in range(8):
            row = i // cols
            col = i % cols
            if row:
                col = cols - 1 - col
            positions.append((16 + gap * (col + 0.5), y1 if row == 0 else y2))
        self._draw_weights(c, width, positions, comp, now)
        for i in range(7):
            x, y = positions[i]
            nx, ny = positions[i + 1]
            if y == ny:
                sign = 1 if nx > x else -1
                a, b = (x + sign * box_w / 2, y), (nx - sign * box_w / 2, ny)
            else:
                a, b = (x, y + 23), (nx, ny - 23)
            flowing = comp in {"forward", "backward", "evaluate"} or (comp == "data" and i == 0)
            reverse = comp == "backward"
            c.create_line(*a, *b, fill=color if flowing else "#354660", width=2,
                          arrow=tk.FIRST if reverse else tk.LAST)
            if flowing:
                t = ((now * 1.1) + i * 0.12) % 1
                if reverse:
                    t = 1 - t
                px, py = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
                c.create_oval(px - 3.5, py - 3.5, px + 3.5, py + 3.5, fill=color, outline="")
        for i, ((x, y), (label, detail)) in enumerate(zip(positions, labels)):
            highlighted = ((comp == "data" and i == 0)
                           or (comp == "quantize" and i in {1, 3})
                           or (comp == "update" and i in {1, 3, 6, 7})
                           or (comp == "results" and i == 7)
                           or comp in {"forward", "backward", "evaluate"})
            edge = color if highlighted else "#354660"
            c.create_rectangle(x - box_w / 2, y - 23, x + box_w / 2, y + 23,
                               fill=PANEL, outline=edge, width=2 if highlighted else 1)
            c.create_text(x, y - 7, text=label, fill=TEXT, font=("Segoe UI", 9, "bold"))
            c.create_text(x, y + 10, text=detail, fill=MUTED, font=("Segoe UI", 8))
        bottom = min(height - 13, (y2 if narrow else y1) + 47)
        state = "Latest observed: " + (self._component or "waiting").replace("_", " ")
        if not active and self._component:
            state += " · awaiting the next event" if not self._terminal else " · run ended"
        c.create_text(width / 2, bottom, text=state, fill=color, font=("Segoe UI", 9))

    def _draw_weights(self, c, width, positions, comp, now):
        bits = str(self._bits) if self._bits is not None else "k"
        text = ("FP32 master weights → " + bits + "-bit temporary weights")
        if self._bits == 32:
            text = "32-bit baseline · convolution weights pass through unchanged"
        c.create_text(width / 2, 14, text=text, fill=COLORS["quantize"], font=("Segoe UI", 9, "bold"))
        if self._values:
            cell_w = min(42, (width - 40) / len(self._values))
            start = width / 2 - len(self._values) * cell_w / 2
            for i, value in enumerate(self._values):
                x = start + i * cell_w
                # Sign colors represent sample values, not activation magnitude.
                fill = "#594880" if value >= 0 else "#23677b"
                c.create_rectangle(x + 1, 27, x + cell_w - 1, 47, fill=fill, outline="")
                c.create_text(x + cell_w / 2, 37, text=f"{value:.2f}", fill=TEXT, font=("Segoe UI", 8))
            c.create_text(width / 2, 57, text="Most recent supplied weight sample · + violet / − cyan",
                          fill=MUTED, font=("Segoe UI", 8))
        else:
            c.create_text(width / 2, 38, text="Quantization changes the weight values used by Conv 1 and Conv 2",
                          fill=MUTED, font=("Segoe UI", 9))
        if comp in {"quantize", "update"}:
            color = COLORS[comp]
            for idx in (1, 3):
                x, y = positions[idx]
                c.create_line(x, 65, x, y - 27, fill=color, width=2, arrow=tk.LAST)
                t = (now * 1.4) % 1
                py = 65 + (y - 92) * t
                c.create_oval(x - 3, py - 3, x + 3, py + 3, fill=color, outline="")
