/* Quantization Lab: measured network inspection with illustrative signal motion. */
(() => {
  'use strict';
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const finite = v => typeof v === 'number' && Number.isFinite(v);
  const fmt = v => finite(v) ? (Math.abs(v) < .0001 && v !== 0 ? v.toExponential(2) : v.toFixed(4)) : 'not sampled';
  const PALETTES = {
    MNIST: {positive: '#a6f2d2', negative: '#b49bff', accent: '#d6ff69', dim: '#477b78', ink: '#e6f6eb'},
    CIFAR10: {positive: '#ffbe77', negative: '#ff74bd', accent: '#a0ddff', dim: '#955d6d', ink: '#fff1e5'}
  };
  const LIGHT_PALETTES = {
    MNIST: {positive: '#18705c', negative: '#7853ac', accent: '#245b8b', dim: '#55796c', ink: '#233a37'},
    CIFAR10: {positive: '#a54d16', negative: '#a72d69', accent: '#245b8b', dim: '#925d54', ink: '#3e2525'}
  };
  const DESCRIPTORS = [
    {name: 'input', title: 'THE IMAGE', subtitle: '3 × 32 × 32 pixels', count: 1, x: 95, cols: 1},
    {name: 'conv1', title: 'FIND EDGES', subtitle: '32 feature-map channels', count: 32, x: 290, cols: 4},
    {name: 'conv2', title: 'FIND SHAPES', subtitle: '64 feature-map channels', count: 64, x: 505, cols: 8},
    {name: 'fc1', title: 'COMBINE CLUES', subtitle: '128 individual neurons', count: 128, x: 765, cols: 8},
    {name: 'fc2', title: 'MAKE A GUESS', subtitle: '10 class-score neurons', count: 10, x: 1005, cols: 1}
  ];
  const CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
  const PHASE = {
    data: 'Images enter the network', quantize: 'Convolution weights take fewer possible values',
    forward: 'Clues flow toward a prediction', backward: 'The error signal travels backward',
    update: 'The optimizer adjusts stored weights', evaluate: 'Testing without changing weights', results: 'Comparing what each model learned'
  };

  class NetworkView {
    constructor(canvas, options = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.options = options;
      this.snapshot = null;
      this.layers = new Map();
      this.dataset = 'MNIST';
      this.palette = PALETTES.MNIST;
      this.component = 'forward';
      this.stage = 'ready';
      this.paused = false;
      this.reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      this.active = false;
      this.alive = true;
      this.clock = 0;
      this.activityAt = -10;
      this.lastFrame = 0;
      this.selected = null;
      this.inspectorScroll = 0;
      this.inspectorMaxScroll = 0;
      this.hover = null;
      this.pointer = null;
      this.drag = null;
      this.mapHover = null;
      this.hitMap = null;
      this.hitWeights = null;
      this.imageCanvas = null;
      this.camera = {zoom: 1, x: 0, y: 0};
      this.target = {...this.camera};
      this.dirty = true;
      this.nodes = [];
      this.nodeByKey = new Map();
      this.links = [];
      this.lastValues = new Map();
      this.width = 1000;
      this.height = 480;
      this._buildTopology();
      this.listeners = [];
      this._bind();
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(canvas);
      this.resize();
      this.canvas.setAttribute('tabindex', '0');
      this.canvas.setAttribute('role', 'img');
      this.canvas.setAttribute('aria-label', 'Interactive network. Convolution nodes represent channels; dense nodes represent neurons. Click to inspect. Arrow keys choose nodes, Enter selects, Escape closes, plus and minus zoom, R resets.');
      this.frame = requestAnimationFrame(t => this._frame(t));
    }

    _buildTopology() {
      for (const [layerIndex, d] of DESCRIPTORS.entries()) {
        const rows = Math.ceil(d.count / d.cols);
        for (let index = 0; index < d.count; index++) {
          const col = index % d.cols, row = Math.floor(index / d.cols);
          const dx = d.name === 'fc1' ? 17 : d.name === 'conv2' ? 15 : 21;
          const dy = d.name === 'fc1' ? 20 : d.name === 'fc2' ? 29 : 27;
          const node = {name: d.name, index, layerIndex, x: d.x + (col - (d.cols - 1) / 2) * dx,
            y: 250 + (row - (rows - 1) / 2) * dy, r: d.name === 'input' ? 39 : d.name === 'fc2' ? 8 : 5.3, value: 0};
          this.nodes.push(node);
          this.nodeByKey.set(`${d.name}:${index}`, node);
        }
      }
      // These are deliberately sparse representative routes, not an all-to-all CNN.
      for (let l = 1; l < DESCRIPTORS.length; l++) {
        const before = DESCRIPTORS[l - 1], after = DESCRIPTORS[l];
        for (let j = 0; j < Math.min(after.count, 32); j++) {
          const dst = this.nodeByKey.get(`${after.name}:${Math.floor(j * after.count / Math.min(after.count, 32))}`);
          for (let branch = 0; branch < 2; branch++) {
            const src = this.nodeByKey.get(`${before.name}:${(j * 7 + branch * 11) % before.count}`);
            this.links.push({src, dst, layer: l, seed: (j * .137 + branch * .41 + l * .2) % 1});
          }
        }
      }
    }

    _on(type, listener, options) {
      this.canvas.addEventListener(type, listener, options);
      this.listeners.push([type, listener, options]);
    }

    _bind() {
      this._on('wheel', e => {
        e.preventDefault();
        const p = this._point(e);
        if (this.selected && this._inInspector(p)) {
          this.inspectorScroll = clamp(this.inspectorScroll + e.deltaY * .55, 0, this.inspectorMaxScroll);
          this.dirty = true;
          return;
        }
        this._zoom(Math.exp(-e.deltaY * .0012), p);
      }, {passive: false});
      this._on('pointerdown', e => {
        const p = this._point(e);
        if (this.selected && this._inInspector(p)) {
          if (p.y < 65 && p.x > this.width - 54) this._select(null);
          return;
        }
        this.drag = {start: p, x: this.target.x, y: this.target.y, moved: false};
        this.canvas.setPointerCapture(e.pointerId);
      });
      this._on('pointermove', e => {
        const p = this._point(e);
        this.pointer = p;
        if (this.drag) {
          const dx = p.x - this.drag.start.x, dy = p.y - this.drag.start.y;
          if (Math.abs(dx) + Math.abs(dy) > 5) this.drag.moved = true;
          this.target.x = this.drag.x + dx;
          this.target.y = this.drag.y + dy;
          this.dirty = true;
        } else {
          this.hover = this._hit(p);
          this._mapPoint(p);
          this.canvas.style.cursor = this.hover || (this.selected && this._inInspector(p)) ? 'pointer' : 'grab';
          this.dirty = true;
        }
      });
      this._on('pointerup', e => {
        if (this.drag && !this.drag.moved) this._select(this._hit(this._point(e)));
        this.drag = null;
        if (this.canvas.hasPointerCapture(e.pointerId)) this.canvas.releasePointerCapture(e.pointerId);
      });
      this._on('pointercancel', () => { this.drag = null; });
      this._on('pointerleave', () => { this.hover = null; this.pointer = null; this.mapHover = null; this.dirty = true; });
      this._on('dblclick', () => this.resetView());
      this._on('keydown', e => {
        if (e.key === 'Escape') { this._select(null); e.preventDefault(); }
        else if (e.key.toLowerCase() === 'r') { this.resetView(); e.preventDefault(); }
        else if (e.key === '+' || e.key === '=') { this._zoom(1.2); e.preventDefault(); }
        else if (e.key === '-') { this._zoom(1 / 1.2); e.preventDefault(); }
        else if (e.key === 'Enter' && !this.selected) { this._select(this.nodeByKey.get('conv1:0')); e.preventDefault(); }
        else if (e.key.startsWith('Arrow')) {
          e.preventDefault();
          const current = this.selected || this.nodeByKey.get('conv1:0');
          let l = current.layerIndex, index = current.index;
          if (e.key === 'ArrowRight') l = Math.min(4, l + 1);
          if (e.key === 'ArrowLeft') l = Math.max(1, l - 1);
          if (e.key === 'ArrowUp') index--;
          if (e.key === 'ArrowDown') index++;
          index = (index + DESCRIPTORS[l].count) % DESCRIPTORS[l].count;
          this._select(this.nodeByKey.get(`${DESCRIPTORS[l].name}:${index}`));
        }
      });
    }

    resize() {
      const rect = this.canvas.getBoundingClientRect();
      this.width = Math.max(1, rect.width);
      this.height = Math.max(1, rect.height);
      this.dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.canvas.width = Math.round(this.width * this.dpr);
      this.canvas.height = Math.round(this.height * this.dpr);
      this.baseScale = Math.min(this.width / 1120, this.height / 500);
      this.scaleX = this.width / 1120;
      this.scaleY = this.height / 500;
      this.dirty = true;
      // ResizeObserver runs after animation frames. Repaint now so CSS layout
      // transitions don't clear the canvas immediately before every screen paint.
      this._draw();
    }

    _point(e) { const r = this.canvas.getBoundingClientRect(); return {x: e.clientX - r.left, y: e.clientY - r.top}; }
    _project(node) {
      return {x: (node.x - 560) * this.scaleX * this.camera.zoom + this.width / 2 + this.camera.x,
        y: (node.y - 250) * this.scaleY * this.camera.zoom + this.height / 2 + this.camera.y};
    }
    _zoom(factor, p = {x: this.width / 2, y: this.height / 2}) {
      const old = this.target.zoom, next = clamp(old * factor, .65, 4.5), ratio = next / old;
      this.target.x = p.x - this.width / 2 - (p.x - this.width / 2 - this.target.x) * ratio;
      this.target.y = p.y - this.height / 2 - (p.y - this.height / 2 - this.target.y) * ratio;
      this.target.zoom = next;
      this.dirty = true;
    }
    _hit(p) {
      if (this.selected && this._inInspector(p)) return null;
      let winner = null, best = 14;
      for (const node of this.nodes) {
        const n = this._project(node), distance = Math.hypot(n.x - p.x, n.y - p.y);
        const radius = node.name === 'input' ? 40 * this.baseScale * this.camera.zoom : 12;
        if (distance < radius && distance < best + (node.name === 'input' ? radius : 0)) { winner = node; best = distance; }
      }
      return winner;
    }
    _inInspector(p) { return p.x > this.width - this._inspectorWidth() - 14 && p.y > 14 && p.y < this.height - 14; }
    _inspectorWidth() { return clamp(this.width * .32, 270, 360); }

    _select(node) {
      this.selected = node;
      this.inspectorScroll = 0;
      this.mapHover = null;
      if (node) {
        const layer = this.layers.get(node.name);
        const output = {layer: node.name, index: node.index, kind: layer?.kind || (node.name === 'input' ? 'image' : 'unknown'),
          activation: layer?.activations?.[node.index], inputCount: layer?.input_count,
          shownWeights: layer?.weights?.[node.index]?.length || 0};
        this.options.onSelect?.(output);
        this.target.zoom = 1.2;
        this.target.x = this.width * .34 - this.width / 2 - (node.x - 560) * this.scaleX * this.target.zoom;
        this.target.y = -(node.y - 250) * this.scaleY * this.target.zoom;
      } else {
        this.options.onSelect?.(null);
      }
      this.dirty = true;
    }

    setSnapshot(event) {
      if (!event || !Array.isArray(event.layers)) return;
      if (this.paused) { this.pendingSnapshot = event; return; }
      if (event.dataset) this.setTheme(event.dataset, this.stage);
      this.snapshot = event;
      this.layers = new Map(event.layers.map(layer => [layer.name, layer]));
      if (event.input?.pixels?.length >= 32 * 32 * 3) {
        this.imageCanvas = document.createElement('canvas');
        this.imageCanvas.width = this.imageCanvas.height = 32;
        const ctx = this.imageCanvas.getContext('2d'), image = ctx.createImageData(32, 32);
        for (let i = 0; i < 1024; i++) {
          for (let ch = 0; ch < 3; ch++) image.data[i * 4 + ch] = event.input.pixels[i * 3 + ch];
          image.data[i * 4 + 3] = 255;
        }
        ctx.putImageData(image, 0, 0);
      }
      this.dirty = true;
    }

    setActivity(event = {}) {
      if (event.type === 'status') {
        this.active = ['running', 'starting'].includes(event.status);
        if (!this.active) this.activityAt = -10;
      } else {
        this.active = true;
        this.component = event.component || (event.phase === 'evaluate' ? 'evaluate' : this.component);
        this.activityAt = this.clock;
      }
      this.dirty = true;
    }
    setTheme(dataset = 'MNIST', stage = 'forward') {
      const nextDataset = String(dataset).toUpperCase().replace(/[^A-Z0-9]/g, '') === 'CIFAR10' ? 'CIFAR10' : 'MNIST';
      if (nextDataset !== this.dataset) {
        this.snapshot = null;
        this.layers.clear();
        this.imageCanvas = null;
        for (const node of this.nodes) node.value = 0;
      }
      this.dataset = nextDataset;
      this.light = ['data', 'update', 'evaluate', 'results'].includes(stage);
      this.palette = (this.light ? LIGHT_PALETTES : PALETTES)[this.dataset];
      this.stage = stage;
      this.dirty = true;
    }
    setPaused(value) {
      this.paused = Boolean(value);
      if (!this.paused && this.pendingSnapshot) {
        const pending = this.pendingSnapshot; this.pendingSnapshot = null; this.setSnapshot(pending);
      }
      this.dirty = true;
    }
    setReducedMotion(value) { this.reduced = Boolean(value); this.dirty = true; }
    resetView() { this.target = {zoom: 1, x: 0, y: 0}; this._select(null); }
    destroy() {
      this.alive = false;
      cancelAnimationFrame(this.frame);
      this.resizeObserver.disconnect();
      for (const [type, fn, options] of this.listeners) this.canvas.removeEventListener(type, fn, options);
    }

    _frame(timestamp) {
      if (!this.alive) return;
      const dt = Math.min(.05, (timestamp - (this.lastFrame || timestamp)) / 1000);
      this.lastFrame = timestamp;
      this.valueBlend = 1 - Math.exp(-dt * 7.7);
      if (!this.paused && !this.reduced) this.clock += dt;
      let moving = false;
      for (const key of ['zoom', 'x', 'y']) {
        const delta = this.target[key] - this.camera[key];
        if (Math.abs(delta) > .001) { this.camera[key] += delta * (this.reduced ? 1 : 1 - Math.exp(-dt * 10)); moving = true; }
      }
      if (this.dirty || moving || (!this.paused && !this.reduced && this.active)) {
        this._draw();
        this.dirty = false;
      }
      this.frame = requestAnimationFrame(t => this._frame(t));
    }

    _text(text, x, y, size = 12, color = this.palette.ink, align = 'left', weight = 400) {
      const c = this.ctx;
      if (this.light && !this.inspectorDrawing && ['#8eaaa9', '#b8c3ce', '#829b98'].includes(color)) color = '#526b67';
      c.font = `${weight} ${size}px "Segoe UI", Arial, sans-serif`;
      c.fillStyle = color;
      c.textAlign = align;
      c.textBaseline = 'middle';
      c.fillText(text, x, y);
    }
    _wrap(text, x, y, maxWidth, size = 11, lineHeight = 16, color = '#b8c3ce') {
      const c = this.ctx;
      c.font = `${size}px "Segoe UI", Arial, sans-serif`;
      let line = '';
      for (const word of String(text).split(' ')) {
        if (c.measureText(line + word).width > maxWidth && line) {
          this._text(line.trim(), x, y, size, color); y += lineHeight; line = '';
        }
        line += word + ' ';
      }
      if (line) this._text(line.trim(), x, y, size, color);
      return y + lineHeight;
    }
    _roundRect(x, y, w, h, r = 12) {
      this.ctx.beginPath(); this.ctx.roundRect(x, y, w, h, r);
    }
    _curve(a, b, t) {
      const bend = (b.x - a.x) * .45, u = 1 - t;
      return {x: u ** 3 * a.x + 3 * u * u * t * (a.x + bend) + 3 * u * t * t * (b.x - bend) + t ** 3 * b.x,
        y: u ** 3 * a.y + 3 * u * u * t * a.y + 3 * u * t * t * b.y + t ** 3 * b.y};
    }
    _drawLink(src, dst, color, alpha, width = .65, packet = null) {
      const c = this.ctx, a = this._project(src), b = this._project(dst), bend = (b.x - a.x) * .45;
      c.globalAlpha = alpha; c.strokeStyle = color; c.lineWidth = width;
      c.beginPath(); c.moveTo(a.x, a.y); c.bezierCurveTo(a.x + bend, a.y, b.x - bend, b.y, b.x, b.y); c.stroke();
      if (packet !== null) {
        const p = this._curve(a, b, packet);
        c.globalAlpha = alpha * 2.5;
        c.fillStyle = color; c.beginPath(); c.arc(p.x, p.y, 2.2, 0, Math.PI * 2); c.fill();
      }
      c.globalAlpha = 1;
    }

    _selectedEdges() {
      const node = this.selected, layer = this.layers.get(node?.name);
      if (!node || !layer?.weights?.[node.index]) return null;
      const weights = layer.weights[node.index], indices = layer.input_indices || weights.map((_, i) => i);
      const previous = DESCRIPTORS[node.layerIndex - 1];
      if (!previous) return null;
      const grouped = new Map();
      weights.forEach((w, i) => {
        let sourceIndex = indices[i];
        if (node.name === 'conv1') sourceIndex = 0;
        if (node.name === 'conv2') sourceIndex = Math.floor(sourceIndex / 9);
        if (node.name === 'fc1') sourceIndex = Math.floor(sourceIndex / 64);
        const item = grouped.get(sourceIndex) || {sum: 0, count: 0};
        item.sum += finite(w) ? w : 0; item.count++;
        grouped.set(sourceIndex, item);
      });
      return [...grouped].map(([index, item]) => ({src: this.nodeByKey.get(`${previous.name}:${index}`), dst: node, weight: item.sum / item.count})).filter(e => e.src);
    }

    _draw() {
      const c = this.ctx, w = this.width, h = this.height, palette = this.palette;
      c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0); c.clearRect(0, 0, w, h);
      const gradient = c.createRadialGradient(w * .5, h * .5, 10, w * .5, h * .5, w * .65);
      gradient.addColorStop(0, this.dataset === 'MNIST' ? 'rgba(56,108,91,.10)' : 'rgba(130,49,90,.12)');
      gradient.addColorStop(1, 'rgba(0,0,0,0)'); c.fillStyle = gradient; c.fillRect(0, 0, w, h);
      const activeStrength = this.active ? Math.max(.12, Math.exp(-(this.clock - this.activityAt) * .25)) : 0;
      const reverse = this.component === 'backward';
      for (const link of this.links) {
        const packet = this.active && !this.reduced ? (this.clock * .16 + link.seed) % 1 : null;
        const finalPacket = packet === null ? null : reverse ? 1 - packet : packet;
        this._drawLink(link.src, link.dst, palette.dim, this.selected ? .08 : .18 + activeStrength * .13, .7, finalPacket);
      }
      const selectedEdges = this._selectedEdges();
      if (selectedEdges) {
        const maxWeight = Math.max(...selectedEdges.map(e => Math.abs(e.weight)), .00001);
        for (const edge of selectedEdges) this._drawLink(edge.src, edge.dst, edge.weight < 0 ? palette.negative : palette.positive,
          .3 + Math.abs(edge.weight) / maxWeight * .4, .6 + Math.abs(edge.weight) / maxWeight * 1.7);
      }
      for (const node of this.nodes) {
        const p = this._project(node);
        if (p.x < -60 || p.x > w + 60 || p.y < -60 || p.y > h + 60) continue;
        if (node.name === 'input') { this._drawInput(p); continue; }
        const targetValue = this.layers.get(node.name)?.activations?.[node.index];
        const isMeasured = finite(targetValue);
        const value = isMeasured ? targetValue : 0;
        // Time-based blending keeps measured activation changes consistent on
        // different refresh rates; paused inspection retains its frozen values.
        node.value += (value - node.value) * (this.reduced ? 1 : this.paused ? 0 : (this.valueBlend || 0));
        const strength = isMeasured ? Math.tanh(Math.abs(node.value)) : 0;
        const color = node.value < 0 ? palette.negative : palette.positive;
        const focused = this.selected === node || this.hover === node;
        const radius = clamp(node.r * this.baseScale * this.camera.zoom, 2.2, 18);
        if (focused || strength > .3) {
          const glow = c.createRadialGradient(p.x, p.y, radius * .2, p.x, p.y, radius * (focused ? 3.4 : 2.3));
          glow.addColorStop(0, color + (focused ? '77' : '35')); glow.addColorStop(1, color + '00');
          c.fillStyle = glow; c.beginPath(); c.arc(p.x, p.y, radius * 3.4, 0, Math.PI * 2); c.fill();
        }
        c.beginPath(); c.arc(p.x, p.y, radius, 0, Math.PI * 2);
        c.fillStyle = isMeasured ? color : '#172229'; c.globalAlpha = isMeasured ? .3 + strength * .7 : .8; c.fill();
        c.globalAlpha = 1; c.lineWidth = focused ? 2 : .8; c.strokeStyle = focused ? palette.accent : isMeasured ? color : palette.dim; c.stroke();
        if (this.camera.zoom > 2 || node.name === 'fc2') this._text(String(node.index), p.x + (node.name === 'fc2' ? radius + 10 : 0), p.y,
          node.name === 'fc2' ? 10 : 8, node.name === 'fc2' ? '#b8c3ce' : '#0b1619', node.name === 'fc2' ? 'left' : 'center', 600);
      }
      for (const d of DESCRIPTORS) {
        const p = this._project({x: d.x, y: 42}), bottom = this._project({x: d.x, y: 450});
        this._text(d.title, p.x, p.y, 10, palette.ink, 'center', 700);
        this._text(d.subtitle, p.x, p.y + 18, 10, '#8eaaa9', 'center');
        if (d.name === 'conv1' || d.name === 'conv2') this._text(`${this.snapshot?.bits ?? '—'}-bit weights · ReLU → pool`, bottom.x, bottom.y, 9, palette.negative, 'center');
        if (d.name === 'fc1') this._text('float32 weights · ReLU', bottom.x, bottom.y, 9, '#8eaaa9', 'center');
        if (d.name === 'fc2') this._text('float32 · raw scores', bottom.x, bottom.y, 9, '#8eaaa9', 'center');
      }
      this._text(this.snapshot ? `${this.snapshot.sample || 'Sampled image'} · ${this.snapshot.sample_id !== undefined ? 'sample ' + this.snapshot.sample_id : 'latest observed values'}` : 'Waiting for a sampled image — no synthetic values are shown.', 15, h - 15, 10, '#8eaaa9');
      if (this.selected) this._drawInspector();
      else if (this.hover) this._drawTooltip();
    }

    _drawInput(p) {
      const c = this.ctx, size = clamp(78 * this.baseScale * this.camera.zoom, 25, 150);
      c.save(); this._roundRect(p.x - size / 2, p.y - size / 2, size, size, 10); c.clip();
      c.fillStyle = '#132026'; c.fillRect(p.x - size / 2, p.y - size / 2, size, size);
      if (this.imageCanvas) { c.imageSmoothingEnabled = false; c.drawImage(this.imageCanvas, p.x - size / 2, p.y - size / 2, size, size); }
      else { this._text('IMAGE', p.x, p.y, 11, '#829b98', 'center'); }
      c.restore();
      this._roundRect(p.x - size / 2, p.y - size / 2, size, size, 10); c.strokeStyle = this.palette.accent; c.lineWidth = 1; c.stroke();
      if (this.snapshot?.input) {
        const label = this.snapshot.input.label, pred = this.snapshot.input.prediction;
        this._text(`Label: ${this.dataset === 'MNIST' ? label : CLASSES[label] || label}`, p.x, p.y + size / 2 + 22, 11, this.palette.ink, 'center');
        this._text(`Guess: ${this.dataset === 'MNIST' ? pred : CLASSES[pred] || pred}`, p.x, p.y + size / 2 + 39, 10, '#8eaaa9', 'center');
      }
    }

    _drawTooltip() {
      const node = this.hover, value = this.layers.get(node.name)?.activations?.[node.index], p = this._project(node);
      const text = node.name === 'input' ? 'The first image in this sampled batch' : `${node.name} ${node.name.startsWith('conv') ? 'channel' : 'neuron'} ${node.index} · ${fmt(value)}`;
      this.ctx.font = '11px "Segoe UI", Arial';
      const width = this.ctx.measureText(text).width + 22, x = clamp(p.x - width / 2, 8, this.width - width - 8), y = Math.max(5, p.y - 40);
      this._roundRect(x, y, width, 27, 6); this.ctx.fillStyle = '#111c24'; this.ctx.fill();
      this._text(text, x + 11, y + 14, 11, '#e6f6eb');
    }

    _mapPoint(p) {
      this.mapHover = null;
      if (p.y < 63 || p.y > this.height - 78) return;
      for (const hit of [this.hitMap, this.hitWeights]) {
        if (!hit || p.x < hit.x || p.y < hit.y || p.x >= hit.x + hit.w || p.y >= hit.y + hit.h) continue;
        const col = Math.floor((p.x - hit.x) / hit.w * hit.cols), row = Math.floor((p.y - hit.y) / hit.h * hit.rows), index = row * hit.cols + col;
        if (index < hit.values.length) this.mapHover = {kind: hit.kind, index, col, row, value: hit.values[index], source: hit.indices?.[index] ?? index};
      }
    }

    _heatmap(values, cols, rows, x, y, width, height, kind, indices) {
      const c = this.ctx, max = Math.max(...values.map(v => Math.abs(v)), .00001);
      const cellW = width / cols, cellH = height / rows;
      c.fillStyle = '#111c24'; c.fillRect(x, y, width, height);
      for (let index = 0; index < values.length; index++) {
        const value = values[index];
        c.globalAlpha = .15 + .85 * Math.sqrt(Math.abs(value) / max);
        c.fillStyle = value < 0 ? this.palette.negative : this.palette.positive;
        c.fillRect(x + index % cols * cellW, y + Math.floor(index / cols) * cellH, Math.max(.6, cellW - .3), Math.max(.6, cellH - .3));
      }
      c.globalAlpha = 1;
      const hit = {x, y: y - this.inspectorScroll, w: width, h: height, cols, rows, values, kind, indices};
      if (kind === 'activation') this.hitMap = hit; else this.hitWeights = hit;
    }

    _drawInspector() {
      const scenePalette = this.palette;
      this.palette = PALETTES[this.dataset];
      this.inspectorDrawing = true;
      const c = this.ctx, node = this.selected, layer = this.layers.get(node.name), width = this._inspectorWidth();
      const x = this.width - width - 14, top = 14, inner = width - 36;
      this.hitMap = this.hitWeights = null;
      c.save(); this._roundRect(x, top, width, this.height - 28, 16);
      c.fillStyle = 'rgba(10,17,23,.97)'; c.fill(); c.strokeStyle = '#3c545b'; c.lineWidth = 1; c.stroke();
      c.clip();
      this._text(node.name === 'input' ? 'THE INPUT IMAGE' : `${node.name.toUpperCase()} / ${layer?.kind === 'conv' ? 'CHANNEL' : 'NEURON'} ${node.index}`, x + 18, top + 25, 12, this.palette.accent, 'left', 700);
      this._text('×', x + width - 25, top + 25, 22, '#b8c3ce', 'center');
      c.save(); c.beginPath(); c.rect(x + 1, top + 43, width - 2, Math.max(1, this.height - 130)); c.clip();
      c.translate(0, -this.inspectorScroll);
      let y = top + 53;
      if (!layer) {
        y = this._wrap(node.name === 'input' ? 'This is the first image of the sampled batch. Images are resized to 32 × 32 and supplied in three channels.' : 'Run the program to inspect measured activations and weights. The overview shows the actual architecture, but values are not yet available.', x + 18, y, inner);
        if (node.name === 'input' && this.imageCanvas) {
          c.imageSmoothingEnabled = false; c.drawImage(this.imageCanvas, x + 18, y + 10, Math.min(inner, 200), Math.min(inner, 200));
        }
      } else {
        y = this._wrap(layer.kind === 'conv' ? 'One circle is a whole feature-map channel. Each cell below is one spatial neuron before ReLU.' : 'One circle is one actual neuron. Its activation below is the raw output before ReLU (hidden layer) or the class score (output layer).', x + 18, y, inner);
        this._text(`${layer.kind === 'conv' ? 'Channel mean' : 'Raw output'}  ${fmt(layer.activations?.[node.index])}`, x + 18, y + 8, 13, this.palette.ink, 'left', 600); y += 32;
        const map = layer.maps?.[node.index];
        if (map?.length) {
          const rows = layer.shape?.[1] || Math.sqrt(map.length), cols = layer.shape?.[2] || rows;
          const mapSize = 135;
          this._heatmap(map, cols, rows, x + 18, y, mapSize, mapSize, 'activation');
          this._text(`${cols} × ${rows}`, x + 33 + mapSize, y + 12, 12, this.palette.ink);
          this._wrap('Hover a cell for its exact sampled value.', x + 33 + mapSize, y + 33, inner - mapSize - 15, 10, 14);
          y += mapSize + 20;
        }
        const weights = layer.weights?.[node.index] || [];
        if (weights.length) {
          this._text(`INCOMING WEIGHTS · ${weights.length} / ${layer.input_count}`, x + 18, y, 10, this.palette.accent, 'left', 700); y += 16;
          const cols = layer.kind === 'conv' ? 9 : 16, rows = Math.ceil(weights.length / cols);
          const mapHeight = layer.kind === 'conv' ? Math.min(128, Math.max(36, rows * 8)) : 80;
          this._heatmap(weights, cols, rows, x + 18, y, inner, mapHeight, 'weight', layer.input_indices);
          y += mapHeight + 18;
          y = this._wrap(layer.kind === 'conv' ? 'Each row is a 3 × 3 kernel from an input channel. Curves group connections by channel; the grid shows individual weights.' : node.name === 'fc1' ? '32 evenly spaced weights are sampled from 4,096 pooled inputs. Curves group those inputs by feature-map channel.' : 'All 128 incoming weights from the previous dense layer are shown. Each curve is a real connection.', x + 18, y, inner, 10, 14);
        }
      }
      this.inspectorMaxScroll = Math.max(0, y - (this.height - 92));
      c.restore();
      const footerY = this.height - 54;
      c.fillStyle = '#0a1117'; c.fillRect(x + 1, footerY - 8, width - 2, 40);
      const detail = this.mapHover ? this.mapHover.kind === 'activation' ? `Neuron [${this.mapHover.row}, ${this.mapHover.col}] = ${fmt(this.mapHover.value)}` : `Weight ${this.mapHover.source} = ${fmt(this.mapHover.value)}` : 'Mint / peach: positive · violet / pink: negative';
      this._text(detail, x + 18, footerY + 4, 10, this.palette.ink);
      this._text(this.inspectorMaxScroll > 0 ? 'Scroll this panel for more · brighter = larger magnitude' : 'Brighter = larger magnitude · sampled values', x + 18, footerY + 20, 9, '#8eaaa9');
      c.restore();
      this.palette = scenePalette;
      this.inspectorDrawing = false;
    }
  }
  window.NetworkView = NetworkView;
})();
