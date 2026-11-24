/* Quantization Lab: measured network inspection with illustrative signal motion. */
(() => {
  'use strict';
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const finite = v => typeof v === 'number' && Number.isFinite(v);
  const fmt = v => finite(v) ? (Math.abs(v) < .0001 && v !== 0 ? v.toExponential(2) : v.toFixed(4)) : 'not sampled';
  // Values retain one visual meaning across every dataset and operation.
  const DARK_PALETTE = {positive: '#a6d6c6', negative: '#a7bdcb', accent: '#b8ded2', dim: '#66867d', ink: '#e6f0eb'};
  const LIGHT_PALETTE = {positive: '#18705c', negative: '#607d8b', accent: '#245b68', dim: '#8baca3', ink: '#233a37'};
  const PALETTES = {MNIST: DARK_PALETTE, CIFAR10: DARK_PALETTE};
  const LIGHT_PALETTES = {MNIST: LIGHT_PALETTE, CIFAR10: LIGHT_PALETTE};
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
      this.bits = 32;
      this.palette = LIGHT_PALETTE;
      this.light = true;
      this.component = 'welcome';
      this.stage = 'ready';
      this.paused = false;
      this.reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      this.active = false;
      this.alive = true;
      this.clock = 0;
      this.motionRate = 1;
      this.activityAt = -10;
      // A presentation reveal of the fixed architecture, never a change to the model.
      this.construction = 0;
      this.constructionTarget = 0;
      this.constructionFrom = 0;
      this.constructionElapsed = 0;
      this.constructionDuration = 0;
      this.readingElapsed = 0;
      this.readingDuration = 0;
      this.readingIndex = 0;
      this.readingShots = [];
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
      this.cameraManual = false;
      this.cameraFocus = null;
      this.shotKey = null;
      this.stageClock = 0;
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
          if (Math.abs(dx) + Math.abs(dy) > 5) { this.drag.moved = true; this.cameraManual = true; }
          this.target.x = this.drag.x + dx;
          this.target.y = this.drag.y + dy;
          if(this.paused) this.camera={...this.target};
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
        else if (e.key === 'Enter' && !this.selected) { this._select(this._firstInspectable()); e.preventDefault(); }
        else if (e.key.startsWith('Arrow')) {
          e.preventDefault();
          const current = this.selected || this._firstInspectable();
          let l = current.layerIndex, index = current.index;
          const lastLayer = Math.max(0, ...this.nodes.filter(n => this._nodeVisibility(n) > .65).map(n => n.layerIndex));
          if (e.key === 'ArrowRight') l = Math.min(lastLayer, l + 1);
          if (e.key === 'ArrowLeft') l = Math.max(0, l - 1);
          if (e.key === 'ArrowUp') index--;
          if (e.key === 'ArrowDown') index++;
          index = (index + DESCRIPTORS[l].count) % DESCRIPTORS[l].count;
          const candidates = this.nodes.filter(n => n.layerIndex === l && this._nodeVisibility(n) > .65);
          this._select(candidates.find(n => n.index === index) || candidates[0] || current);
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
      // Layer spacing uses available width; nodes retain uniform circular size.
      this.scaleX = this.width / 1120;
      this.scaleY = this.baseScale;
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
      this.cameraManual = true;
      const old = this.target.zoom, next = clamp(old * factor, .65, 4.5), ratio = next / old;
      this.target.x = p.x - this.width / 2 - (p.x - this.width / 2 - this.target.x) * ratio;
      this.target.y = p.y - this.height / 2 - (p.y - this.height / 2 - this.target.y) * ratio;
      this.target.zoom = next;
      if(this.paused) this.camera={...this.target};
      this.dirty = true;
    }
    _hit(p) {
      if (this.selected && this._inInspector(p)) return null;
      let winner = null, best = 14;
      for (const node of this.nodes) {
        if (this._nodeVisibility(node) < .65) continue;
        const n = this._project(node), distance = Math.hypot(n.x - p.x, n.y - p.y);
        const radius = node.name === 'input' ? 40 * this.baseScale * this.camera.zoom : 12;
        if (distance < radius && distance < best + (node.name === 'input' ? radius : 0)) { winner = node; best = distance; }
      }
      return winner;
    }
    _inInspector(p) { return p.x > this.width - this._inspectorWidth() - 14 && p.y > 14 && p.y < this.height - 14; }
    _inspectorWidth() { return clamp(this.width * .4, 320, 420); }

    _select(node) {
      this.cameraManual = true;
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
        this.target = {zoom: 1, x: 0, y: 0};
        this.options.onSelect?.(null);
      }
      if(this.paused) this.camera={...this.target};
      this.dirty = true;
    }

    setSnapshot(event) {
      if (!event || !Array.isArray(event.layers)) return;
      if (this.paused) { this.pendingSnapshot = event; return; }
      if (event.dataset) this.setTheme(event.dataset, this.stage);
      this.snapshot = event;
      this.bits = event.bits ?? this.bits;
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
      if (this.paused) { this.pendingActivity = event; return; }
      const previousBits = this.bits;
      this.bits = event.bits ?? this.bits;
      if (event.type === 'status') {
        this.active = ['running', 'starting'].includes(event.status);
        if (!this.active) this.activityAt = -10;
      } else {
        this.active = true;
        this.component = event.component || (event.phase === 'evaluate' ? 'evaluate' : this.component);
        this.activityAt = this.clock;
      }
      if (previousBits !== this.bits && this.readingElapsed > 0) this._startReading();
      this.dirty = true;
    }
    setTheme(dataset = 'MNIST', stage = 'forward') {
      const nextDataset = String(dataset).toUpperCase().replace(/[^A-Z0-9]/g, '') === 'CIFAR10' ? 'CIFAR10' : 'MNIST';
      const datasetChanged = nextDataset !== this.dataset;
      const stageChanged = stage !== this.stage;
      if (datasetChanged) {
        this.snapshot = null;
        this.layers.clear();
        this.imageCanvas = null;
        for (const node of this.nodes) node.value = 0;
      }
      this.dataset = nextDataset;
      this.light = true;
      this.palette = LIGHT_PALETTE;
      this.stage = stage;
      if (datasetChanged || stage === 'welcome') this.resetConstruction();
      if (stageChanged || datasetChanged) {
        this.stageClock = this.clock;
        this.shotKey = null;
        if (!this.selected && !this.drag) this.cameraManual = false;
        if (stage === 'quantize') this.setConstructionProgress(Math.max(this.constructionTarget, 2));
        if (stage === 'forward') this.setConstructionProgress(4);
        // A user can jump straight to these chapters or replay only recent work.
        if (['backward', 'update', 'evaluate', 'results'].includes(stage)) this.setConstructionProgress(4, {immediate: true});
        this._startReading();
      }
      this.dirty = true;
    }
    // 0 = input; 1 = edges; 2 = shapes; 3 = combined clues; 4 = class scores.
    // Fractional values allow either a scroll driver or a timed story to reveal
    // channels one by one while retaining their measured values and identity.
    setConstructionProgress(value, options = {}) {
      const next = clamp(Number(value) || 0, 0, 4);
      if (next === this.constructionTarget && !options.immediate) return;
      this.constructionFrom = this.construction;
      this.constructionTarget = next;
      this.constructionElapsed = 0;
      this.constructionDuration = Math.max(.45, Number(options.duration) || Math.abs(next - this.construction) * 2.4);
      if (options.immediate) this.construction = next;
      if (this.selected && this._nodeVisibility(this.selected) < .65) this._select(null);
      this.dirty = true;
    }
    resetConstruction() { this.setConstructionProgress(0, {immediate: true}); }
    _nodeVisibility(node) {
      if (node.layerIndex === 0) return 1;
      const progress = this.construction - node.layerIndex + 1;
      const stagger = node.index / Math.max(1, DESCRIPTORS[node.layerIndex].count - 1) * .65;
      return clamp((progress - stagger) / .35, 0, 1);
    }
    _firstInspectable() {
      return this.nodes.find(n => n.name === 'conv1' && this._nodeVisibility(n) > .65) || this.nodes[0];
    }
    setPaused(value) {
      this.paused = Boolean(value);
      if (!this.paused && this.pendingSnapshot) {
        const pending = this.pendingSnapshot; this.pendingSnapshot = null; this.setSnapshot(pending);
      }
      if (!this.paused && this.pendingActivity) {
        const pending = this.pendingActivity; this.pendingActivity = null; this.setActivity(pending);
      }
      this.dirty = true;
    }
    setSpeed(value) { this.motionRate = clamp(Number(value) || 1, .25, 3); }
    setReducedMotion(value) {
      this.reduced = Boolean(value);
      if (this.reduced) {
        if (!this.selected && !this.cameraManual) { this.target = {zoom: 1, x: 0, y: 0}; this.camera = {...this.target}; }
      }
      this.shotKey = null;
      if(this.reduced) this.options.onCameraShot?.({stage:this.stage,layer:null,building:false});
      this.dirty = true;
    }
    resetView() { this.target = {zoom: 1, x: 0, y: 0}; this._select(null); this.cameraFocus=null; this.options.onCameraShot?.({stage:this.stage,layer:null,building:false}); }
    destroy() {
      this.alive = false;
      cancelAnimationFrame(this.frame);
      this.resizeObserver.disconnect();
      for (const [type, fn, options] of this.listeners) this.canvas.removeEventListener(type, fn, options);
    }

    // Reading time uses unscaled elapsed seconds: visual speed cannot rush text.
    _startReading() {
      const layers = [];
      for (let layer = Math.floor(this.construction) + 1; layer <= this.constructionTarget; layer++) layers.push({layer, building:true});
      const routes = {welcome:[0],data:[0],quantize:[1,2],forward:[1,2,3,4],backward:[4,3,2,1],update:[2,3,1,4],evaluate:[0,4]};
      const shots = layers.length ? layers : (routes[this.stage] || []).map(layer => ({layer,building:false}));
      shots.push({layer:null,building:false});
      this.readingShots = shots.map(shot => ({...shot, stage:this.stage,
        duration:clamp(Number(this.options.getReadingSeconds?.({...shot,stage:this.stage})) || 8,8,11)}));
      this.readingElapsed=0; this.readingIndex=0;
      this.readingDuration=this.readingShots.reduce((sum,shot)=>sum+shot.duration,0);
      this.shotKey=null;
    }
    isReadingComplete() { return this.readingElapsed >= this.readingDuration && this.construction >= this.constructionTarget; }
    _advanceReading(dt) {
      if(this.paused || this.selected || this.drag) return false;
      if (this.isReadingComplete()) return false;
      this.readingElapsed=Math.min(this.readingDuration,this.readingElapsed+Math.max(0,dt));
      let elapsed=this.readingElapsed, index=0;
      while(index<this.readingShots.length-1 && elapsed>=this.readingShots[index].duration) {
        const completed=this.readingShots[index++];
        elapsed-=completed.duration;
        if(completed.building) this.construction=Math.max(this.construction,Math.min(this.constructionTarget,completed.layer));
      }
      this.readingIndex=index;
      const shot=this.readingShots[index];
      if(shot?.building){
        const from=Math.max(this.constructionFrom,shot.layer-1);
        const progress=this.reduced ? 1 : clamp(elapsed/1.5,0,1);
        this.construction=Math.max(this.construction,Math.min(this.constructionTarget,from+(shot.layer-from)*progress));
      }
      this.constructionElapsed=this.readingElapsed;
      return true;
    }
    _directCamera() {
      if(this.paused || this.selected || this.drag || this.cameraManual) return;
      const shot=this.readingShots[this.readingIndex] || {stage:this.stage,layer:null,building:false};
      const {layer,building}=shot;
      const key=`${this.stage}:${this.readingIndex}:${this.width}:${this.height}`;
      if(key===this.shotKey) return;
      this.shotKey=key;
      this.cameraFocus=layer;
      this.options.onCameraShot?.({stage:this.stage,layer,building});
      if(!this.cameraManual){
        const zoom=layer===0?1.7:layer===3?1.08:layer===4?1.2:1.45;
        this.target=this.reduced || layer===null ? {zoom:1,x:0,y:0} : {
          zoom,x:-(DESCRIPTORS[layer].x-560)*this.scaleX*zoom,y:0
        };
      }
      this.hover=null;this.dirty=true;
    }

    _frame(timestamp) {
      if (!this.alive) return;
      const dt = Math.min(.05, (timestamp - (this.lastFrame || timestamp)) / 1000);
      this.lastFrame = timestamp;
      this.valueBlend = 1 - Math.exp(-dt * this.motionRate * 7.7);
      if (!this.paused && !this.reduced) this.clock += dt * this.motionRate;
      let moving = false;
      if (this._advanceReading(dt)) moving = true;
      this._directCamera();
      const manualCamera = this.cameraManual || this.selected || this.drag;
      for (const key of ['zoom', 'x', 'y']) {
        const delta = this.target[key] - this.camera[key];
        if (Math.abs(delta) > .001 && !this.paused) {
          this.camera[key] += delta * (this.reduced ? 1 : 1 - Math.exp(-dt * (manualCamera ? 10 : 2.2) * (manualCamera ? 1 : this.motionRate)));
          moving = true;
        }
      }
      if (this.dirty || moving || (!this.paused && !this.reduced && this.active)) {
        this._draw();
        this.dirty = false;
      }
      this.frame = requestAnimationFrame(t => this._frame(t));
    }

    _text(text, x, y, size = 12, color = this.palette.ink, align = 'left', weight = 400) {
      const c = this.ctx;
      size += 4;
      if (this.light && !this.inspectorDrawing && ['#8eaaa9', '#b8c3ce', '#829b98'].includes(color)) color = '#526b67';
      c.font = `${weight} ${size}px "Segoe UI", Arial, sans-serif`;
      c.fillStyle = color;
      c.textAlign = align;
      c.textBaseline = 'middle';
      c.fillText(text, x, y);
    }
    _wrap(text, x, y, maxWidth, size = 11, lineHeight = 16, color = '#b8c3ce') {
      const c = this.ctx;
      lineHeight += 6;
      c.font = `${size + 4}px "Segoe UI", Arial, sans-serif`;
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
      const visibility = Math.min(this._nodeVisibility(src), this._nodeVisibility(dst));
      if (visibility <= 0) return;
      const c = this.ctx, a = this._project(src), b = this._project(dst), bend = (b.x - a.x) * .45;
      alpha *= visibility;
      c.globalAlpha = alpha; c.strokeStyle = color; c.lineWidth = width;
      c.beginPath(); c.moveTo(a.x, a.y);
      if (visibility < 1) {
        for (let step = 1; step <= 20; step++) { const p = this._curve(a, b, visibility * step / 20); c.lineTo(p.x, p.y); }
      } else c.bezierCurveTo(a.x + bend, a.y, b.x - bend, b.y, b.x, b.y);
      c.stroke();
      if (packet !== null && packet <= visibility) {
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
      gradient.addColorStop(0, 'rgba(56,108,91,.06)');
      gradient.addColorStop(1, 'rgba(0,0,0,0)'); c.fillStyle = gradient; c.fillRect(0, 0, w, h);
      const activeStrength = this.active ? Math.max(.12, Math.exp(-(this.clock - this.activityAt) * .25)) : 0;
      const reverse = this.component === 'backward';
      this._drawOperation();
      for (const link of this.links) {
        const packet = this.active && !this.reduced && ['forward', 'backward', 'evaluate'].includes(this.component) ? (this.clock * .16 + link.seed) % 1 : null;
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
        const visibility = this._nodeVisibility(node);
        if (visibility <= 0) continue;
        const p = this._project(node);
        if (p.x < -60 || p.x > w + 60 || p.y < -60 || p.y > h + 60) continue;
        if (node.name === 'input') { this._drawInput(p); continue; }
        const targetValue = this.layers.get(node.name)?.activations?.[node.index];
        const isMeasured = finite(targetValue);
        const value = isMeasured ? targetValue : 0;
        // Time-based blending keeps measured activation changes consistent on
        // different refresh rates; paused inspection retains its frozen values.
        node.value += (value - node.value) * (this.paused ? 0 : this.reduced ? 1 : (this.valueBlend || 0));
        const strength = isMeasured ? Math.tanh(Math.abs(node.value)) : 0;
        const color = node.value < 0 ? palette.negative : palette.positive;
        const focused = this.selected === node || this.hover === node;
        const radius = clamp(node.r * this.baseScale * this.camera.zoom, 2.2, 18) * (.35 + .65 * visibility);
        if (focused || strength > .3) {
          const glow = c.createRadialGradient(p.x, p.y, radius * .2, p.x, p.y, radius * (focused ? 3.4 : 2.3));
          glow.addColorStop(0, color + (focused ? '77' : '35')); glow.addColorStop(1, color + '00');
          c.fillStyle = glow; c.beginPath(); c.arc(p.x, p.y, radius * 3.4, 0, Math.PI * 2); c.fill();
        }
        c.beginPath(); c.arc(p.x, p.y, radius, 0, Math.PI * 2);
        c.fillStyle = isMeasured ? color : '#e0e9e3'; c.globalAlpha = visibility * (isMeasured ? .3 + strength * .7 : .8); c.fill();
        c.globalAlpha = 1; c.lineWidth = focused ? 2 : .8; c.strokeStyle = focused ? palette.accent : isMeasured ? color : palette.dim; c.stroke();
        if (this.active && !this.reduced && ['quantize', 'update'].includes(this.component) && (this.component === 'update' || (this.bits < 32 && node.name.startsWith('conv')))) {
          const cycle = (this.clock * .45 - node.index * .017 + node.layerIndex * .13) % 1;
          if (cycle >= 0 && cycle < .4) {
            c.globalAlpha = visibility * (1 - cycle / .4) * .55;
            c.strokeStyle = palette.accent;
            c.beginPath(); c.arc(p.x, p.y, radius + 2 + cycle * 12, 0, Math.PI * 2); c.stroke(); c.globalAlpha = 1;
          }
        }
        if (this.camera.zoom > 2 || node.name === 'fc2') this._text(String(node.index), p.x + (node.name === 'fc2' ? radius + 10 : 0), p.y,
          node.name === 'fc2' ? 10 : 8, node.name === 'fc2' ? '#b8c3ce' : '#0b1619', node.name === 'fc2' ? 'left' : 'center', 600);
      }
      // Labels travel with the map, above/below its full topology. Never pin
      // them to the viewport, where panned neurons could pass through them.
      const labelTop = this._project({x:560,y:90}).y - 50;
      const labelBottom = this._project({x:560,y:410}).y + 28;
      for (const [layerIndex, d] of DESCRIPTORS.entries()) {
        if (layerIndex > 0 && this.construction <= layerIndex - 1) continue;
        const p = this._project({x:d.x,y:250});
        const subtitle = this.width < 1000 ? ({input:'Pixel values',conv1:'32 channels',conv2:'64 channels',fc1:'128 neurons',fc2:'10 class scores'})[d.name] : d.subtitle;
        // At extreme zoom-out omit crowded labels instead of shrinking or
        // shifting them into another layer. Inspection remains available.
        c.font='700 14px "Segoe UI", Arial';
        const spacing=Math.min(d.x-(DESCRIPTORS[layerIndex-1]?.x ?? d.x-250),(DESCRIPTORS[layerIndex+1]?.x ?? d.x+250)-d.x)*this.scaleX*this.camera.zoom;
        if(c.measureText(d.title).width+12>spacing)continue;
        this._text(d.title, p.x, labelTop, 10, palette.ink, 'center', 700);
        c.font='14px "Segoe UI", Arial';
        if(c.measureText(subtitle).width+12<=spacing)this._text(subtitle, p.x, labelTop+23, 10, '#8eaaa9', 'center');
        const caption=d.name.startsWith('conv')?`${this.snapshot?.bits ?? this.bits}-bit · ReLU / pool`:d.name==='fc1'?'float32 · ReLU':d.name==='fc2'?'float32 · raw scores':'';
        c.font='13px "Segoe UI", Arial';
        if(c.measureText(caption).width+12<=spacing)this._text(caption,p.x,labelBottom,9,palette.negative,'center');
      }
      c.fillStyle = '#f4f3ed'; c.fillRect(0, h - 27, w, 27);
      this._text(this.snapshot ? `${this.snapshot.sample || 'Sampled image'} · ${this.snapshot.sample_id !== undefined ? 'sample ' + this.snapshot.sample_id : 'latest observed values'}` : 'Waiting for a sampled image — no synthetic values are shown.', 15, h - 15, 10, '#8eaaa9');
      if (this.selected) this._drawInspector();
      else if (this.hover) this._drawTooltip();
    }

    _drawOperation() {
      const c = this.ctx;
      const building = this.construction < this.constructionTarget;
      const layerIndex = clamp(Math.ceil(this.construction), 1, 4);
      const explanations = [
        'The input image becomes a grid of pixel values.',
        'Adding edge detectors · each channel scans local 3 × 3 patches.',
        'Adding shape detectors · channels combine the earlier patterns.',
        'Adding 128 neurons · combine pooled features into evidence.',
        'Adding 10 outputs · one raw score for each possible class.'
      ];
      let caption = building ? explanations[layerIndex] : this.construction === 0 ? explanations[0] : PHASE[this.component] || 'Explore the fixed network, one layer at a time';
      if (this.component === 'quantize' && !building) caption = this.bits === 32 ? 'Full-precision baseline · convolution weights keep their original settings.' : 'Weight settings become discrete · the network connections stay in place.';
      if (this.component === 'backward') caption = 'Error signals move right to left · gradients guide the next adjustment.';
      if (this.component === 'update') caption = 'Small rings mark weight adjustments · the architecture stays the same.';
      if (this.component === 'evaluate') caption = 'Signals travel forward on held-out images · no weights are updated.';
      // The synchronized, readable explanation lives in the story panel.
      if (this.selected || this.construction === 0) return;
      let highlighted = building ? layerIndex : this.component === 'quantize' ? 1 + Math.floor(this.clock * .3) % 2 : this.component === 'evaluate' ? 4 : null;
      if (['forward', 'backward'].includes(this.component) && !building) {
        const step = Math.floor(this.clock * .65) % 4;
        highlighted = this.component === 'backward' ? 4 - step : 1 + step;
      }
      if (!this.cameraManual && !this.reduced && this.cameraFocus !== null && this.cameraFocus > 0) highlighted = this.cameraFocus;
      if (highlighted === null) return;
      const d = DESCRIPTORS[highlighted], a = this._project({x: d.x - 80, y: 80}), b = this._project({x: d.x + 80, y: 425});
      c.fillStyle = '#18705c09'; c.strokeStyle = '#18705c33'; c.lineWidth = 1;
      this._roundRect(a.x, a.y, b.x - a.x, b.y - a.y, 12); c.fill(); c.stroke();
    }

    _drawInput(p) {
      const c = this.ctx, size = clamp(78 * this.baseScale * this.camera.zoom, 58, 150);
      c.save(); this._roundRect(p.x - size / 2, p.y - size / 2, size, size, 10); c.clip();
      c.fillStyle = '#e0e9e3'; c.fillRect(p.x - size / 2, p.y - size / 2, size, size);
      if (this.imageCanvas) { c.imageSmoothingEnabled = false; c.drawImage(this.imageCanvas, p.x - size / 2, p.y - size / 2, size, size); }
      else { this._text('IMAGE', p.x, p.y, 11, '#829b98', 'center'); }
      if (this.component === 'data' && this.active) {
        const scan = this.reduced ? .5 : (this.clock * .18) % 1;
        c.fillStyle = '#18705c25';
        c.fillRect(p.x - size / 2, p.y - size / 2 + scan * (size - 8), size, 8);
      }
      c.restore();
      this._roundRect(p.x - size / 2, p.y - size / 2, size, size, 10); c.strokeStyle = this.palette.accent; c.lineWidth = 1; c.stroke();
      if (this.snapshot?.input) {
        const label = this.snapshot.input.label, pred = this.snapshot.input.prediction;
        const below = p.y + size / 2 + 22;
        const textY = below + 25 < this.height - 32 ? below : p.y - size / 2 - 39;
        if (textY >= 74 && textY + 25 < this.height - 32) {
          this._text(`Label: ${this.dataset === 'MNIST' ? label : CLASSES[label] || label}`, p.x, textY, 11, this.palette.ink, 'center');
          this._text(`Guess: ${this.dataset === 'MNIST' ? pred : CLASSES[pred] || pred}`, p.x, textY + 20, 10, '#8eaaa9', 'center');
        }
      }
    }

    _drawTooltip() {
      const node = this.hover, value = this.layers.get(node.name)?.activations?.[node.index], p = this._project(node);
      const text = node.name === 'input' ? 'The first image in this sampled batch' : `${node.name} ${node.name.startsWith('conv') ? 'channel' : 'neuron'} ${node.index} · ${fmt(value)}`;
      this.ctx.font = '15px "Segoe UI", Arial';
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
      c.fillStyle = '#0a1117'; c.fill(); c.strokeStyle = '#3c545b'; c.lineWidth = 1; c.stroke();
      c.clip();
      this._text(node.name === 'input' ? 'THE INPUT IMAGE' : `${node.name.toUpperCase()} / ${layer?.kind === 'conv' ? 'CHANNEL' : 'NEURON'} ${node.index}`, x + 18, top + 25, 12, this.palette.accent, 'left', 700);
      this._text('×', x + width - 25, top + 25, 22, '#b8c3ce', 'center');
      c.save(); c.beginPath(); c.rect(x + 1, top + 43, width - 2, Math.max(1, this.height - 130)); c.clip();
      c.translate(0, -this.inspectorScroll);
      let y = top + 53;
      const descriptions = {
        input: 'The input is the image represented as numbers. The same pixels feed every first-layer filter.',
        conv1: 'This channel uses a learned 3 × 3 filter across the image to respond to local patterns such as edges. ReLU keeps positive responses; pooling shrinks the map.',
        conv2: 'This channel combines patterns from the first convolution layer into more complex features. Its learned filters operate on feature maps, rather than the original pixels.',
        fc1: 'This neuron combines the pooled features using 4,096 weighted inputs and a bias. ReLU then keeps its positive response as evidence for the next layer.',
        fc2: 'This output neuron combines 128 hidden activations into one class score. The largest score determines the predicted class; the raw score is not a probability.'
      };
      y = this._wrap(descriptions[node.name], x + 18, y, inner, 11, 16, '#d5e2dc') + 10;
      if (!layer) {
        y = this._wrap(node.name === 'input' ? 'This is the first image of the sampled batch. Images are resized to 32 × 32 and supplied in three channels.' : 'Run the program to inspect measured activations and weights. The construction reveals the fixed architecture for teaching; training changes weights, not the number of neurons.', x + 18, y, inner);
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
      const detail = this.mapHover ? this.mapHover.kind === 'activation' ? `Neuron [${this.mapHover.row}, ${this.mapHover.col}] = ${fmt(this.mapHover.value)}` : `Weight ${this.mapHover.source} = ${fmt(this.mapHover.value)}` : 'Teal: positive · slate: negative';
      this._text(detail, x + 18, footerY + 4, 10, this.palette.ink);
      this._text(this.inspectorMaxScroll > 0 ? 'Scroll for more · brightness = magnitude' : 'Brightness = magnitude · sampled values', x + 18, footerY + 20, 9, '#8eaaa9');
      c.restore();
      this.palette = scenePalette;
      this.inspectorDrawing = false;
    }
  }
  window.NetworkView = NetworkView;
})();
