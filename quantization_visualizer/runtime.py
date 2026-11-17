"""Execute the original source with in-memory instrumentation; emit JSON events."""
from __future__ import annotations

import argparse
import ast
import contextlib
import datetime
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'neural_network_quantization.py'
WIRE = sys.stdout


def emit(kind, **payload):
    WIRE.write(json.dumps({'type': kind, **payload}, default=str, ensure_ascii=True) + '\n')
    WIRE.flush()


class LogStream(io.TextIOBase):
    def __init__(self):
        self.pending = ''

    def write(self, value):
        self.pending += value.replace('\r', '\n')
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            if line.strip():
                emit('log', message=line[:3000])
        if len(self.pending) > 3000:
            emit('log', message=self.pending[:3000])
            self.pending = ''
        return len(value)

    def flush(self):
        if self.pending.strip():
            emit('log', message=self.pending)
        self.pending = ''


class Instrument(ast.NodeTransformer):
    """Insert observation calls; preserve original mathematical expressions."""
    def __init__(self, quick=False):
        self.quick = quick

    @staticmethod
    def hook(name):
        return ast.Expr(ast.Call(ast.Name('_observe', ast.Load()),
                                 [ast.Constant(name), ast.Call(ast.Name('locals', ast.Load()), [], [])], []))

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.name in ('get_mnist_loaders', 'get_cifar10_loaders', 'train_epoch', 'evaluate', 'run_experiment', 'plot_results', 'create_results_table'):
            idx = 1 if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) else 0
            node.body.insert(idx, self.hook('enter:' + node.name))
        return node

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        if node.name == 'SimpleCNN':
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == '__init__':
                    method.body.append(self.hook('model_ready'))
        return node

    def visit_Return(self, node):
        return [self.hook('return'), node]

    def visit_Assign(self, node):
        self.generic_visit(node)
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if 'quantized_weight' in names:
            return [node, self.hook('quantize')]
        if 'output' in names or 'outputs' in names or 'x' in names:
            return [node, self.hook('forward')]
        if 'loss' in names:
            return [node, self.hook('loss')]
        return node

    def visit_Expr(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            method = node.value.func.attr
            if method in ('backward', 'step'):
                return [self.hook('backward' if method == 'backward' else 'update'), node]
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if self.quick:
            for kw in node.keywords:
                if kw.arg == 'epochs':
                    kw.value = ast.Constant(1)
            if isinstance(node.func, ast.Name) and node.func.id == 'range' and len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and node.args[0].value == 2:
                node.args[0] = ast.Constant(1)
        return node


def instrument_source(source, quick=False):
    tree = Instrument(quick).visit(ast.parse(source))
    # Major sections are emitted exactly where their first executable statement runs.
    headings = ('# Cell 1:', '# Test the quantization', '# Test the CNN',
                '# Load datasets and visualize', '# Test training on one',
                '# MNIST experiment', '# CIFAR-10 experiment', '# Generate plots',
                '# Generate table for paper', '# Analysis for Discussion')
    sections = [(n, line.lstrip('# ').strip()) for n, line in enumerate(source.splitlines(), 1)
                if line.startswith(headings)]
    body = []
    index = 0
    for stmt in tree.body:
        line = getattr(stmt, 'lineno', 0)
        while index < len(sections) and sections[index][0] <= line:
            label = sections[index][1]
            body.append(ast.Expr(ast.Call(ast.Name('_stage', ast.Load()), [ast.Constant(label)], [])))
            index += 1
        body.append(stmt)
    tree.body = body
    return ast.fix_missing_locations(tree)


def run_real(mode):
    emit('stage', stage='Setup', message='Importing PyTorch and scientific libraries…')
    try:
        import torch
        import torchvision  # noqa: F401
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pandas as pd
        import tqdm.notebook
    except (ImportError, OSError) as exc:
        raise RuntimeError('Scientific dependencies could not load. Run setup_visualizer.ps1, then launch_visualizer.py. ' + str(exc)) from exc

    # Small models run more efficiently with a bounded CPU thread pool.
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    quick = mode == 'quick'
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output_dir = ROOT / 'visualizer_runs' / (stamp + '-' + mode)
    output_dir.mkdir(parents=True)
    os.chdir(output_dir)
    emit('log', message=f'Outputs: {output_dir}')
    emit('log', message='Full precision master weights are trained. Only convolution weights are quantized; dense layers stay float32. Source compression figures are theoretical all-parameter estimates, not measured storage savings.')
    if quick:
        emit('log', message='QUICK REAL RUN: 256 training / 128 test examples per dataset, one epoch per model. Accuracy is not a full experiment result.')
    context = {'dataset': 'MNIST', 'bits': 32, 'epoch': 1, 'phase': 'setup', 'batch': 0, 'total_batches': 0, 'experiment': 'sanity'}
    active = {'sample': True, 'last_emit': {}, 'layer': 0,
              'tensors': {}, 'weights': {}, 'snapshot_time': 0, 'snapshot_key': None, 'sample_id': 0}
    namespace = {}

    def metadata(local=None):
        local = local or {}
        model = local.get('model', local.get('self'))
        if model is not None and hasattr(model, 'k_bits'):
            context['bits'] = model.k_bits
        if 'k_bits' in local:
            context['bits'] = local['k_bits']
        if 'epoch' in local:
            context['epoch'] = local['epoch'] + 1
        return dict(context)

    def activity(component, local, **extra):
        meta = metadata(local)
        if not active['sample']:
            return
        key = component + str(extra.get('layer', ''))
        now = time.monotonic()
        if now - active['last_emit'].get(key, 0) < .12:
            return
        active['last_emit'][key] = now
        emit('activity', **meta, component=component, **extra)

    def snapshot(local):
        """Bounded real samples: all conv activations; 32 FC1 incoming weights per neuron."""
        model = local['model']
        key = (context['dataset'], model.k_bits, context['phase'], context['experiment'])
        now = time.monotonic()
        if not active['sample'] or (key == active['snapshot_key'] and now - active['snapshot_time'] < 3):
            return
        if not all(name in active['tensors'] for name in ('conv1', 'conv2', 'fc1', 'fc2')):
            return
        active['snapshot_time'] = now
        active['snapshot_key'] = key
        active['sample_id'] += 1

        def numbers(tensor):
            return torch.nan_to_num(tensor.detach().float(), nan=0.0, posinf=1e6, neginf=-1e6).cpu().numpy().round(5).tolist()

        layers = []
        for name in ('conv1', 'conv2', 'fc1', 'fc2'):
            module = getattr(model, name)
            tensor = active['tensors'][name]
            convolution = name.startswith('conv')
            weights = active['weights'].get(name, module.weight.detach()) if convolution else module.weight.detach()
            matrix = weights.reshape(weights.shape[0], -1)
            count = matrix.shape[1]
            indices = (torch.linspace(0, count - 1, 32).long() if name == 'fc1' else torch.arange(count))
            row = dict(name=name, kind='conv' if convolution else 'dense', shape=list(tensor.shape),
                       activations=numbers(tensor.mean(dim=(1, 2)) if convolution else tensor),
                       weights=numbers(matrix[:, indices.to(matrix.device)]), input_indices=indices.tolist(),
                       input_count=count, weight_shape=list(weights.shape), activation_kind='pre-ReLU output',
                       weight_kind='quantized convolution weight' if convolution and model.k_bits != 32 else 'full-precision weight')
            if convolution:
                row['maps'] = numbers(tensor.reshape(tensor.shape[0], -1))
            layers.append(row)
        pixels = local['inputs'][0].detach().cpu()
        if context['dataset'] == 'MNIST':
            pixels = pixels * .5 + .5
        else:
            pixels = pixels * torch.tensor([.2023, .1994, .2010])[:, None, None] + torch.tensor([.4914, .4822, .4465])[:, None, None]
        pixels = (pixels.clamp(0, 1).permute(1, 2, 0) * 255).byte().reshape(-1).tolist()
        emit('snapshot', **metadata(local), layers=layers,
             input={'shape': [3, 32, 32], 'pixels': pixels, 'label': int(local['targets'][0].item()),
                    'prediction': int(local['outputs'][0].argmax().item())},
             sample='First image of sampled batch; displayed numbers rounded to 5 decimals.',
             sample_id=active['sample_id'])

    def observe(event, local):
        frame = sys._getframe(1)
        function = frame.f_code.co_name
        if event == 'model_ready':
            active['tensors'] = {}
            active['weights'] = {}
            for name in ('conv1', 'conv2', 'fc1', 'fc2'):
                def capture(module, inputs, output, layer_name=name):
                    if active['sample']:
                        active['tensors'][layer_name] = output[0].detach()
                getattr(local['self'], name).register_forward_hook(capture)
            return
        if event.startswith('enter:'):
            name = event.split(':', 1)[1]
            if name.startswith('get_'):
                context['dataset'] = 'MNIST' if 'mnist' in name else 'CIFAR10'
                active['sample'] = True
                emit('stage', **metadata(local), stage='Dataset loading', message=f"Reading {context['dataset']} from disk, downloading if missing.")
                activity('data', local)
            elif name in ('train_epoch', 'evaluate'):
                context['phase'] = 'train' if name == 'train_epoch' else 'evaluate'
                caller = frame.f_back.f_locals
                context['epoch'] = caller.get('epoch', 0) + 1
                context['batch'] = 0
                metadata(local)
                emit('stage', **metadata(local), stage='Training' if name == 'train_epoch' else 'Evaluation', message=f"{context['dataset']} · {context['bits']}-bit · epoch {context['epoch']}")
            elif name == 'run_experiment':
                context['dataset'] = local['dataset_name'].upper()
                context['experiment'] = 'experiment'
            else:
                active['sample'] = True
                activity('results', local)
            return
        if event == 'return':
            if function in ('train_epoch', 'evaluate'):
                loss = local.get('epoch_loss', local.get('test_loss'))
                acc = local.get('epoch_acc', local.get('test_acc'))
                emit('epoch', **metadata(local), loss=float(loss), accuracy=float(acc))
            elif function == 'run_experiment':
                for row in local['results']:
                    emit('result', dataset=row['Dataset'], bits=row['Bitwidth (W)'], accuracy=row['Best Test Acc (%)'])
            return
        if event == 'quantize':
            if active['sample']:
                weight = local['quantized_weight'].detach().reshape(-1)[:24].cpu().tolist()
                layer = 'conv1' if getattr(local['self'], 'in_channels', 0) == 3 else 'conv2'
                active['weights'][layer] = local['quantized_weight'].detach()
                activity('quantize', local, layer=layer, values=weight)
        elif event == 'forward':
            layer = 'network'
            if function == 'forward' and 'self' in local:
                obj = local['self']
                if hasattr(obj, 'in_channels'):
                    layer = 'conv1' if obj.in_channels == 3 else 'conv2'
                else:
                    layer = 'activations'
            activity('evaluate' if context['phase'] == 'evaluate' else 'forward', local, layer=layer)
        elif event == 'loss':
            if function in ('train_epoch', 'evaluate') and active['sample']:
                outputs, targets = local['outputs'], local['targets']
                emit('batch', **metadata(local), loss=float(local['loss'].item()), accuracy=float((outputs.argmax(1) == targets).float().mean().item() * 100))
                snapshot(local)
        else:
            activity(event, local)

    original_loader = torch.utils.data.DataLoader

    def loader(dataset, *args, **kwargs):
        kwargs['num_workers'] = 0  # No recursive Windows spawn for a notebook export.
        kwargs['pin_memory'] = torch.cuda.is_available()
        if quick:
            count = 256 if getattr(dataset, 'train', False) else 128
            dataset = torch.utils.data.Subset(dataset, range(min(count, len(dataset))))
        return original_loader(dataset, *args, **kwargs)

    # Persistent shared dataset cache, independent of each run's result directory.
    from torchvision import datasets
    for dataset_class in (datasets.MNIST, datasets.CIFAR10):
        original_init = dataset_class.__init__
        def dataset_init(self, *args, _init=original_init, **kwargs):
            kwargs['root'] = str(ROOT / 'data')
            try:
                return _init(self, *args, **kwargs)
            except urllib.error.URLError:
                if not isinstance(self, datasets.CIFAR10):
                    raise
                emit('stage', stage='Dataset download', message='Primary CIFAR-10 server unavailable; downloading checksum-verified archive from Zenodo.')
                from torchvision.datasets.utils import download_and_extract_archive
                download_and_extract_archive(
                    'https://zenodo.org/records/10089977/files/cifar-10-python.tar.gz?download=1',
                    str(ROOT / 'data'), filename=self.filename, md5=self.tgz_md5)
                return _init(self, *args, **kwargs)
        dataset_class.__init__ = dataset_init
    torch.utils.data.DataLoader = loader

    class Progress:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
        def __len__(self):
            return len(self.iterable)
        def __iter__(self):
            context['total_batches'] = len(self)
            iterator = iter(self.iterable)
            for index in range(len(self)):
                context['batch'] = index + 1
                active['sample'] = index % 20 == 0 or index == len(self) - 1
                activity('data', {})
                yield next(iterator)
        def set_postfix(self, *args, **kwargs):
            pass
    tqdm.notebook.tqdm = Progress

    def display(value):
        if isinstance(value, pd.DataFrame):
            emit('table', title='Results from original script', columns=[str(x) for x in value.columns],
                 rows=[[str(x) for x in row] for row in value.itertuples(index=False, name=None)])
        else:
            emit('log', message=str(value))

    figure_count = 0
    def show(*args, **kwargs):
        nonlocal figure_count
        for number in plt.get_fignums():
            figure_count += 1
            path = output_dir / f'figure-{figure_count}.png'
            plt.figure(number).savefig(path, dpi=120, bbox_inches='tight')
            emit('artifact', path=str(path))
        plt.close('all')
    plt.show = show
    namespace.update(__name__='__quantization_target__', __file__=str(SOURCE), display=display,
                     _observe=observe, _stage=lambda label: emit('stage', stage=label, message=label))
    tree = instrument_source(SOURCE.read_text(encoding='utf-8-sig'), quick)
    exec(compile(tree, str(SOURCE), 'exec'), namespace)
    for file in output_dir.glob('quantization_results.*'):
        emit('artifact', path=str(file))


def run_demo():
    """Deliberately illustrative events, never presented as training measurements."""
    emit('stage', stage='Illustrative demo', message='Simulated events only — no training or dataset access. Choose Full or Quick for real computation.')
    for dataset in ('MNIST', 'CIFAR10'):
        for bits in (1, 4, 8, 32):
            meta = dict(dataset=dataset, bits=bits, epoch=1, phase='train')
            emit('stage', **meta, stage='Illustrative demo', message=f'SIMULATED: {dataset}, {bits}-bit model')
            emit('snapshot', **meta, **demo_snapshot(dataset, bits))
            for batch in range(1, 5):
                for component in ('data', 'quantize', 'forward', 'backward', 'update', 'evaluate'):
                    emit('activity', **meta, component=component, layer='conv1',
                         values=[-0.5, 0.5, -0.5, 0.5] if bits == 1 else [-.8, -.3, .1, .6])
                    time.sleep(.22)
                emit('batch', **meta, batch=batch, total_batches=4, loss=2 / batch, accuracy=40 + 10 * batch)
            accuracy = dict(zip((1, 4, 8, 32), (78, 90, 92, 94) if dataset == 'MNIST' else (42, 58, 63, 67)))[bits]
            emit('epoch', **meta, loss=.5, accuracy=accuracy)
            emit('result', dataset=dataset, bits=bits, accuracy=accuracy)
    emit('table', title='SIMULATED example — not experimental results', columns=['Dataset', 'Example accuracy'], rows=[['MNIST', '80% (illustrative)'], ['CIFAR10', '80% (illustrative)']])


def demo_snapshot(dataset, bits):
    """Synthetic layer inspection for the explicitly labeled presentation demo."""
    import math
    layers = []
    for index, (name, count, inputs, side) in enumerate((('conv1', 32, 27, 32), ('conv2', 64, 288, 16), ('fc1', 128, 4096, 0), ('fc2', 10, 128, 0))):
        indices = [round(i * (inputs - 1) / 31) for i in range(32)] if name == 'fc1' else list(range(inputs))
        weights = [[round(math.sin(i * .9 + j * .4) * .25, 4) for j in indices] for i in range(count)]
        if side and bits == 1:
            weights = [[.2 if w >= 0 else -.2 for w in row] for row in weights]
        row = dict(name=name, kind='conv' if side else 'dense', shape=[count, side, side] if side else [count],
                   activations=[round(math.sin(i * .43 + index), 4) for i in range(count)],
                   weights=weights, input_indices=indices, input_count=inputs,
                   weight_shape=[count, inputs // 9, 3, 3] if side else [count, inputs],
                   activation_kind='illustrative pre-ReLU output', weight_kind='illustrative weights')
        if side:
            row['maps'] = [[round(math.sin(x / side * 8 + c * .2) * math.cos(y / side * 6), 3)
                            for y in range(side) for x in range(side)] for c in range(count)]
        layers.append(row)
    pixels = []
    for y in range(32):
        for x in range(32):
            value = int(255 * (abs(math.hypot(x - 16, y - 16) - 10) < 2))
            pixels.extend([value] * 3 if dataset == 'MNIST' else [x * 8, y * 8, 150])
    return dict(layers=layers, input={'shape': [3, 32, 32], 'pixels': pixels, 'label': 0, 'prediction': 0},
                sample='SIMULATED illustration — no measured neurons or connections.', sample_id=bits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('full', 'quick', 'demo'), default='full')
    args = parser.parse_args()
    emit('status', status='running', message=f'Starting {args.mode} run')
    stream = LogStream()
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            (run_demo() if args.mode == 'demo' else run_real(args.mode))
        stream.flush()
        emit('status', status='completed', message='Illustrative demonstration finished.' if args.mode == 'demo' else 'All original script sections completed.')
    except Exception as exc:
        stream.flush()
        emit('log', message=traceback.format_exc())
        emit('status', status='error', message=str(exc))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
