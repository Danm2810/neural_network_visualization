"""Focused regression checks for instrumentation and desktop lifecycle."""
import ast
import unittest
from pathlib import Path

from quantization_visualizer.runtime import Instrument, instrument_source, SOURCE


class InstrumentationTests(unittest.TestCase):
    def test_full_source_compiles_in_both_modes(self):
        source = SOURCE.read_text(encoding='utf-8-sig')
        for quick in (False, True):
            compile(instrument_source(source, quick), str(SOURCE), 'exec')

    def test_observation_preserves_network_outputs_and_gradients(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        source = ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
        names = {'QuantizeWeights', 'quantize_weights', 'QuantizableConv2d', 'SimpleCNN'}
        original = ast.Module(body=[n for n in source.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names], type_ignores=[])
        import copy
        observed = ast.fix_missing_locations(Instrument().visit(copy.deepcopy(original)))
        base = dict(torch=torch, nn=nn, F=F)
        instrumented = dict(base, _observe=lambda *_: None)
        exec(compile(original, '<original>', 'exec'), base)
        exec(compile(observed, '<observed>', 'exec'), instrumented)
        torch.set_num_threads(2)
        for bits in (1, 4, 8, 32):
            torch.manual_seed(42)
            model_a = base['SimpleCNN'](k_bits=bits)
            torch.manual_seed(42)
            model_b = instrumented['SimpleCNN'](k_bits=bits)
            inputs = torch.randn(2, 3, 32, 32)
            a, b = model_a(inputs), model_b(inputs)
            self.assertTrue(torch.equal(a, b), f'{bits}-bit forward changed')
            a.sum().backward()
            b.sum().backward()
            for p, q in zip(model_a.parameters(), model_b.parameters()):
                self.assertTrue(torch.equal(p.grad, q.grad), f'{bits}-bit gradient changed')

    def test_gui_events_and_cleanup(self):
        from quantization_visualizer.app import QuantizationApp
        app = QuantizationApp()
        app.withdraw()
        try:
            for event in [
                {'type': 'status', 'status': 'running'},
                {'type': 'stage', 'stage': 'Training', 'message': 'Test stage'},
                {'type': 'activity', 'component': 'quantize', 'bits': 4, 'values': [-.3, .2]},
                {'type': 'batch', 'batch': 2, 'total_batches': 4, 'accuracy': 75., 'loss': .4},
                {'type': 'epoch', 'epoch': 1, 'dataset': 'MNIST', 'bits': 4, 'phase': 'train', 'accuracy': 75., 'loss': .4},
                {'type': 'table', 'columns': ['Name'], 'rows': [['Result']]},
                {'type': 'status', 'status': 'completed'},
            ]:
                app._event(event)
                app.update()
            self.assertEqual(app.metric_vars['Accuracy'].get(), '75.00%')
            self.assertEqual(len(app.tree.get_children()), 1)
            self.assertTrue(app.terminal_status)
        finally:
            app._close()


if __name__ == '__main__':
    unittest.main()
