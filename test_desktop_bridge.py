"""Regression tests for the new desktop bridge and real neuron telemetry."""
import json
import unittest

from quantization_visualizer.desktop import RuntimeBridge
from quantization_visualizer.runtime import demo_snapshot


class DesktopBridgeTests(unittest.TestCase):
    def test_invalid_command_and_arbitrary_file_cannot_be_opened(self):
        bridge = RuntimeBridge()
        self.assertFalse(bridge.start('not-a-mode')['ok'])
        self.assertFalse(bridge.open_artifact('neural_network_quantization.py')['ok'])

    def test_poll_is_bounded_and_cursor_advances_without_duplicates(self):
        bridge = RuntimeBridge()
        for i in range(1500):
            bridge._append({'type': 'log', 'message': str(i)}, 0)
        first = bridge.poll(0)
        self.assertTrue(first['dropped'])
        self.assertEqual(len(first['events']), 80)
        second = bridge.poll(first['latest'])
        self.assertGreater(second['events'][0]['seq'], first['latest'])
        self.assertLessEqual(len(bridge._events), 1200)

    def test_stale_worker_does_not_enter_new_run(self):
        bridge = RuntimeBridge()
        bridge._run_id = 2
        bridge._append({'type': 'log', 'message': 'stale'}, 1)
        self.assertEqual(bridge.poll()['events'], [])

    def test_demo_has_inspectable_maps_and_honest_sampled_connections(self):
        snap = demo_snapshot('MNIST', 1)
        self.assertIn('SIMULATED', snap['sample'])
        self.assertEqual(len(snap['layers'][0]['maps']), 32)
        self.assertEqual(len(snap['layers'][0]['maps'][0]), 1024)
        self.assertEqual(len(snap['layers'][2]['activations']), 128)
        self.assertEqual(snap['layers'][2]['input_count'], 4096)
        self.assertEqual(len(snap['layers'][2]['input_indices']), 32)
        self.assertEqual(len(snap['layers'][3]['weights'][0]), 128)
        json.dumps(snap, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
