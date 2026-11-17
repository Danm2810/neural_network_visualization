"""Load the real WebView2 desktop renderer and exercise its Python bridge."""
from pathlib import Path
import json
import threading
import time
import webview
from quantization_visualizer.desktop import RuntimeBridge

root = Path(__file__).resolve().parent
bridge = RuntimeBridge()
window = webview.create_window('Quantization Lab — renderer verification',
    str(root / 'quantization_visualizer/web/index.html'), js_api=bridge,
    width=1200, height=800)
report = {'ok': False}


def verify():
    try:
        deadline = time.monotonic() + 35
        while time.monotonic() < deadline:
            ready = window.evaluate_js('Boolean(window.lab && window.pywebview?.api?.poll && window.gsap && window.Lenis)')
            if ready:
                break
            time.sleep(.25)
        else:
            raise RuntimeError('Native bridge did not initialize')
        window.evaluate_js('void window.pywebview.api.poll(0).then(r=>window.__bridgeTest=r)')
        deadline = time.monotonic() + 10
        while not window.evaluate_js('Boolean(window.__bridgeTest)'):
            if time.monotonic() > deadline:
                raise RuntimeError('Native bridge polling timed out')
            time.sleep(.1)
        result = window.evaluate_js('window.__bridgeTest')
        assert 'events' in result, result
        report.update(ok=True, gsap=window.evaluate_js('gsap.version'),
                      nodes=window.evaluate_js('lab.network.nodes.length'), renderer='Edge WebView2')
    except Exception as exc:
        report['error'] = str(exc)
    finally:
        window.destroy()


try:
    webview.start(verify, gui='edgechromium')
finally:
    bridge._close()
    (root / 'native_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
if not report['ok']:
    raise SystemExit(1)
