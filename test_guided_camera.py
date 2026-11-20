"""Guided camera follows significant phases and yields to the reader."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 1440, 'height': 940})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto((ROOT / 'quantization_visualizer/web/index.html').as_uri())
        page.wait_for_function('!!window.lab?.network')
        page.evaluate("lab.applyScene({component:'data',dataset:'MNIST',bits:4,event:{}})")
        page.wait_for_function('lab.network.camera.zoom>1.6')
        assert page.evaluate('lab.network.cameraFocus') == 0
        page.evaluate('lab.network.setPaused(true)')
        frozen = page.evaluate('({...lab.network.camera})')
        page.wait_for_timeout(200)
        assert page.evaluate('({...lab.network.camera})') == frozen
        page.evaluate("lab.network.setPaused(false);lab.applyScene({component:'forward',dataset:'MNIST',bits:4,event:{}});lab.network.setConstructionProgress(4,{immediate:true})")
        for elapsed, expected in [(0, 1), (2.1, 2), (4.1, 3), (6.1, 4), (8.1, None)]:
            page.evaluate('t=>{lab.network.stageClock=lab.network.clock-t;lab.network._directCamera()}', elapsed)
            assert page.evaluate('lab.network.cameraFocus') == expected
        assert page.evaluate('lab.network.target.zoom') == 1
        page.evaluate("lab.applyScene({component:'backward',dataset:'MNIST',bits:4,event:{}});lab.network._directCamera()")
        assert page.evaluate('lab.network.cameraFocus') == 4
        page.evaluate('lab.network.stageClock=lab.network.clock-4.1;lab.network._directCamera()')
        assert page.evaluate('lab.network.cameraFocus') == 2

        # User zoom remains in charge even when the next timed shot is due.
        page.evaluate('lab.network._zoom(1.2)')
        manual = page.evaluate('({...lab.network.target})')
        page.evaluate('lab.network.stageClock=lab.network.clock-7;lab.network._directCamera()')
        assert page.evaluate('({...lab.network.target})') == manual
        page.evaluate("lab.network._select(lab.network.nodeByKey.get('conv1:0'))")
        selected = page.evaluate('({...lab.network.target})')
        page.wait_for_timeout(200)
        assert page.evaluate('({...lab.network.target})') == selected
        page.keyboard.press('Escape')
        page.evaluate("lab.applyScene({component:'evaluate',dataset:'MNIST',bits:4,event:{}});lab.setReduced(true)")
        assert page.evaluate('lab.network.camera.zoom') == 1
        page.wait_for_timeout(200)
        assert page.evaluate('lab.network.camera.zoom') == 1

        # Save an actual sampled close-up for visual review when fixture exists.
        fixture = ROOT / 'web_quick_validation.jsonl'
        if fixture.exists():
            with fixture.open(encoding='utf-8-sig') as stream:
                sample = next(e for line in stream if (e := json.loads(line)).get('type') == 'snapshot')
            page.evaluate("s=>{lab.setReduced(false);lab.applyScene({component:'backward',dataset:s.dataset,bits:s.bits,event:{},snapshot:s});lab.network.stageClock=lab.network.clock-2.1}", sample)
            page.wait_for_timeout(1200)
            page.screenshot(path=str(ROOT / 'visualizer_previews/guided-camera.png'))
        assert not errors, errors
        browser.close()
    print('PASS: guided zoom/pan, phase sequence, pullback, pause, manual control, inspection, reduced motion.')


if __name__ == '__main__':
    main()
