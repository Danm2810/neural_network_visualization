"""Browser-level presentation checks, optionally against the actual Python worker.

Run with the optional Playwright dev dependency and installed Chromium:
  .venv/Scripts/python.exe test_web_presentation.py [--live]
"""
import argparse
import json
from pathlib import Path
import time
from playwright.sync_api import sync_playwright
from quantization_visualizer.desktop import RuntimeBridge

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    records = [json.loads(line) for line in (ROOT / 'web_quick_validation.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    snapshots = [e for e in records if e['type'] == 'snapshot']
    screenshots = ROOT / 'visualizer_previews'
    screenshots.mkdir(exist_ok=True)
    errors = []
    bridge = RuntimeBridge()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 1440, 'height': 940}, device_scale_factor=1)
        page.on('pageerror', lambda error: errors.append(str(error)))
        if args.live:
            page.expose_function('_runtimeStart', bridge.start)
            page.expose_function('_runtimeStop', bridge.stop)
            page.expose_function('_runtimePoll', bridge.poll)
            page.expose_function('_runtimeOpen', lambda path: {'ok': True})
            page.add_init_script("window.pywebview={api:{start:window._runtimeStart,stop:window._runtimeStop,poll:window._runtimePoll,open_artifact:window._runtimeOpen}};")
        page.goto((ROOT / 'quantization_visualizer/web/index.html').as_uri())
        page.wait_for_function('!!window.lab && !!window.gsap && !!window.Lenis')
        assert not errors, errors
        page.screenshot(path=str(screenshots / '01-welcome.png'))
        if args.live:
            page.evaluate("window.dispatchEvent(new Event('pywebviewready'))")
            page.click('#start')
            page.wait_for_function('lab.state.snapshots.size > 0', timeout=90000)
            page.click('#pause')
            page.wait_for_timeout(300)
            scene = page.evaluate('lab.state.current.component')
            sequence = page.evaluate('lab.state.seq')
            page.wait_for_timeout(1000)
            assert page.evaluate('lab.state.current.component') == scene, 'Scene changed while paused'
            assert page.evaluate('lab.state.paused'), 'Pause failed'
            page.wait_for_function('lab.state.terminal === "completed"', timeout=120000)
            assert page.evaluate('lab.state.results.size') == 8
            assert page.evaluate('lab.state.seq') > sequence, 'Live worker stopped while presentation paused'
            assert page.locator('#results').is_hidden(), 'Results interrupted paused presentation'
            page.click('#pause')
            print('Actual Quick worker completed all eight models while visuals paused; live telemetry continued.')
        # Deterministic visual checks use real recorded snapshots.
        page.evaluate("lab.state.queue=[];lab.state.replayQueue=[];lab.state.replaying=false;lab.state.transitionTl?.kill();lab.state.transitioning=false;lab.setPaused(false);lab.setReduced(false)")
        sample = next(e for e in snapshots if e['dataset'] == 'MNIST' and e['bits'] == 4)
        page.evaluate("s=>{lab.receive(s);lab.applyScene({component:'forward',dataset:s.dataset,bits:s.bits,event:{component:'forward'},snapshot:s});}", sample)
        page.wait_for_timeout(700)
        page.screenshot(path=str(screenshots / '02-mnist-network.png'))
        # Every dense neuron is present; full convolution spatial data is inspectable.
        assert page.evaluate("lab.network.nodes.filter(n=>n.name==='fc1').length") == 128
        page.locator('#network-canvas').focus()
        page.keyboard.press('Enter')
        page.wait_for_timeout(500)
        assert page.evaluate('lab.network.selected.name') == 'conv1'
        assert page.evaluate("lab.network.layers.get('conv1').maps[0].length") == 1024
        page.screenshot(path=str(screenshots / '03-neuron-inspection.png'))
        page.keyboard.press('Escape')
        sample = next(e for e in snapshots if e['dataset'] == 'CIFAR10' and e['bits'] == 4)
        page.evaluate("s=>{lab.receive(s);lab.transition({component:'data',dataset:s.dataset,bits:s.bits,event:{component:'data'},snapshot:s},true);}", sample)
        page.wait_for_function('!lab.state.transitioning', timeout=10000)
        page.screenshot(path=str(screenshots / '04-cifar-world.png'))
        page.evaluate("lab.state.queue=[];lab.state.completed=false;lab.receive({type:'activity',component:'results'});")
        assert page.locator('#results').is_hidden(), 'Intermediate plotting incorrectly revealed final results'
        for event in records:
            if event['type'] == 'result':
                page.evaluate('e=>lab.receive(e)', event)
        page.evaluate("lab.state.completed=true;lab.transition({component:'results',dataset:'CIFAR10',bits:32,event:{}},true)")
        page.wait_for_function('!lab.state.transitioning', timeout=10000)
        page.wait_for_timeout(2200)
        assert page.locator('.score-card').count() == 8
        assert page.locator('#results').is_visible()
        page.screenshot(path=str(screenshots / '05-results.png'))
        page.locator('#results').hover()
        page.mouse.wheel(0, 700)
        page.wait_for_timeout(800)
        assert page.evaluate('document.getElementById("results").scrollTop') > 0, 'Results cannot scroll'
        page.screenshot(path=str(screenshots / '06-results-detail.png'))
        page.set_viewport_size({'width': 1000, 'height': 700})
        page.wait_for_timeout(300)
        page.screenshot(path=str(screenshots / '07-small-window.png'))
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Horizontal overflow'
        page.evaluate('lab.setReduced(true)')
        assert page.evaluate('lab.network.reduced')
        # Returning from replay must restore the current final scene, not get
        # stranded on an earlier snapshot when the worker has already finished.
        page.evaluate("lab.replay(); for(let i=0;i<24;i++){lab.network._advanceReading(lab.network.readingDuration);lab.network._directCamera();lab.state.sceneStarted=-Infinity;lab.pump();}")
        assert page.evaluate('lab.state.current.component') == 'results', 'Replay failed to return to results'
        # Completion preserves educational topics instead of erasing them.
        page.evaluate("lab.state.completed=false;lab.state.seenScenes.clear();lab.state.sceneArchive.clear();lab.state.queue=[];")
        for dataset in ('MNIST', 'CIFAR10'):
            for component in ('data', 'quantize', 'forward', 'backward', 'update', 'evaluate'):
                page.evaluate('e=>lab.receive(e)', {'type': 'activity', 'component': component, 'dataset': dataset, 'bits': 4})
        page.evaluate("lab.receive({type:'status',status:'completed'});for(let i=0;i<20;i++){lab.network._advanceReading(lab.network.readingDuration);lab.network._directCamera();lab.state.sceneStarted=-Infinity;lab.pump();}")
        assert page.evaluate('lab.state.seenScenes.size') >= 13, page.evaluate('({seen:[...lab.state.seenScenes],current:lab.state.current,paused:lab.state.paused,focused:lab.state.networkFocused,queued:lab.state.queue.map(x=>x.component)})')
        assert not errors, errors
        print('Browser checks passed: real neurons/maps, dataset transition, pause, results, scrolling, reduced motion, no JavaScript errors.')
        browser.close()
    bridge._close()


if __name__ == '__main__':
    main()
