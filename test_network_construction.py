"""Progressive reveal respects per-layer reading time and exact visual pause."""
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 940})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto((ROOT / "quantization_visualizer/web/index.html").as_uri())
        page.wait_for_function("!!window.lab?.network")
        page.evaluate("lab.setPaused(true);lab.applyScene({component:'data',dataset:'MNIST',bits:4,event:{}})")
        visible = "lab.network.nodes.filter(n=>lab.network._nodeVisibility(n)>.99).length"
        assert page.evaluate(visible) == 1
        assert page.evaluate("lab.network._hit(lab.network._project(lab.network.nodeByKey.get('fc1:0')))===null")
        page.locator('#network-canvas').focus()
        page.keyboard.press('Enter')
        assert page.evaluate('lab.network.selected.name') == 'input'
        page.keyboard.press('Escape')
        coordinates = page.evaluate('lab.network.nodes.map(n=>[n.x,n.y])')
        for progress, count in [(0, 1), (1, 33), (2, 97), (3, 225), (4, 235)]:
            page.evaluate('v=>lab.network.setConstructionProgress(v,{immediate:true})', progress)
            assert page.evaluate(visible) == count
            assert page.evaluate('lab.network.nodes.map(n=>[n.x,n.y])') == coordinates

        # Real animation stops immediately, including an in-flight camera ease.
        page.evaluate("lab.setPaused(false);lab.network.resetConstruction();lab.applyScene({component:'quantize',dataset:'MNIST',bits:4,event:{}});lab.setSpeed(2)")
        page.wait_for_function('lab.network.construction>.15')
        page.evaluate('lab.setPaused(true)')
        state = '({clock:lab.network.clock,construction:lab.network.construction,reading:lab.network.readingElapsed,camera:{...lab.network.camera},values:lab.network.nodes.map(n=>n.value)})'
        frozen = page.evaluate(state)
        page.wait_for_timeout(300)
        assert page.evaluate(state) == frozen
        page.evaluate('lab.network._advanceReading(100)')
        assert page.evaluate(state) == frozen

        # Deterministic elapsed time proves each boundary, independently of speed.
        page.evaluate("lab.setPaused(false);cancelAnimationFrame(lab.network.frame);lab.network.resetConstruction();lab.network.setConstructionProgress(2);lab.network._startReading();lab.network.setSpeed(3)")
        durations = page.evaluate('lab.network.readingShots.map(s=>s.duration)')
        assert all(8 <= seconds <= 11 for seconds in durations)
        page.evaluate('lab.network._advanceReading(7.99)')
        assert page.evaluate('lab.network.readingIndex') == 0
        assert page.evaluate(visible) == 33, 'A second layer appeared before minimum reading time'
        page.evaluate('d=>lab.network._advanceReading(d-7.99+.01)', durations[0])
        assert page.evaluate('lab.network.readingIndex') == 1
        assert page.evaluate('lab.network.construction') > 1
        page.evaluate('lab.network._advanceReading(lab.network.readingDuration)')
        assert page.evaluate('lab.network.isReadingComplete()')
        assert page.evaluate(visible) == 97
        page.evaluate("lab.network.setTheme('MNIST','forward');lab.network._advanceReading(lab.network.readingDuration)")
        assert page.evaluate(visible) == 235
        assert page.evaluate('lab.network.isReadingComplete()')
        for phase in ['backward', 'update', 'evaluate', 'quantize']:
            page.evaluate('phase=>lab.network.setTheme("MNIST",phase)', phase)
            assert page.evaluate(visible) == 235

        # Reduced motion reveals one whole layer, then holds its reading interval.
        page.evaluate("lab.network.setTheme('CIFAR10','data');lab.setReduced(true);lab.network.setTheme('CIFAR10','quantize');lab.network._advanceReading(0)")
        assert page.evaluate(visible) == 33
        page.evaluate('lab.network._advanceReading(7.99)')
        assert page.evaluate(visible) == 33
        page.evaluate('lab.network._advanceReading(lab.network.readingShots[0].duration-7.99+.01)')
        assert page.evaluate(visible) == 97
        assert not page.evaluate('lab.network.isReadingComplete()')
        page.evaluate('lab.network._advanceReading(lab.network.readingDuration)')
        assert page.evaluate('lab.network.isReadingComplete()')
        assert not errors, errors
        browser.close()
    print('PASS: 8–11 second layer boundaries, speed independence, exact visual pause, large elapsed steps, and reduced-motion pacing.')


if __name__ == '__main__':
    main()
