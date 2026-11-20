"""Progressive fixed-architecture reveal, interaction, and playback regression."""
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
        page.evaluate("lab.state.paused=true;lab.network.setPaused(true);lab.applyScene({component:'data',dataset:'MNIST',bits:4,event:{}})")
        visible = "lab.network.nodes.filter(n=>lab.network._nodeVisibility(n)>.99).length"
        assert page.evaluate(visible) == 1
        assert page.evaluate("lab.network._hit(lab.network._project(lab.network.nodeByKey.get('fc1:0'))) === null"), 'Unrevealed node is clickable'
        page.locator('#network-canvas').focus()
        page.keyboard.press('Enter')
        assert page.evaluate('lab.network.selected.name') == 'input'
        page.keyboard.press('Escape')

        # Every layer adds real inspectable objects in stable positions.
        previews = ROOT / 'visualizer_previews'
        previews.mkdir(exist_ok=True)
        coordinates = page.evaluate('lab.network.nodes.map(n=>[n.x,n.y])')
        for progress, count in [(0, 1), (1, 33), (2, 97), (3, 225), (4, 235)]:
            page.evaluate('v=>lab.network.setConstructionProgress(v,{immediate:true})', progress)
            page.wait_for_timeout(40)
            assert page.evaluate(visible) == count
            assert page.evaluate('lab.network.nodes.map(n=>[n.x,n.y])') == coordinates
            page.screenshot(path=str(previews / f'construction-{progress}.png'))

        # The real timer reveals successive layers, freezes, and resumes.
        page.evaluate("lab.network.resetConstruction();lab.network.setTheme('MNIST','quantize');lab.network.setSpeed(3);lab.network.setPaused(false)")
        page.wait_for_function('lab.network.construction>.2')
        page.evaluate('lab.network.setPaused(true)')
        frozen = page.evaluate('lab.network.construction')
        page.wait_for_timeout(200)
        assert page.evaluate('lab.network.construction') == frozen
        page.evaluate('lab.network.setPaused(false)')
        page.wait_for_function('lab.network.construction===2', timeout=5000)
        assert page.evaluate(visible) == 97
        page.evaluate("lab.network.setTheme('MNIST','forward')")
        page.wait_for_function('lab.network.construction===4', timeout=5000)
        assert page.evaluate(visible) == 235
        for phase in ['backward', 'update', 'evaluate', 'quantize']:
            page.evaluate('phase=>lab.network.setTheme("MNIST",phase)', phase)
            assert page.evaluate(visible) == 235, 'Later operations demolished the network'

        # Reduced motion reveals complete stages without continuous animation.
        page.evaluate("lab.network.setTheme('CIFAR10','data');lab.setReduced(true);lab.network.setTheme('CIFAR10','quantize')")
        assert page.evaluate(visible) == 97
        page.evaluate("lab.network.setTheme('CIFAR10','forward')")
        assert page.evaluate(visible) == 235
        page.locator('#network-canvas').focus()
        page.keyboard.press('Enter')
        assert page.evaluate('lab.network.selected.name') == 'conv1'
        assert not errors, errors
        browser.close()
    print('PASS: ordered construction, stable topology, hidden-node interaction, pause/resume, completed learning phases, and reduced motion.')


if __name__ == '__main__':
    main()
