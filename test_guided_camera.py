"""Reading intervals, synchronized camera copy, and whole-presentation pause."""
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 1440, 'height': 940})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        # Drive queue advancement explicitly, independently of machine speed.
        page.add_init_script("window.setInterval=()=>0")
        page.goto((ROOT / 'quantization_visualizer/web/index.html').as_uri())
        page.wait_for_function('!!window.lab?.network')
        page.evaluate("lab.applyScene({component:'data',dataset:'MNIST',bits:4,event:{}})")
        page.wait_for_function('lab.network.camera.zoom>1.6')
        assert page.evaluate('lab.network.cameraFocus') == 0
        page.evaluate('lab.setPaused(true)')
        frozen = page.evaluate('({camera:{...lab.network.camera},clock:lab.network.clock,reading:lab.network.readingElapsed})')
        page.wait_for_timeout(250)
        assert page.evaluate('({camera:{...lab.network.camera},clock:lab.network.clock,reading:lab.network.readingElapsed})') == frozen
        page.evaluate('lab.setPaused(false);cancelAnimationFrame(lab.network.frame)')

        # The same displayed words determine the interval at every visual speed.
        page.evaluate("lab.network.setConstructionProgress(4,{immediate:true});lab.applyScene({component:'forward',dataset:'MNIST',bits:4,event:{}})")
        for speed in range(3):
            page.evaluate('v=>lab.setSpeed(v)', speed)
            assert page.evaluate("lab.getReadingSeconds({stage:'forward',layer:1,building:false})") == 8
            assert 8 <= page.evaluate("lab.getReadingSeconds({stage:'forward',layer:null,building:false})") <= 11
        shots = page.evaluate('lab.network.readingShots')
        assert [s['layer'] for s in shots] == [1, 2, 3, 4, None]
        assert all(8 <= s['duration'] <= 11 for s in shots)
        assert shots[-1]['duration'] > shots[0]['duration'], 'Long overview did not get extra reading time'
        for index, shot in enumerate(shots):
            page.evaluate('lab.network._directCamera()')
            assert page.evaluate('lab.network.cameraFocus') == shot['layer']
            assert page.evaluate('lab.state.cameraShot.layer') == shot['layer']
            copy = page.locator('#scene-description').inner_text()
            assert copy.strip()
            if shot['layer'] is not None:
                assert len(copy.split()) <= 18
            page.evaluate("lab.receive({type:'batch',dataset:'MNIST',epoch:99,batch:999,total_batches:1000,loss:.42})")
            assert page.locator('#scene-description').inner_text() == copy
            page.evaluate('d=>lab.network._advanceReading(d-.01)', shot['duration'])
            assert page.evaluate('lab.network.readingIndex') == index, 'Shot advanced before full reading interval'
            page.evaluate('lab.network._advanceReading(.01);lab.network._directCamera()')
        assert page.evaluate('lab.network.isReadingComplete()')
        assert page.evaluate('lab.network.target.zoom') == 1

        # Queue progress cannot bypass unfinished reading, even with an old timer.
        page.evaluate("lab.applyScene({component:'backward',dataset:'MNIST',bits:4,event:{}});lab.state.queue=[{component:'update',dataset:'MNIST',bits:4,event:{}}];lab.state.sceneStarted=-Infinity;lab.pump()")
        assert page.evaluate('lab.state.current.component') == 'backward'
        page.evaluate('lab.network._advanceReading(lab.network.readingDuration);lab.network._directCamera();lab.state.sceneStarted=-Infinity;lab.pump()')
        assert page.evaluate('lab.state.current.component') == 'update'
        page.wait_for_function('!lab.state.transitioning')

        # Actual GSAP fades and a live CSS progress transition freeze mid-flight.
        page.evaluate("lab.transition({component:'evaluate',dataset:'MNIST',bits:4,event:{}},true);lab.receive({type:'batch',batch:1,total_batches:10})")
        page.wait_for_timeout(60)
        page.evaluate('lab.setPaused(true)')
        frozen = page.evaluate("({opacity:getComputedStyle(document.querySelector('.scene-copy')).opacity,width:getComputedStyle(document.getElementById('progress-fill')).width,global:gsap.globalTimeline.time()})")
        page.evaluate("lab.receive({type:'batch',batch:8,total_batches:10});lab.receive({type:'log',message:'Background work continues'})")
        page.wait_for_timeout(300)
        actual = page.evaluate("({opacity:getComputedStyle(document.querySelector('.scene-copy')).opacity,width:getComputedStyle(document.getElementById('progress-fill')).width,global:gsap.globalTimeline.time()})")
        assert actual == frozen, (frozen, actual)
        assert page.locator('#metric-batch').inner_text() == '1 / 10'
        assert 'Background work continues' not in page.locator('#log').inner_text()
        page.evaluate('lab.setPaused(false)')
        page.wait_for_function('!lab.state.transitioning')
        assert page.locator('#metric-batch').inner_text() == '8 / 10'
        assert 'Background work continues' in page.locator('#log').inner_text()

        # Deliberate manual control keeps its framing and description.
        page.evaluate("lab.applyScene({component:'backward',dataset:'MNIST',bits:4,event:{}});lab.network._zoom(1.2)")
        manual = page.evaluate('({...lab.network.target})')
        manual_copy = page.locator('#scene-description').inner_text()
        page.evaluate('lab.network._advanceReading(9);lab.network._directCamera()')
        assert page.evaluate('({...lab.network.target})') == manual
        assert page.locator('#scene-description').inner_text() == manual_copy
        page.evaluate("lab.network._select(lab.network.nodeByKey.get('conv1:0'))")
        page.keyboard.press('Escape')
        assert page.evaluate('lab.state.cameraShot.layer') is None
        page.evaluate("lab.applyScene({component:'evaluate',dataset:'MNIST',bits:4,event:{}});lab.setReduced(true)")
        assert page.evaluate('lab.network.camera.zoom') == 1
        assert not errors, errors
        browser.close()
    print('PASS: 8–11s reading at every speed, synchronized copy, queue gating, whole-animation pause, metric/log caching, manual control, reduced motion.')


if __name__ == '__main__':
    main()
