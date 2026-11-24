"""Regression: global visual pause is independent of inspection and Python stop."""
from pathlib import Path
from playwright.sync_api import sync_playwright

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 940})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.add_init_script("window.stopCalls=0;window.pywebview={api:{start:async()=>({ok:true}),stop:async()=>{window.stopCalls++;return {ok:true}},poll:async()=>({events:[]})}}")
        page.goto((Path(__file__).resolve().parent / "quantization_visualizer/web/index.html").as_uri())
        page.wait_for_function("!!window.lab")
        page.click("#start")
        assert page.locator("#pause").is_enabled()
        page.evaluate("lab.setReduced(true);lab.applyScene({component:'forward',dataset:'MNIST',bits:4,event:{}});lab.setReduced(false)")
        # Expanding a neuron holds the reading timer without changing the button.
        page.evaluate("lab.network._select(lab.network.nodeByKey.get('conv1:0'));lab.state.sceneStarted=0;lab.receive({type:'activity',component:'update',dataset:'MNIST',bits:4});lab.pump()")
        assert page.evaluate("!lab.state.paused && lab.state.networkFocused")
        assert page.evaluate("lab.state.current.component") == "forward"
        page.click("#pause")
        visual_state = "({clock:lab.network.clock,reading:lab.network.readingElapsed,construction:lab.network.construction,camera:{...lab.network.camera},values:lab.network.nodes.map(n=>n.value)})"
        frozen = page.evaluate(visual_state)
        page.wait_for_timeout(250)
        assert page.evaluate(visual_state) == frozen, 'Visual state continued after pause'
        # Incoming measurements and a forced repaint must not alter paused values.
        page.evaluate("lab.network.setSnapshot({dataset:'MNIST',bits:4,layers:[{name:'conv1',activations:Array(32).fill(9)}]});lab.network._draw()")
        assert page.evaluate(visual_state) == frozen, 'Paused repaint consumed new measurements'
        assert page.evaluate("window.stopCalls") == 0
        assert page.locator("#stop").is_enabled()
        # Closing details, Go live, replay and a new run cannot override pause.
        page.click("#reset-view")
        assert page.evaluate("lab.state.paused && !lab.network.selected")
        reset_camera = page.evaluate('({...lab.network.camera})')
        page.wait_for_timeout(150)
        assert page.evaluate('({...lab.network.camera})') == reset_camera, 'Reset view kept easing while paused'
        assert page.evaluate("performance.now()-lab.state.sceneStarted") < 1000
        page.click("#go-live")
        assert page.evaluate("lab.state.paused && lab.network.paused && lab.state.queue.length===0")
        page.evaluate("lab.replay()")
        assert page.evaluate("lab.state.paused")
        page.click("#stop")
        assert page.evaluate("window.stopCalls") == 1
        assert page.evaluate("lab.state.paused")
        page.evaluate("lab.receive({type:'status',status:'stopped'})")
        page.click("#start")
        assert page.evaluate("lab.state.paused && lab.network.paused")
        page.click("#pause")
        assert page.evaluate("!lab.state.paused && !lab.network.paused")
        # Stopping Python while visuals run also leaves the playback choice alone.
        page.click("#stop")
        page.evaluate("lab.receive({type:'status',status:'stopped'})")
        assert page.evaluate("!lab.state.paused && !lab.network.paused")
        assert page.locator("#pause").is_enabled()
        assert not errors, errors
        browser.close()
    print("PASS: global pause survives inspection, Go live, replay, stop and restart; controls stay independent.")

if __name__ == "__main__":
    main()
