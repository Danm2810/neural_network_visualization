"""Independent regression checks for classroom text clearance and transition covers."""
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent


def main():
    failures = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto((ROOT / 'quantization_visualizer/web/index.html').as_uri())
        page.wait_for_function('!!window.lab')
        page.evaluate('lab.state.paused=true;lab.setReduced(true)')
        for width, height in [(1000, 700), (1366, 768), (1440, 940), (1920, 1080)]:
            page.set_viewport_size({'width': width, 'height': height})
            for dataset in ['MNIST', 'CIFAR10']:
                for scene in ['welcome', 'data', 'quantize', 'forward', 'backward', 'update', 'evaluate']:
                    page.evaluate("s=>lab.applyScene({component:s[0],dataset:s[1],bits:4,event:{}})", [scene, dataset])
                    page.wait_for_timeout(30)
                    errors = page.evaluate('''() => {
                        const ids=['scene-title','scene-description','scene-detail'];
                        const rect=id=>document.getElementById(id).getBoundingClientRect();
                        const toolbar=rect('network-toolbar'),canvas=rect('network-canvas');
                        const errors=[];
                        for(const id of ids){const r=rect(id);if(r.bottom>toolbar.top-2)errors.push(`${id} crosses toolbar: ${r.bottom.toFixed(1)} > ${toolbar.top.toFixed(1)}`);}
                        for(let i=0;i<ids.length;i++)for(let j=i+1;j<ids.length;j++){
                            const a=rect(ids[i]),b=rect(ids[j]);
                            if(Math.min(a.right,b.right)>Math.max(a.left,b.left)+2&&Math.min(a.bottom,b.bottom)>Math.max(a.top,b.top)+2)errors.push(`${ids[i]} overlaps ${ids[j]}`);
                        }
                        if(canvas.height<130)errors.push('Canvas too short');
                        if(document.documentElement.scrollWidth>innerWidth)errors.push('Horizontal overflow');
                        return errors;
                    }''')
                    failures.extend(f'{width}x{height} {dataset}/{scene}: {e}' for e in errors)
        # Sample actual GSAP intermediate states. Transition lettering must stay
        # on its opaque panel, never directly over the exposed graph.
        page.set_viewport_size({'width': 1366, 'height': 768})
        page.evaluate("lab.state.paused=false;lab.setReduced(false);lab.transition({component:'data',dataset:'MNIST',bits:4,event:{}},true);void lab.state.transitionTl.pause(0)")
        duration = page.evaluate('lab.state.transitionTl.duration()')
        for step in range(1, 60):
            page.evaluate('t=>{lab.state.transitionTl.time(t,false);}', duration * step / 60)
            violations = page.evaluate('''() => {
                const title=document.querySelector('.transition-title');
                const overlay=document.getElementById('transition');
                if(Number(getComputedStyle(title).opacity)<.05||getComputedStyle(overlay).visibility==='hidden')return [];
                const sheet=document.querySelector('.transition-sheet'),s=sheet.getBoundingClientRect();
                const errors=[];
                for(const id of ['transition-kicker','transition-word']){
                    const r=document.getElementById(id).getBoundingClientRect();
                    if(r.top<s.top-1||r.bottom>s.bottom+1||r.left<s.left-1||r.right>s.right+1)errors.push(id+' escaped transition cover');
                }
                const color=getComputedStyle(sheet).backgroundColor;
                if(color==='transparent'||color==='rgba(0, 0, 0, 0)')errors.push('Transition cover is transparent');
                return errors;
            }''')
            failures.extend(f'Transition frame {step}: {e}' for e in violations)
        text = page.locator('body').inner_text()
        for broken in ['\u00c2\u00b7', '\u00e2\u20ac', '\u00e2\u2020']:
            if broken in text:
                failures.append('UI contains corrupted UTF-8 punctuation')
        browser.close()
    if failures:
        raise AssertionError('\n'.join(failures))
    print('All 56 scene/dataset/window combinations have clear text; 59 intermediate transition frames retain an opaque title cover.')


if __name__ == '__main__':
    main()
