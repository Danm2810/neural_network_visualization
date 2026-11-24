"""Regression checks for readable text and uninterrupted network transitions."""
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
        page.evaluate('lab.setPaused(true);lab.setReduced(true)')
        for width, height in [(900, 700), (1000, 700), (1366, 768), (1440, 940), (1920, 1080)]:
            page.set_viewport_size({'width': width, 'height': height})
            scene_geometry = None
            scene_palette = None
            for dataset in ['MNIST', 'CIFAR10']:
                for scene in ['welcome', 'data', 'quantize', 'forward', 'backward', 'update', 'evaluate']:
                    page.evaluate("s=>lab.applyScene({component:s[0],dataset:s[1],bits:4,event:{}})", [scene, dataset])
                    page.wait_for_timeout(30)
                    geometry = page.locator('#network-canvas').bounding_box()
                    palette = page.evaluate("['--paper','--ink','--accent'].map(p=>getComputedStyle(document.body).getPropertyValue(p))")
                    if scene_geometry is not None and any(abs(geometry[key] - scene_geometry[key]) > 1 for key in geometry):
                        failures.append(f'{width}x{height} {dataset}/{scene}: scene change moved the network')
                    if scene_palette is not None and palette != scene_palette:
                        failures.append(f'{dataset}/{scene}: scene change altered palette')
                    scene_geometry, scene_palette = geometry, palette
                    errors = page.evaluate('''() => {
                        const ids=['scene-title','scene-description','scene-detail'];
                        const rect=id=>document.getElementById(id).getBoundingClientRect();
                        const canvas=rect('network-canvas'),panel=rect('story-panel');
                        const errors=[];
                        const intro=document.querySelector('.scene-intro');
                        if(!['auto','scroll','hidden'].includes(getComputedStyle(intro).overflowY))errors.push('Long explanation can paint over controls');
                        for(const id of ids){
                            const el=document.getElementById(id),r=rect(id);if(!r.width||!r.height)continue;
                            if(r.left<panel.left-1||r.right>panel.right+1)errors.push(id+' overflows story panel');
                            if(el.scrollWidth>el.clientWidth+1||el.scrollHeight>el.clientHeight+1)errors.push(id+' has clipped text');
                        }
                        for(let i=0;i<ids.length;i++)for(let j=i+1;j<ids.length;j++){
                            const a=rect(ids[i]),b=rect(ids[j]);
                            if(Math.min(a.right,b.right)>Math.max(a.left,b.left)+2&&Math.min(a.bottom,b.bottom)>Math.max(a.top,b.top)+2)errors.push(`${ids[i]} overlaps ${ids[j]}`);
                        }
                        if(canvas.height<innerHeight*.45)errors.push('Canvas too short for network-first layout');
                        if(canvas.width*canvas.height<=panel.width*panel.height)errors.push('Story panel dominates network');
                        const controls=['start','stop','pause','go-live','replay','motion','messages','speed-toggle','run-mode'];
                        for(const id of controls){
                            const el=document.getElementById(id),r=rect(id);
                            if(!document.getElementById('story-panel').contains(el))errors.push(id+' outside unified story panel');
                            if(r.left<panel.left-1||r.right>panel.right+1||r.top<panel.top-1||r.bottom>panel.bottom+1)errors.push(id+' clipped by panel');
                            const minFont=['start','stop','run-mode'].includes(id)?16:14;
                            if(parseFloat(getComputedStyle(el).fontSize)<minFont)errors.push(id+' font too small');
                        }
                        for(let i=0;i<controls.length;i++)for(let j=i+1;j<controls.length;j++){
                            const a=rect(controls[i]),b=rect(controls[j]);
                            if(Math.min(a.right,b.right)>Math.max(a.left,b.left)+2&&Math.min(a.bottom,b.bottom)>Math.max(a.top,b.top)+2)errors.push(controls[i]+' overlaps '+controls[j]);
                        }
                        if(parseFloat(getComputedStyle(document.getElementById('scene-description')).fontSize)<22)errors.push('Description font below requested increase');
                        if(parseFloat(getComputedStyle(document.getElementById('scene-detail')).fontSize)<16)errors.push('Detail font below requested increase');
                        if(document.documentElement.scrollWidth>innerWidth)errors.push('Horizontal overflow');
                        return errors;
                    }''')
                    failures.extend(f'{width}x{height} {dataset}/{scene}: {e}' for e in errors)
        # Sample actual GSAP intermediate states: only copy may crossfade.
        page.set_viewport_size({'width': 1366, 'height': 768})
        page.evaluate("lab.setPaused(false);lab.setReduced(false);lab.transition({component:'data',dataset:'MNIST',bits:4,event:{}},true);void lab.state.transitionTl.pause(0)")
        duration = page.evaluate('lab.state.transitionTl.duration()')
        assert duration <= .5, 'Description transition is distracting/too long'
        baseline = page.locator('#network-canvas').bounding_box()
        for step in range(1, 60):
            page.evaluate('t=>{lab.state.transitionTl.time(t,false);}', duration * step / 60)
            violations = page.evaluate('''() => {
                const overlay=document.getElementById('transition');
                const errors=[];
                if(overlay&&getComputedStyle(overlay).visibility!=='hidden'&&getComputedStyle(overlay).display!=='none')errors.push('Full-screen cover is visible');
                for(const id of ['scene','network-stage','network-canvas']){
                    const s=getComputedStyle(document.getElementById(id));
                    if(Number(s.opacity)!==1||s.visibility==='hidden'||s.display==='none')errors.push(id+' disappears during text transition');
                    if(s.transform!=='none')errors.push(id+' moves during text transition');
                }
                return errors;
            }''')
            failures.extend(f'Transition frame {step}: {e}' for e in violations)
            current = page.locator('#network-canvas').bounding_box()
            if any(abs(current[key] - baseline[key]) > 1 for key in baseline):
                failures.append(f'Transition frame {step}: network geometry changed')
        text = page.locator('body').inner_text()
        for broken in ['\u00c2\u00b7', '\u00e2\u20ac', '\u00e2\u2020']:
            if broken in text:
                failures.append('UI contains corrupted UTF-8 punctuation')
        # Results badges reserve their own row rather than covering precision.
        page.evaluate("lab.setReduced(true);lab.state.completed=true;for(const dataset of ['MNIST','CIFAR10'])for(const bits of [1,4,8,32])lab.receive({type:'result',dataset,bits,accuracy:bits});lab.applyScene({component:'results',dataset:'MNIST',bits:32,event:{}})")
        for width,height in [(900,700),(1366,768),(1920,1080)]:
            page.set_viewport_size({'width':width,'height':height})
            violations=page.evaluate('''() => [...document.querySelectorAll('.score-card.best')].flatMap(card=>{
                const tag=card.querySelector('.best-tag').getBoundingClientRect(),precision=card.querySelector('.precision').getBoundingClientRect(),score=card.querySelector('.score').getBoundingClientRect();
                return tag.top<precision.bottom||tag.bottom>score.top?['Results badge overlaps score or precision']:[];
            })''')
            failures.extend(violations)
        browser.close()
    if failures:
        raise AssertionError('\n'.join(failures))
    print('All 70 scene/dataset/window combinations preserve readable unified controls and a dominant network; 59 crossfade frames stay stationary.')


if __name__ == '__main__':
    main()
