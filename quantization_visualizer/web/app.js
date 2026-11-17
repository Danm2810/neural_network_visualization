/* Desktop presentation: live measurements and an independently paced visual story. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  // Reading time begins after the entrance animation, independent of training speed.
  const MIN_SLIDE_READING_MS = 12000;
  const sceneDefs = {
    welcome: {number:'00',chapter:'MEET THE MACHINE',eyebrow:'A LEARNING MACHINE, REVEALED',title:['Small weights.','Big ideas.'],description:'Watch a neural network learn, then see what happens when its weights have fewer numbers to choose from.',detail:'Start a real run. Zoom into the network. Follow the ideas.',word:'Let’s learn.',color:'#c3f47b'},
    data:{number:'01',chapter:'MEET THE DATA',eyebrow:'EVERY LEARNING STORY STARTS HERE',title:['First,','look closely.'],description:'Images become numbers. The network reads a small group of images at a time, called a batch.',detail:'MNIST contains handwritten digits from 0 to 9. The same network will later tackle color photographs.',word:'Meet the data.',color:'#d1dfef'},
    quantize:{number:'02',chapter:'CHOOSE THE WEIGHT LEVELS',eyebrow:'FEWER SETTINGS. A DIFFERENT CHALLENGE.',title:['Less precision.','Still learning.'],description:'Weights are adjustable knobs. Quantization gives the convolution weights fewer possible settings.',detail:'The network keeps full-precision master weights for learning. These temporary versions are used to make predictions.',word:'Simplify.',color:'#dfaff7'},
    forward:{number:'03',chapter:'MAKE A PREDICTION',eyebrow:'FROM PIXELS TO A BEST GUESS',title:['A signal','finds its way.'],description:'Layers find patterns, combine them, and make a guess. Follow the signal from the image to the ten possible answers.',detail:'Convolution finds local patterns. ReLU keeps positive responses. Pooling shrinks maps. Dense layers combine the evidence.',word:'Make a guess.',color:'#91dcf5'},
    backward:{number:'04',chapter:'LEARN FROM THE MISTAKE',eyebrow:'THE ANSWER SENDS A SIGNAL BACK',title:['Wrong guesses','teach us.'],description:'The loss measures how far the guess was from the right answer. Gradients tell the network which adjustments could help.',detail:'The learning signal travels backward. A straight-through estimator lets it pass through the quantization step.',word:'Learn why.',color:'#f7a3ae'},
    update:{number:'05',chapter:'ADJUST & TRY AGAIN',eyebrow:'ONE SMALL CHANGE AT A TIME',title:['Turn the knobs.','Try again.'],description:'Adam, the optimizer, uses the gradients to adjust the stored weights. The next batch gets a slightly different network.',detail:'All trainable layers are updated. Only the convolution weights are quantized for the next forward pass.',word:'Adjust.',color:'#f2d274'},
    evaluate:{number:'06',chapter:'TAKE THE TEST',eyebrow:'NEW IMAGES. NO ANSWER SHEET.',title:['Now show','what you know.'],description:'The network sees held-out images. It makes predictions without changing its weights, so we can measure how well it learned.',detail:'Accuracy is the share of correct guesses. 80% means about 80 correct answers out of every 100 images.',word:'Take the test.',color:'#b5dfca'},
    results:{number:'07',chapter:'SEE THE BIG PICTURE',eyebrow:'EIGHT MODELS. ONE QUESTION.',title:['What did','we learn?'],description:'Compare how well each model recognized new images. Does using fewer weight levels still give good answers?',detail:'The simplest answer comes from comparing the test scores side by side.',word:'The big picture.',color:'#d0f18c'}
  };
  const state={seq:0,running:false,mode:'quick',paused:false,replaying:false,queue:[],replayQueue:[],history:[],snapshots:new Map(),results:new Map(),artifacts:[],logs:[],tables:[],seenDatasets:new Set(),current:null,sceneStarted:0,transitioning:false,transitionTl:null,revealTl:null,pauseAt:0,reduced:matchMedia('(prefers-reduced-motion: reduce)').matches,terminal:null,completed:false,networkFocused:false};
  let network=null, polling=false, lenis=null, toastTimer=null;
  state.sceneArchive=new Map();state.seenScenes=new Set();state.replayReturn=null;
  const normalizeDataset = value => String(value||'MNIST').toUpperCase().replace(/[^A-Z0-9]/g,'')==='CIFAR10'?'CIFAR10':'MNIST';
  const keyFor=e=>`${normalizeDataset(e.dataset)}:${e.bits??32}`;
  const api=()=>window.pywebview?.api;
  const finite = v => typeof v==='number' && Number.isFinite(v);

  function toast(message){$('toast').textContent=message;clearTimeout(toastTimer);if(window.gsap){gsap.to('#toast',{opacity:1,y:0,duration:.25});toastTimer=setTimeout(()=>gsap.to('#toast',{opacity:0,y:20,duration:.35}),4000);}else{$('toast').style.opacity='1';toastTimer=setTimeout(()=>$('toast').style.opacity='0',4000);}}
  function log(message){state.logs.push(String(message));if(state.logs.length>350)state.logs.splice(0,state.logs.length-350);$('log').textContent=state.logs.join('\n');$('message-count').textContent=String(state.logs.length);if(!$('message-panel').hidden)$('log').scrollTop=$('log').scrollHeight;}
  function titleLines(lines){const h=$('scene-title');h.replaceChildren(document.createTextNode(lines[0]),document.createElement('br'));const em=document.createElement('em');em.textContent=lines[1];h.append(em);}
  function setModeNote(){const notes={quick:'QUICK RUN · Real training, smaller dataset. These scores are a demonstration, not a benchmark.',full:'FULL EXPERIMENT · Original datasets and epoch counts. Live metrics continue while the visual story is paused.',demo:'ILLUSTRATIVE DEMO · Generated events and example scores only. No network is being trained.'};$('mode-banner').textContent=notes[state.mode];}
  function setNetworkFocus(focused){state.networkFocused=focused;document.body.classList.toggle('network-focused',focused);$('reset-view').textContent=focused?'Back to overview ↙':'Reset view ↙';}
  function ensureNetwork(){if(!network && window.NetworkView){network=new NetworkView($('network-canvas'),{onSelect:selection=>{if(selection)setNetworkFocus(true);else setNetworkFocus(false);}});network.setReducedMotion?.(state.reduced);}return network;}
  function enqueue(component,event={}){
    if(!sceneDefs[component])return;
    const scene={component,dataset:normalizeDataset(event.dataset||state.current?.dataset),bits:event.bits??state.current?.bits??32,epoch:event.epoch??1,phase:event.phase||component,event:{...event},time:Date.now(),snapshot:null};
    scene.snapshot=state.snapshots.get(keyFor(scene))||null;
    if(component!=='results')state.sceneArchive.set(`${scene.dataset}:${component}`,scene);
    if(component==='results'){
      if(!state.completed)return;
      const missing=[];
      for(const dataset of ['MNIST','CIFAR10'])for(const phase of ['data','quantize','forward','backward','update','evaluate']){
        const key=`${dataset}:${phase}`,observed=state.sceneArchive.get(key);
        if(observed&&!state.seenScenes.has(key))missing.push(observed);
      }
      state.queue=[...missing,scene];return;
    }
    // Keep one recent occurrence per operation/model. Dataset handoffs are retained.
    const duplicate=state.queue.findIndex(x=>x.component===component&&x.dataset===scene.dataset&&x.bits===scene.bits);
    if(duplicate>=0){state.queue[duplicate]=scene;return;}
    if(state.current && state.current.component===component&&state.current.dataset===scene.dataset&&state.current.bits===scene.bits&&!state.transitioning){state.current.event={...event};return;}
    state.queue.push(scene);
    if(state.queue.length>16){const firstNextDataset=state.queue.findIndex(x=>x.dataset!==state.current?.dataset);const keep=state.queue.filter((x,i)=>i===firstNextDataset||i>=state.queue.length-12);state.queue=keep;}
  }
  function applyScene(scene){
    const d=sceneDefs[scene.component];state.current=scene;state.sceneStarted=performance.now();
    $('results').hidden=scene.component!=='results';$('scene').hidden=scene.component==='results';
    document.body.dataset.scene=scene.component;document.body.dataset.dataset=scene.dataset==='CIFAR10'?'cifar':'mnist';
    $('chapter-number').textContent=d.number;$('chapter-label').textContent=d.chapter;$('eyebrow').textContent=d.eyebrow;titleLines(d.title);
    $('scene-description').textContent=d.description;$('scene-detail').textContent=d.detail;
    $('dataset-name').textContent=scene.dataset==='CIFAR10'?'CIFAR–10':'MNIST';$('dataset-kind').textContent=scene.dataset==='CIFAR10'?'COLOR PHOTOGRAPHS':'HANDWRITTEN DIGITS';$('dataset-icon').textContent=scene.dataset==='CIFAR10'?'◈':'✳';
    if(scene.component==='data'&&scene.dataset==='CIFAR10'){$('scene-description').textContent='A whole new challenge: color photographs. The network now learns to tell apart planes, cars, birds, cats, and six more kinds of objects.';$('scene-detail').textContent='CIFAR-10 has more varied shapes, colors, and backgrounds than handwritten digits. These decorative cards signal the change; inspect the input for the real image.';titleLines(['A new world.','In full color.']);}
    if(scene.component==='quantize'&&scene.bits===32){$('scene-description').textContent='This is our full-precision comparison model. Its convolution weights skip quantization and keep their original values.';$('scene-detail').textContent='Comparing other bit widths against this model helps us see what changes when the weights have fewer settings.';titleLines(['Every setting.','The baseline.']);}
    $('scene-position').textContent=`${String(Number(d.number)+1).padStart(2,'0')} / 08`;
    $('playback-label').textContent=state.replaying?'REPLAY / RECENT OBSERVED ACTIVITY':state.mode==='demo'?'ILLUSTRATIVE EVENTS / NOT REAL TRAINING':'RECENT OBSERVED ACTIVITY / METRICS STAY LIVE';
    const nv=ensureNetwork();nv?.setTheme(scene.dataset,scene.component);nv?.setActivity({...scene.event,component:scene.component});
    const snapshot=scene.snapshot||state.snapshots.get(keyFor(scene));if(snapshot)nv?.setSnapshot(snapshot);
    setNetworkFocus(false);nv?.resetView();
    if(scene.component==='results')renderResults();
    if(!state.replaying){state.seenDatasets.add(scene.dataset);state.seenScenes.add(`${scene.dataset}:${scene.component}`);state.history.push({...scene});if(state.history.length>16)state.history.shift();$('replay').disabled=state.history.length<2;}
  }
  function clearTransition() {
    state.transitionTl?.kill();
    state.transitionTl=null;
    state.transitioning=false;
    if(window.gsap){
      gsap.set('#transition',{visibility:'hidden'});
      gsap.set('.transition-sheet',{x:0,y:0,xPercent:0,yPercent:102});
      gsap.set('.transition-title',{opacity:0,x:0,y:0});
      gsap.set('.scene-copy > *',{clearProps:'transform,opacity'});
    }
  }
  function transition(scene,force=false){
    if(state.transitioning&&!force)return;
    clearTransition();
    state.transitioning=true;
    const datasetChanged=state.current&&state.current.dataset!==scene.dataset;
    const d=sceneDefs[scene.component];
    $('transition-kicker').textContent=scene.component==='results'?'THE EXPERIMENT IS COMPLETE':datasetChanged?'A DIFFERENT DATASET · A NEW CHALLENGE':`CHAPTER ${d.number} / ${d.chapter}`;
    $('transition-word').textContent=datasetChanged?(scene.dataset==='CIFAR10'?'Hello, color.':'Back to digits.'):d.word;
    const color=datasetChanged?(scene.dataset==='CIFAR10'?'#ffb58a':'#d8e7ee'):d.color;
    if(state.reduced||!window.gsap){applyScene(scene);state.transitioning=false;return;}
    const duration=scene.component==='results'?1.6:datasetChanged?1.45:1.25;
    gsap.set('#transition',{visibility:'visible'});
    // Reset pixel and percent coordinates independently. A CSS translateY and
    // GSAP yPercent otherwise add together and can leave the cover offscreen.
    gsap.set('.transition-sheet',{x:0,y:0,xPercent:0,yPercent:102,backgroundColor:color});
    gsap.set('.transition-title',{opacity:0,x:0,y:22});
    state.transitionTl=gsap.timeline({paused:state.paused,onComplete:()=>{
      state.transitioning=false;
      state.sceneStarted=performance.now();
      gsap.set('#transition',{visibility:'hidden'});
      gsap.set('.scene-copy > *',{clearProps:'transform,opacity'});
      if(scene.component==='results')revealResults();
    }})
      .to('.transition-sheet',{yPercent:0,duration,ease:'power3.inOut'})
      .to('.transition-title',{opacity:1,y:0,duration:.65,ease:'power2.out'})
      .call(()=>applyScene(scene))
      .to('.transition-title',{opacity:0,y:-16,duration:.55,ease:'power2.in'},'+=.7')
      .to('.transition-sheet',{yPercent:-102,duration:duration*.95,ease:'power3.inOut'},'+=.06')
      .fromTo('.scene-copy > *',{y:12,opacity:0},{y:0,opacity:1,stagger:.09,duration:.8,ease:'power2.out'},'-=.3');
  }
  function pump(){
    if(state.paused||state.transitioning)return;
    const delay=MIN_SLIDE_READING_MS;
    if(state.current&&performance.now()-state.sceneStarted<delay)return;
    let next;
    if(state.replaying){next=state.replayQueue.shift();if(!next){state.replaying=false;$('replay').textContent='↶ Replay recent';$('playback-label').textContent='RECENT OBSERVED ACTIVITY / METRICS STAY LIVE';toast('Back to the live presentation.');}}
    if(!next)next=state.queue.shift();
    if(!next&&state.replayReturn){next=state.replayReturn;state.replayReturn=null;}
    if(next)transition(next);
  }
  function updateMetrics(event){
    if(event.dataset)$('metric-dataset').textContent=normalizeDataset(event.dataset)==='CIFAR10'?'CIFAR-10':'MNIST';
    if(event.bits!=null)$('metric-bits').textContent=String(event.bits);
    if(event.epoch!=null)$('metric-epoch').textContent=String(event.epoch);
    if(event.batch!=null){$('metric-batch').textContent=`${event.batch}${event.total_batches?' / '+event.total_batches:''}`;if(event.total_batches)$('progress-fill').style.width=`${Math.min(100,event.batch/event.total_batches*100)}%`;}
    if(finite(event.loss))$('metric-loss').textContent=event.loss.toFixed(3);
    if(finite(event.accuracy))$('metric-accuracy').textContent=event.accuracy.toFixed(1)+'%';
    if(event.type==='batch')$('metric-note').textContent=`LIVE ${event.phase==='evaluate'?'EVALUATION':'TRAINING'} · Current batch measurements. Visual playback may show a recent earlier operation.`;
    if(event.type==='epoch')$('metric-note').textContent=`LIVE ${event.phase==='evaluate'?'EVALUATION':'TRAINING'} · Whole-epoch measurements. Pausing visuals does not pause training.`;
  }
  function status(event){
    const name=event.status;$('status').textContent=event.message||name;$('status-light').className=name==='running'?'running':name==='error'?'error':'';
    if(['completed','complete','finished','error','failed','stopped'].includes(name)){
      state.running=false;state.terminal=name;$('start').disabled=false;$('stop').disabled=true;$('run-mode').disabled=false;
      if(['completed','complete','finished'].includes(name)){state.completed=true;enqueue('results',state.current||{});$('metric-note').textContent='PROGRAM COMPLETE · Final results use the best held-out test accuracy for each model.';}
      else{state.queue=[];state.replayQueue=[];state.replaying=false;clearTransition();state.revealTl?.kill();network?.setPaused(true);if(window.gsap)gsap.set('#transition',{visibility:'hidden'});$('playback-label').textContent=name==='stopped'?'PROGRAM STOPPED · PARTIAL RUN':'PROGRAM ERROR · SEE MESSAGES';toast(event.message||'The run ended before completion.');if(name==='error')showMessages(true);}
    }
  }
  function receive(event){
    if(event.seq!=null)state.seq=Math.max(state.seq,event.seq);
    if(event.type==='snapshot'){
      state.snapshots.set(keyFor(event),event);if(state.snapshots.size>10)state.snapshots.delete(state.snapshots.keys().next().value);
      if(!state.paused&&!state.replaying&&state.current&&keyFor(event)===keyFor(state.current))network?.setSnapshot(event);
      return;
    }
    if(['batch','epoch','result'].includes(event.type))updateMetrics(event);
    if(event.type==='result'&&finite(event.accuracy)){state.results.set(keyFor(event),{...event,dataset:normalizeDataset(event.dataset)});return;}
    if(event.type==='artifact'){if(event.path&&!state.artifacts.includes(event.path))state.artifacts.push(event.path);return;}
    if(event.type==='table'){
      state.tables.push(event);if(state.tables.length>20)state.tables.shift();$('table-list').replaceChildren();
      for(const tableEvent of state.tables){const section=document.createElement('section');const title=document.createElement('h3');title.textContent=tableEvent.title||'Program table';const table=document.createElement('table');const head=document.createElement('tr');for(const value of tableEvent.columns||[]){const cell=document.createElement('th');cell.textContent=String(value);head.append(cell);}table.append(head);for(const row of (tableEvent.rows||[]).slice(0,100)){const tr=document.createElement('tr');for(const value of Array.isArray(row)?row:(tableEvent.columns||[]).map(k=>row[k])){const td=document.createElement('td');td.textContent=String(value??'');tr.append(td);}table.append(tr);}section.append(title,table);$('table-list').append(section);}$('table-count').textContent=String(state.tables.length);return;
    }
    if(event.type==='activity'&&event.component){if(event.component!=='results')enqueue(event.component,event);return;}
    if(event.type==='stage'){
      const text=String(event.stage||'').toLowerCase();
      if(text.includes('dataset')||text.includes('loading'))enqueue('data',event);
      else if(text.includes('evaluation'))enqueue('evaluate',event);
      else if(text.includes('training'))enqueue('forward',event);
      else if(text.includes('quantization'))enqueue('quantize',event);
      if(event.message)log(event.message);
    }
    if(event.type==='log'&&event.message)log(event.message);
    if(event.type==='status'){status(event);if(event.message)log(event.message);}
  }
  async function poll(){
    if(polling||!api())return;polling=true;
    try{const response=await api().poll(state.seq);if(response.dropped)log('Some older visual events were skipped to keep the display current.');for(const e of response.events||[])receive(e);$('bridge-note').textContent='DESKTOP CONNECTED';}
    catch(e){$('bridge-note').textContent='Connection interrupted';log('Desktop connection: '+e.message);}
    finally{polling=false;}
  }
  function reset(){
    document.body.classList.remove('visuals-paused');state.cssPausedAnimations?.forEach(a=>a.play());state.cssPausedAnimations=[];
    state.sceneArchive.clear();state.seenScenes.clear();state.replayReturn=null;network?.destroy();network=null;ensureNetwork();
    state.queue=[];state.replayQueue=[];state.history=[];state.snapshots.clear();state.results.clear();state.seenDatasets.clear();state.artifacts=[];state.logs=[];state.tables=[];$('table-list').replaceChildren();$('table-count').textContent='0';state.paused=false;state.replaying=false;state.terminal=null;state.completed=false;clearTransition();state.revealTl?.kill();state.current=null;
    if(window.gsap)gsap.set('#transition',{visibility:'hidden'});$('log').textContent='';$('message-count').textContent='0';$('pause').textContent='Ⅱ Pause visuals';$('pause').setAttribute('aria-pressed','false');$('replay').textContent='↶ Replay recent';$('replay').disabled=true;for(const name of ['dataset','bits','epoch','batch','loss','accuracy'])$('metric-'+name).textContent='—';$('progress-fill').style.width='0';$('results').hidden=true;$('scene').hidden=false;network?.setPaused(false);setNetworkFocus(false);setModeNote();
  }
  async function start(){
    if(state.running)return;if(!api()){toast('The desktop bridge is not connected. Launch with launch_visualizer.py.');return;}
    state.mode=$('run-mode').value;reset();$('start').disabled=true;$('run-mode').disabled=true;
    try{const response=await api().start(state.mode);if(!response.ok)throw Error(response.error||'Could not start Python.');state.running=true;$('stop').disabled=false;$('status').textContent='Starting the learning machine…';$('status-light').className='running';transition({component:'data',dataset:'MNIST',bits:32,event:{},phase:'setup'},true);}
    catch(e){$('start').disabled=false;$('run-mode').disabled=false;$('status').textContent='Could not start';toast(e.message);log(e.message);}
  }
  async function stop(){if(!api())return;try{const response=await api().stop();if(!response.ok)throw Error(response.error||'Could not stop');}catch(e){toast(e.message);}}
  function togglePause(){
    state.paused=!state.paused;$('pause').setAttribute('aria-pressed',String(state.paused));$('pause').textContent=state.paused?'▶ Resume visuals':'Ⅱ Pause visuals';
    document.body.classList.toggle('visuals-paused',state.paused);
    if(state.paused){state.cssPausedAnimations=document.getAnimations().filter(a=>a.playState==='running');state.cssPausedAnimations.forEach(a=>a.pause());}
    else{state.cssPausedAnimations?.forEach(a=>a.play());state.cssPausedAnimations=[];}
    network?.setPaused(state.paused);if(state.paused){state.pauseAt=performance.now();state.transitionTl?.pause();state.revealTl?.pause();$('playback-label').textContent='VISUALS PAUSED / TRAINING CONTINUES';}
    else{state.sceneStarted+=performance.now()-state.pauseAt;state.transitionTl?.resume();state.revealTl?.resume();$('playback-label').textContent=state.replaying?'REPLAY / RECENT OBSERVED ACTIVITY':'RECENT OBSERVED ACTIVITY / METRICS STAY LIVE';}
  }
  function replay(){
    if(state.replaying){state.replaying=false;state.replayQueue=[];state.sceneStarted=0;$('replay').textContent='↶ Replay recent';toast('Returning to live activity.');return;}
    const scenes=state.history.filter(x=>x.component!=='welcome').slice(-8);if(scenes.length<2)return;
    if(state.paused)togglePause();state.replayReturn=state.current;clearTransition();state.replaying=true;state.replayQueue=scenes.map(x=>({...x}));state.sceneStarted=0;$('replay').textContent='↗ Return live';network?.setPaused(false);toast('Replaying recent observed stages. Python keeps running.');pump();
  }
  function showMessages(show){$('message-panel').hidden=!show;$('messages').setAttribute('aria-expanded',String(show));if(show)$('log').scrollTop=$('log').scrollHeight;}
  function renderResults(){
    const target=$('results-cards');target.replaceChildren();
    $('results-mode').textContent=state.mode==='demo'?'ILLUSTRATIVE DEMO — these are example numbers, not measured training results.':state.mode==='quick'?'QUICK REAL-DATA RUN — trained on small subsets for one epoch. These scores show the process, not a fair benchmark of the full datasets.':'FULL EXPERIMENT — best held-out test score observed for each trained model.';
    for(const dataset of ['MNIST','CIFAR10']){
      const rows=[1,4,8,32].map(bits=>state.results.get(`${dataset}:${bits}`)).filter(Boolean);
      if(!rows.length)continue;
      const section=document.createElement('section');section.className='dataset-result '+(dataset==='CIFAR10'?'cifar':'mnist');const header=document.createElement('header');const h=document.createElement('h2');h.textContent=dataset==='CIFAR10'?'CIFAR–10 / color photographs':'MNIST / handwritten digits';const sub=document.createElement('p');sub.textContent='BEST TEST ACCURACY · HIGHER IS BETTER';header.append(h,sub);section.append(header);
      const grid=document.createElement('div');grid.className='score-grid';const best=Math.max(...rows.map(x=>x.accuracy));
      for(const bits of [1,4,8,32]){
        const row=rows.find(x=>Number(x.bits)===bits);const card=document.createElement('article');card.className='score-card'+(row&&row.accuracy===best?' best':'');const precision=document.createElement('div');precision.className='precision';precision.textContent=bits+' bit'+(bits===32?' / baseline':'');card.append(precision);
        if(row&&row.accuracy===best){const tag=document.createElement('span');tag.className='best-tag';tag.textContent=state.mode==='demo'?'EXAMPLE':'HIGHEST SCORE';card.append(tag);}
        const score=document.createElement('div');score.className='score';const number=document.createElement('span');number.className='score-number';number.dataset.score=String(row?.accuracy??0);number.textContent=row?row.accuracy.toFixed(1):'—';const unit=document.createElement('small');unit.textContent=row?'%':'';score.append(number,unit);card.append(score);
        const explanation=document.createElement('p');explanation.textContent=row?`About ${Math.round(row.accuracy)} out of 100 images guessed correctly.`:'No completed measurement for this model.';card.append(explanation);
        const bar=document.createElement('div');bar.className='score-bar';const fill=document.createElement('span');fill.dataset.percent=String(row?Math.max(0,Math.min(100,row.accuracy)):0);fill.style.width=fill.dataset.percent+'%';bar.append(fill);card.append(bar);grid.append(card);
      }
      section.append(grid);const summary=document.createElement('p');summary.className='dataset-summary';const baseline=rows.find(x=>Number(x.bits)===32),lowest=rows.find(x=>Number(x.bits)===1);
      if(baseline&&lowest){const delta=lowest.accuracy-baseline.accuracy;summary.textContent=`Here, the 1-bit model got about ${Math.round(Math.abs(delta))} ${delta>=0?'more':'fewer'} answers right per 100 images than the 32-bit model. ${state.mode==='quick'?'This is a tiny training run, so do not treat the ranking as a final conclusion.':'A different run can give different scores.'}`;}section.append(summary);target.append(section);
    }
    if(!state.results.size){const p=document.createElement('p');p.textContent='The program completed without comparable model results. See Program messages for details.';target.append(p);}
    const files=$('artifacts');files.replaceChildren();for(const path of state.artifacts){const button=document.createElement('button');button.className='artifact-button';button.textContent='Open '+String(path).split(/[\\/]/).pop()+' ↗';button.addEventListener('click',async()=>{try{const r=await api()?.open_artifact(path);if(r&&!r.ok)throw Error(r.error||'Could not open output');}catch(e){toast(e.message);}});files.append(button);}
    $('results').scrollTop=0;
    if(state.reduced)revealResults();
  }
  function revealResults(){
    if(!window.gsap||state.reduced)return;
    state.revealTl?.kill();state.revealTl=gsap.timeline().fromTo('.results-intro > *',{y:45,opacity:0},{y:0,opacity:1,duration:.9,stagger:.12,ease:'power3.out'}).fromTo('.score-card',{y:70,opacity:0,rotate:2},{y:0,opacity:1,rotate:0,duration:1,stagger:.08,ease:'power3.out'},'-=.4');
    document.querySelectorAll('.score-number').forEach(el=>{const score={value:0};state.revealTl.to(score,{value:Number(el.dataset.score),duration:1.4,ease:'power2.out',onUpdate:()=>el.textContent=score.value.toFixed(1)},.65);});
    document.querySelectorAll('.score-bar span').forEach(el=>state.revealTl.fromTo(el,{width:'0%'},{width:el.dataset.percent+'%',duration:1.5,ease:'power3.out'},.7));
  }
  function setReduced(value){state.reduced=value;document.body.classList.toggle('reduced-motion',value);$('motion').setAttribute('aria-pressed',String(value));$('motion').textContent=value?'◉ Reduced':'◉ Motion';network?.setReducedMotion?.(value);if(value){state.transitionTl?.progress(1);state.revealTl?.progress(1);lenis?.stop();}else lenis?.start();}
  $('start').addEventListener('click',start);$('stop').addEventListener('click',stop);$('pause').addEventListener('click',togglePause);$('replay').addEventListener('click',replay);$('motion').addEventListener('click',()=>setReduced(!state.reduced));$('run-mode').addEventListener('change',()=>{state.mode=$('run-mode').value;setModeNote();});$('messages').addEventListener('click',()=>showMessages($('message-panel').hidden));$('close-messages').addEventListener('click',()=>showMessages(false));$('reset-view').addEventListener('click',()=>{setNetworkFocus(false);network?.resetView();});$('close-inspector').addEventListener('click',()=>{$('inspector').hidden=true;setNetworkFocus(false);});$('back-network').addEventListener('click',()=>{const recent=[...state.history].reverse().find(x=>x.component!=='results');if(recent)transition(recent,true);});document.querySelector('.brand').addEventListener('click',e=>{e.preventDefault();network?.resetView();});document.addEventListener('keydown',e=>{if(e.key==='Escape'){showMessages(false);setNetworkFocus(false);network?.resetView();}if(e.code==='Space'&&!['INPUT','SELECT','BUTTON','TEXTAREA'].includes(document.activeElement.tagName)){e.preventDefault();togglePause();}});
  window.addEventListener('pywebviewready',()=>{$('bridge-note').textContent='DESKTOP CONNECTED';poll();});
  ensureNetwork();setReduced(state.reduced);applyScene({component:'welcome',dataset:'MNIST',bits:32,event:{}});
  if(window.Lenis){try{lenis=new Lenis({wrapper:$('results'),content:$('results-content'),smoothWheel:true,lerp:.09,autoRaf:true});if(state.reduced)lenis.stop();}catch(e){log('Smooth scrolling unavailable: '+e.message);}}
  setInterval(poll,200);setInterval(pump,100);
  window.lab={state,receive,pump,transition,applyScene,start,stop,togglePause,replay,renderResults,setReduced,get network(){return network;}};
})();
