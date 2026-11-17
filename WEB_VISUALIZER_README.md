# Quantization Lab — desktop presentation

Run `python launch_visualizer.py`. The launcher uses the project `.venv`. The new application is a native desktop window powered by Microsoft Edge WebView2, with local GSAP animations, Lenis scrolling, and an interactive canvas network. It runs `neural_network_quantization.py`; the original source remains unchanged.

On a fresh machine, run `./setup_visualizer.ps1`. Python and Microsoft Edge WebView2 Runtime are required. Once dependencies and datasets are installed, no internet connection is needed. The previous Tkinter interface is still available via `python -m quantization_visualizer.app`.

## Presentation controls

Each slide stays readable for at least 12 seconds after its entrance transition finishes, including in reduced-motion mode. Transitions move more slowly, and the slide stays longer when no next training event is available. Training can run ahead without shortening this reading time.

* **Start learning:** choose Quick / real data, Full experiment, or Illustrative demo.
* **Pause visuals:** freeze the presentation to discuss a scene. Training and live measurements continue. Resume continues the bounded visual story.
* **Replay recent:** revisit recent observed scenes without retraining. Return live ends the replay.
* **Motion:** enable reduced motion. System reduced-motion preferences are also respected.
* **Stop:** terminate computation. Start begins a fresh run; there is no checkpoint resume.

Scroll over the network to zoom, drag to pan, and select a node to inspect its data. Reset view restores the overview. Keyboard: arrows select nodes, Enter opens details, Escape closes them, +/- zoom, R resets. The selected inspector can be scrolled to reach spatial activations and incoming weights.

The presentation moves through data, quantization, prediction, backward gradients, weight updates, evaluation, and results. MNIST and CIFAR-10 have contrasting visual worlds. Animated particles explain direction and type of computation; they are not measured hardware signals. Live metrics can be ahead of the visual story.

## What the network shows

The overview contains all 32 and 64 convolution feature-map channels, all 128 dense neurons, and all 10 class-score neurons. A convolution channel is a spatial map, not one neuron. Selecting it exposes all spatial values (32×32 for conv1, 16×16 for conv2) from the first image in a sampled batch. These are pre-ReLU outputs. ReLU and pooling between layers are explained in the view.

Connection colors distinguish positive and negative weights; thickness reflects magnitude. Convolution curves summarize channel-to-channel kernels, not all-to-all spatial connections. Convolution inspection contains all 27 or 288 incoming kernel weights. The first dense layer displays **32 sampled incoming weights out of 4,096 per neuron**; the output layer contains all 128 incoming weights per neuron. Values are rounded to five decimals. The input preview and predicted/actual labels come from the same measured sample.

Activity is observed on the first, every twentieth, and final batch. Snapshots are rate limited, with new snapshots at model/dataset/phase changes. All selected batches still compute.

## Modes and results

* **Full:** all original sections, complete datasets, two baseline epochs, and eight five-epoch model experiments. Substantial CPU workload.
* **Quick:** all original sections and all eight models, one epoch, 256 training and 128 test images per dataset. These scores demonstrate execution, not benchmark-quality accuracy.
* **Demo:** explicitly labeled synthetic events, maps, weights, and scores. No actual learning or dataset access.

Final results group each dataset separately and explain test accuracy as correct answers out of 100. Only convolution weights are quantized; dense layers and master weights remain full precision. The source's all-parameter compression estimates overstate mixed-precision storage savings and are not presented as measured savings.

Figures and CSV outputs are saved in timestamped `visualizer_runs/` directories. The worker executes an instrumented copy of the source in memory, supplies notebook display support, captures nonblocking plots, uses zero DataLoader subprocess workers for Windows, and caps CPU threads at four. The source's full-mode mathematical operations and experiment ordering are preserved.

Datasets are cached in `data/`. CIFAR-10 uses a checksum-verified [Zenodo mirror](https://zenodo.org/records/10089977) if its primary host fails. TLS verification remains enabled.

## Code and validation

`desktop.py` supplies the native window and bounded Python bridge. `web/` contains the presentation, transitions, network inspection, and locally bundled GSAP/Lenis. `runtime.py` supplies real telemetry. Source changes that rename observed functions/layers may need corresponding instrumentation updates.

Run `.venv/Scripts/python.exe -m unittest test_visualizer test_desktop_bridge -v`. Tests cover exact forward-output and gradient equivalence, original widget regression, bounded polling, stale-worker isolation, and telemetry shapes. The real Quick workflow has completed with all eight model results, 18 epoch summaries, and generated figures/CSV. Full training has not been run to completion during implementation.

Design references: [TensorFlow Playground](https://playground.tensorflow.org/), [Lenis](https://lenis.dev/), and [Basement](https://basement.studio/). See `web/vendor/THIRD_PARTY.md` for animation-library attribution. Playwright is an optional development/testing dependency, not an application requirement.

Browser checks in `test_web_presentation.py --live` exercise the actual Quick worker, pause while training continues, dataset transitions, every dense neuron, spatial-map inspection, result reveal, scrolling, replay, and reduced motion. `test_native_window.py` verifies the real Edge WebView2 renderer and Python bridge. Preview screenshots are in `visualizer_previews/`. A local headless Chromium measurement at 1440×940 produced a median frame interval of 16.7 ms; animation performance will depend on the presenting computer.
