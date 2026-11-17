# Neural Network Quantization Lab

A classroom desktop application that runs the original PyTorch quantization experiment and explains its progress with an interactive neuron diagram, measured activations and weights, GSAP transitions, and readable results.

## Run on another Windows computer

Install Python (tested with Python 3.13), Git, and Microsoft Edge WebView2 Runtime. Clone this repository, open PowerShell in its directory, then run:

```powershell
./setup_visualizer.ps1
python launch_visualizer.py
```

Choose Quick / real data for a short complete walkthrough, Full experiment for the original workload, or Illustrative demo for synthetic teaching examples. The first real run downloads MNIST and CIFAR-10. Each slide provides at least 12 seconds of reading time; pause and replay affect only the presentation.

See [the full guide](WEB_VISUALIZER_README.md) for controls, sampling limits, outputs, and implementation details.

## What's included

The original experiment, both desktop interfaces, Python instrumentation, locally bundled GSAP/Lenis assets and attribution, setup requirements, documentation, and tests are included. Downloadable datasets, the local Python environment, generated training outputs, preview images, and unrelated coursework are excluded. These are recreated locally as needed.

The app currently targets Windows/WebView2. GitHub provides access to source files on any device, not hosted execution of the desktop app. macOS, Linux, and mobile execution require additional platform work.

## Verification

```powershell
.venv/Scripts/python.exe -m unittest test_visualizer test_desktop_bridge -v
```

Browser tests additionally require Playwright and its Chromium installation. `test_web_presentation.py` uses a recorded real-run fixture: generate it with `.venv/Scripts/python.exe -m quantization_visualizer.runtime --mode quick > web_quick_validation.jsonl` before running that test. Generated recordings are not committed.

Full training has not been run to completion during application verification; Quick mode has been verified through all eight experiment models.
