# Quantization Lab

**The desktop interface has been redesigned with GSAP, Lenis, and interactive neuron inspection. See [the current guide](WEB_VISUALIZER_README.md).** The notes below describe the earlier Tkinter interface, now available through `python -m quantization_visualizer.app`.

A separate Tkinter desktop application that starts and observes `neural_network_quantization.py`. The original file is preserved. All of its sections run in Full experiment mode, including quantization checks, network checks, dataset previews, baseline training, both four-bitwidth experiments, plots, tables, and discussion output.

## Start

From this folder, run `python launch_visualizer.py` (or double-click the launcher if Python file associations are configured). It automatically uses this folder's `.venv` when present. Select a mode and press **Start program**.

On a fresh machine, run `./setup_visualizer.ps1` in PowerShell first. Python with Tkinter is required. Dependencies install only in the project `.venv`. You can also run `python -m quantization_visualizer` with an environment containing the dependencies.

* **Full experiment:** complete original datasets and epoch counts. This is a substantial CPU workload; the baseline alone trains for two epochs and the experiments train eight models for five epochs each.
* **Quick real-data run:** every source section, all eight experiment models, one epoch each, first 256 training and 128 test examples per dataset. This verifies the workflow and demonstrates real learning, but its accuracy is not representative of a full experiment.
* **Illustrative demo:** approximately 45 seconds of labeled simulated events. No dataset access or neural-network computation; useful for checking classroom projection and explaining the colors.

First real run downloads MNIST and CIFAR-10 into `data/`. Internet access is required until both are cached. GUI remains responsive while loading. **Stop** terminates the worker, leaving partial output files. Start again to begin a new run; there is no checkpoint/resume.

If the original CIFAR-10 server fails, the worker uses the [Zenodo archive](https://zenodo.org/records/10089977) and checks it against torchvision's original archive checksum. TLS verification remains enabled. The installed, tested dependency versions are recorded in `requirements-visualizer-lock.txt`.

## Reading the display

Cyan represents data, violet quantized weights, blue forward computation, coral gradients, gold optimizer updates, and green evaluation/results. Explanations accompany the current activity. The picture is schematic, not a view of every neuron. Quantization samples are actual observed convolution weights in real modes. Batch loss/accuracy describe the displayed batch; epoch values are aggregate measurements.

Detailed activity is observed on the first, every twentieth, and final batch of a pass. The screen coalesces rapid events to remain responsive. All batches still compute; the full run is not reduced by visualization sampling. Arrows and traveling particles explain direction and type of computation rather than numerical activation magnitudes or hardware utilization.

Only convolutional weights use the selected quantization. Dense layers remain float32. Full-precision master weights are updated during training. The source's model-size/compression estimates assume all parameters have the chosen bit width; they are theoretical estimates and overstate storage savings for this mixed-precision model. The companion preserves those source results and labels this limitation.

## Outputs and execution details

Each real run writes to a timestamped `visualizer_runs/` directory. View tables, learning curves, and messages in the lower tabs. Saved output choices open generated PNG/CSV files in their associated applications.

The worker inserts observation calls into the source's Python AST in memory, supplies notebook-style `display`, replaces notebook progress bars with structured events, and captures Matplotlib figures without blocking windows. DataLoader runs with zero subprocess workers for Windows compatibility. CPU threads are limited to four. Model equations, optimization, full-mode batch sizes, epochs, and experiment ordering are preserved. Quick mode explicitly reduces data and epoch counts. Source edits that rename functions or variables may require corresponding instrumentation updates.

The GUI and animation code use only the Python standard library; scientific dependencies are needed for real runs. Errors are shown in Program messages. Demo is never silently substituted when a real run fails.

## Validation

Run `.venv/Scripts/python.exe -m unittest test_visualizer -v`. Tests verify source compilation, exact forward-output and gradient equivalence at 1/4/8/32 bits, and GUI event rendering/cleanup. A complete Quick run has also been verified against real MNIST/CIFAR-10: eight experiment results, 18 train/evaluation epoch summaries including the baseline check, and generated figures/CSV. Full mode has not been trained to completion during setup.
