"""Double-click or run with Python to open Quantization Lab."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
local_python = root / '.venv' / 'Scripts' / 'python.exe'
if local_python.exists() and Path(sys.executable).resolve() != local_python.resolve():
    raise SystemExit(subprocess.call([str(local_python), str(Path(__file__).resolve())], cwd=root,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)))
from quantization_visualizer.desktop import main
main()
