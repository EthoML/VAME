import os
import subprocess
import sys


def test_import_vame_does_not_load_umap():
    # umap takes seconds to import, which slows every `import vame` and each preprocessing worker
    code = "import sys, vame; print(vame.__file__); print('umap' in sys.modules)"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    vame_file, umap_loaded = result.stdout.strip().splitlines()[-2:]
    assert umap_loaded == "False", f"umap was imported by {vame_file}"
