import json
from pathlib import Path
from textwrap import dedent

root = Path(__file__).resolve().parent
files = [
    "config/model_50m_colab.yaml",
    "requirements.txt",
    "data/sample_corpus.txt",
    "data/prepare.py",
    "src/__init__.py",
    "src/model.py",
    "src/dataset.py",
    "src/train.py",
    "src/evaluate.py",
    "src/generate.py",
]
manifest = {}
for rel in files:
    manifest[rel] = (root / rel).read_text(encoding="utf-8")

manifest_json = json.dumps(manifest)

cells = []

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Foundation LLM — 50M Parameter GPT\n",
        "\n",
        "Self-bootstrapping notebook for Google Colab (free T4 GPU).\n",
        "\n",
        "**Steps:** Runtime → Change runtime type → **GPU** → Run all\n",
        "\n",
        "Project is created on Google Drive at `MyDrive/foundation-llm/`."
    ]
})

cells.append({
    "cell_type": "code",
    "metadata": {},
    "source": [
        "# Cell 1: GPU check\n",
        "import torch\n",
        "assert torch.cuda.is_available(), 'Enable GPU: Runtime -> Change runtime type -> GPU'\n",
        "!nvidia-smi\n",
        "print('GPU ready:', torch.cuda.get_device_name(0))"
    ],
    "outputs": [],
    "execution_count": None
})

cells.append({
    "cell_type": "code",
    "metadata": {},
    "source": [
        "# Cell 2: Mount Google Drive\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "from pathlib import Path\n",
        "PROJECT_ROOT = Path('/content/drive/MyDrive/foundation-llm')\n",
        "PROJECT_ROOT.mkdir(parents=True, exist_ok=True)\n",
        "%cd {PROJECT_ROOT}\n",
        "print('Project root:', PROJECT_ROOT)"
    ],
    "outputs": [],
    "execution_count": None
})

scaffold_source = dedent(f"""
# Cell 3: Scaffold project files on Google Drive
import json
from pathlib import Path

PROJECT_ROOT = Path('/content/drive/MyDrive/foundation-llm')
MANIFEST = json.loads({manifest_json!r})

written = 0
for rel_path, content in MANIFEST.items():
    out = PROJECT_ROOT / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists() or out.read_text(encoding='utf-8') != content:
        out.write_text(content, encoding='utf-8')
        written += 1

for sub in ['checkpoints', 'data/processed', '.cache']:
    (PROJECT_ROOT / sub).mkdir(parents=True, exist_ok=True)

print(f'Scaffold complete. Updated {{written}} files at {{PROJECT_ROOT}}')
""").strip().splitlines(True)

cells.append({
    "cell_type": "code",
    "metadata": {},
    "source": scaffold_source,
    "outputs": [],
    "execution_count": None
})

cells.append({
    "cell_type": "code",
    "metadata": {},
    "source": [
        "# Cell 4: Install dependencies\n",
        "!pip install -q -r requirements.txt\n",
        "import torch, tiktoken, yaml\n",
        "print('Dependencies installed. PyTorch:', torch.__version__)"
    ],
    "outputs": [],
    "execution_count": None
})

cells.append({
    "cell_type": "code",
    "metadata": {},
    "source": [
        "# Cell 5: Smoke test (sample data + 10 training steps)\n",
        "!python data/prepare.py --sample\n",
        "\n",
        "import subprocess, sys\n",
        "from pathlib import Path\n",
        "\n",
        "PROJECT_ROOT = Path('/content/drive/MyDrive/foundation-llm')\n",
        "result = subprocess.run(\n",
        "    [sys.executable, 'src/train.py', '--config', 'config/model_50m_colab.yaml', '--max_steps', '10'],\n",
        "    cwd=PROJECT_ROOT,\n",
        "    check=False,\n",
        ")\n",
        "if result.returncode != 0:\n",
        "    raise RuntimeError('Smoke test training failed')\n",
        "\n",
        "result = subprocess.run(\n",
        "    [sys.executable, 'src/generate.py', '--checkpoint', 'checkpoints/latest.pt', '--prompt', 'The history of AI', '--max_tokens', '50'],\n",
        "    cwd=PROJECT_ROOT,\n",
        "    check=False,\n",
        ")\n",
        "print('\\n=== SMOKE TEST PASSED ===' if result.returncode == 0 else 'Generation failed')"
    ],
    "outputs": [],
    "execution_count": None
})

cells.append({
    "cell_type": "code",
    "metadata": {},
    "source": [
        "# Cell 6: Prepare OpenWebText subset (first full run only; skip if train.bin exists)\n",
        "from pathlib import Path\n",
        "train_bin = Path('data/processed/train.bin')\n",
        "meta = Path('data/processed/meta.pkl')\n",
        "if train_bin.exists() and meta.exists():\n",
        "    print('Processed data already exists. Skipping download.')\n",
        "    print(f'train.bin size: {train_bin.stat().st_size / 1e9:.2f} GB')\n",
        "else:\n",
        "    !python data/prepare.py --subset --max_docs 200000"
    ],
    "outputs": [],
    "execution_count": None
})

cells.append({
    "cell_type": "code",
    "metadata": {},
    "source": [
        "# Cell 7: Train 50M model (auto-resumes from Drive checkpoints)\n",
        "!python src/train.py --config config/model_50m_colab.yaml --resume auto"
    ],
    "outputs": [],
    "execution_count": None
})

cells.append({
    "cell_type": "code",
    "metadata": {},
    "source": [
        "# Cell 8: Generate sample text\n",
        "!python src/generate.py --checkpoint checkpoints/latest.pt --prompt \"Once upon a time\" --max_tokens 200 --temperature 0.8"
    ],
    "outputs": [],
    "execution_count": None
})

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "colab": {
            "provenance": []
        }
    },
    "cells": cells
}

out = root / "Foundation_LLM_50M.ipynb"
out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print("Wrote", out)
print("Notebook cells:", len(cells))
