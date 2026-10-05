!/usr/bin/env bash #Might need to comment out 
set -e

Recommended: Python >= 3.10 because latest stable PyTorch requires Python 3.10+.
CPU install is safest and portable. For CUDA, replace the PyTorch install line with
the command from https://pytorch.org/get-started/locally/ for your CUDA version.

ENV_NAME="lp-mnist"
PYTHON_VERSION="3.11"

conda create -y -n ${ENV_NAME} python=${PYTHON_VERSION}
conda activate ${ENV_NAME}

python -m pip install --upgrade pip

# # CPU version:
# pip install torch torchvision torchaudio

GPU Version:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Utilities:
pip install numpy pyyaml tqdm matplotlib

#Installing SOCP solvers 
pip install cvxpy scs clarabel
pip install cvxpylayers diffcp cvxpy scs


python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("Built CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    try:
        torch.cuda.init()
    except Exception as e:
        print("ERROR:", repr(e))
PY

# python - <<'PY'
# import torch
# print('Torch:', torch.__version__)
# print(torch.version.cuda)
# print('CUDA available:', torch.cuda.is_available())
# print(torch.cuda.device_count())
# PY
