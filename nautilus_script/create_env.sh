#!/usr/bin/env bash
#
# Create a Conda environment that contains
#   – CUDA-enabled PyTorch
#   – xformers (built from source)
#   – PyTorch3D
# Adjust the version variables below as needed.
#
# ---------------------------- User Settings -----------------------------
ENV_NAME=xformer_cuda_nautilus           # 🔧 Conda environment name
PYTHON_VER=3.10                 # 🔧 Python version (3.10)
CUDA_TAG=cu121                  # 🔧 cu118 · cu121 · etc.
PT_VER=2.1.2                    # 🔧 PyTorch / torchaudio version
TV_VER=0.16.2                   # 🔧 TorchVision version
XFORMERS_VER=0.0.23.post1       # 🔧 xformers git tag (v${XFORMERS_VER})
PYTORCH3D_VER=0.7.7             # 🔧 PyTorch3D version
# -----------------------------------------------------------------------

set -e

# Ensure that Conda is installed.
if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is not installed. Please install Miniconda or Anaconda first." >&2
  exit 1
fi

echo ">>> 1) Creating Conda environment [$ENV_NAME] (Python $PYTHON_VER) ..."
conda create -y -n "$ENV_NAME" python="$PYTHON_VER"

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
echo ">>> 2) Activating Conda environment ..."
conda activate "$ENV_NAME"

echo ">>> 3) Upgrading pip & setuptools ..."
pip install -q --upgrade pip setuptools wheel

echo ">>> 4) Installing PyTorch ($PT_VER + $CUDA_TAG) ..."
pip install \
  --index-url "https://download.pytorch.org/whl/$CUDA_TAG" \
  torch=="$PT_VER" \
  torchvision=="$TV_VER" \
  torchaudio=="$PT_VER"

echo ">>> 5) Installing PyTorch3D ($PYTORCH3D_VER) ..."
PY_NO_DOT=${PYTHON_VER/./}        # e.g., '310'
PT_MAJOR_MINOR=$(echo "$PT_VER" | cut -d'.' -f1,2 | tr -d '.')  # '21' for 2.1.*
CU_NO_DOT=${CUDA_TAG/cu/}         # '121'
WHEEL_URL="https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py${PY_NO_DOT}_cu${CU_NO_DOT}_pyt${PT_MAJOR_MINOR}/download.html"

# Try pre-built wheel first; if not found, fall back to building from source.
if ! pip install "pytorch3d==$PYTORCH3D_VER" -f "$WHEEL_URL"; then
  echo ">>> Pre-compiled wheel unavailable. Building PyTorch3D from source (this may take several minutes) ..."
  # Build dependencies
  pip install -q "cmake>=3.18" ninja "pybind11>=2.12" "packaging" "tqdm"
  pip install "git+https://github.com/facebookresearch/pytorch3d.git@v$PYTORCH3D_VER"
fi

echo ">>> 6) Quick import test ..."
python - <<'PYTEST'
import torch, pytorch3d
print("• PyTorch   :", torch.__version__, torch.version.cuda)
print("• PyTorch3D :", pytorch3d.__version__)
PYTEST

echo "✅  Finished!  Activate the environment with:  conda activate $ENV_NAME"

echo ">>> 7) Running mapping scripts ..."
pip install -e .
pip install open3d
pip install tensorboard
pip install wandb