#!/usr/bin/env bash
#
# Make a pip-only virtualenv containing
#   – CUDA-enabled PyTorch
#   – xformers (with pre-compiled CUDA kernels)
#   – PyTorch3D (matching wheel)
#
# ---------------------------- 사용자 설정 -----------------------------
ENV_DIR=$HOME/venvs/xformer_cuda     # 🔧 가상환경 경로
PYTHON_BIN=python3                   # 🔧 사용할 Python 실행 파일
CUDA_TAG=cu121                       # 🔧 cu118 · cu121 · etc.
PT_VER=2.1.2                         # 🔧 PyTorch / torchaudio 버전
TV_VER=0.16.2                        # 🔧 TorchVision 버전
XFORMERS_VER=0.0.23.post1            # 🔧 xformers 버전
PYTORCH3D_VER=0.7.7                  # 🔧 PyTorch3D 버전
# ---------------------------------------------------------------------

set -e

echo ">>> 1) Creating virtualenv [$ENV_DIR] ..."
$PYTHON_BIN -m venv "$ENV_DIR"

echo ">>> 2) Activating virtualenv ..."
# shellcheck source=/dev/null
source "$ENV_DIR/bin/activate"

echo ">>> 3) Upgrading pip & setuptools ..."
pip install -q --upgrade pip setuptools wheel

echo ">>> 4) Installing PyTorch ($PT_VER + $CUDA_TAG) ..."
pip install \
  --index-url "https://download.pytorch.org/whl/$CUDA_TAG" \
  torch=="$PT_VER" \
  torchvision=="$TV_VER" \
  torchaudio=="$PT_VER"

echo ">>> 5) Installing xformers ($XFORMERS_VER) ..."
pip install "xformers==$XFORMERS_VER"

echo ">>> 6) Installing PyTorch3D ($PYTORCH3D_VER) ..."
PY=${PYTHON_BIN##*python}          # e.g., '3.10'
PY_NO_DOT=${PY/./}                 # '310'
PT_NO_DOT=${PT_VER/./}             # '212'
CU_NO_DOT=${CUDA_TAG/cu/}          # '121'
WHEEL_URL="https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py${PY_NO_DOT}_cu${CU_NO_DOT}_pyt${PT_NO_DOT}/download.html"

pip install "pytorch3d==$PYTORCH3D_VER" -f "$WHEEL_URL"

echo ">>> 7) Quick import test"
python - <<'PYTEST'
import torch, xformers, pytorch3d
print("• PyTorch   :", torch.__version__, torch.version.cuda)
print("• xformers  :", xformers.__version__)
print("• PyTorch3D :", pytorch3d.__version__)
PYTEST

echo "✅  Finished!  Activate with:  source $ENV_DIR/bin/activate"
