#!/usr/bin/env bash
# Monta el entorno de ejecución del proyecto dentro de WSL (Ubuntu 22.04).
#
# El venv vive en el sistema de archivos ext4 de WSL (~/.venvs/niel_landa), NO en
# /mnt/c: instalar miles de archivos pequeños sobre el bridge 9p de /mnt/c es
# ~10x más lento y rompe algunos symlinks de pip.
#
# Uso:  bash scripts/setup_wsl_env.sh
set -euo pipefail

VENV="${VENV:-$HOME/.venvs/niel_landa}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

echo "== repo:  $REPO"
echo "== venv:  $VENV"
echo "== torch: $TORCH_INDEX"

if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "-- venv ya existe y tiene pip, reutilizando"
else
  echo "-- creando venv"
  rm -rf "$VENV"
  # Ubuntu 22.04 parte sin el paquete python3.10-venv, así que `python3 -m venv`
  # falla en el paso de ensurepip. Instalarlo requiere sudo con contraseña, que no
  # siempre está disponible: si el camino normal falla, creamos el venv sin pip y
  # lo arrancamos con get-pip.py, que no necesita permisos de sistema.
  if ! python3 -m venv "$VENV" 2>/dev/null; then
    echo "-- ensurepip no disponible; arrancando pip con get-pip.py"
    rm -rf "$VENV"
    python3 -m venv --without-pip "$VENV"
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$VENV/get-pip.py"
    "$VENV/bin/python" "$VENV/get-pip.py"
    rm -f "$VENV/get-pip.py"
  fi
fi

PY="$VENV/bin/python"
"$PY" -m pip --version
"$PY" -m pip install --upgrade pip wheel setuptools

echo "-- instalando torch (ruedas CUDA)"
"$PY" -m pip install --index-url "$TORCH_INDEX" torch

echo "-- instalando el resto del stack"
"$PY" -m pip install -r "$REPO/requirements.txt"

echo
echo "== verificación =="
"$PY" - <<'PYEOF'
import importlib, platform, sys

print(f"python      {platform.python_version()}  ({sys.executable})")

import torch
print(f"torch       {torch.__version__}")
print(f"cuda avail  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device      {torch.cuda.get_device_name(0)}")
    print(f"capability  sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"vram        {total:.1f} GiB")
    print(f"bf16        {torch.cuda.is_bf16_supported()}")

for mod in ("transformers", "datasets", "accelerate", "huggingface_hub",
            "sentencepiece", "numpy", "scipy", "sklearn"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:<16}{getattr(m, '__version__', 'sin __version__')}")
    except Exception as exc:  # noqa: BLE001 - queremos ver cualquier fallo de import
        print(f"{mod:<16}FALLO: {exc}")
PYEOF

echo
echo "OK. Intérprete del proyecto: $PY"
