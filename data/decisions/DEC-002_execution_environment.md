---
id: DEC-002
type: decision
supersede: null
---

# DEC-002 — Entorno de ejecución en WSL con venv en ext4 y pip arrancado sin sudo

## Trigger
La primera ejecución real del pipeline de EXP-001 quedó bloqueada. El diagnóstico
inicial atribuyó el fallo a problemas de comillas al invocar Python desde WSL y a
ambigüedad sobre qué intérprete usar. La inspección posterior mostró una causa
distinta: **no existía ningún entorno con el stack instalado en ninguna parte**.

Estado encontrado en la máquina:

- `.venv/` del repo (Windows, Python 3.11.9): solo `pip` y `setuptools`, 22 MB.
  Sin `torch`, `transformers` ni `datasets`.
- conda `base`: `torch 2.10.0+cpu`, sin CUDA.
- WSL Ubuntu 22.04.5: Python 3.10.12, `nvidia-smi` operativo (passthrough CUDA OK),
  pero sin venv del proyecto.
- GPU: RTX 4090, 24 GiB, driver 591.86 — visible tanto en Windows como en WSL.

## Alternativas consideradas
- **Opción A — Windows nativo sobre el `.venv` existente.** La 4090 es visible sin
  capa intermedia y desaparece por completo la fricción de rutas/comillas WSL.
- **Opción B — WSL Ubuntu 22.04 con venv propio.** Paridad con el entorno Linux
  habitual del ecosistema de investigación; acceso a `flash-attn`, `vLLM`,
  `bitsandbytes` y demás piezas que en Windows son frágiles o no existen.
- **Opción C — reutilizar un env de conda existente** (`vlm`, `comfyenv`,
  `deepagent`, `fluxgym`). Ahorra descarga pero contamina el entorno de otro
  proyecto y destruye la trazabilidad de dependencias.

## Decisión
**Opción B: WSL Ubuntu 22.04**, con estas concreciones:

1. El venv vive en **`~/.venvs/niel_landa`**, dentro del sistema de archivos ext4
   de la distro — *no* en `/mnt/c`.
2. El stack se declara en `requirements.txt` (capa 2) y se monta con
   `scripts/setup_wsl_env.sh`, que es idempotente y verifica CUDA al terminar.
3. `torch` se instala desde el índice de PyTorch con ruedas CUDA (cu124), no
   desde PyPI, que sirve la variante CPU-only.
4. **pip se arranca sin sudo.** Ubuntu 22.04 viene sin el paquete
   `python3.10-venv`, así que `python3 -m venv` falla en `ensurepip`. Instalarlo
   exige `sudo` con contraseña, que no está disponible en sesiones no
   interactivas. El script detecta el fallo y cae a
   `python3 -m venv --without-pip` + `get-pip.py`, que opera íntegramente en
   espacio de usuario.
5. La caché de Hugging Face se deja en su ruta por defecto dentro de ext4
   (`~/.cache/huggingface`), no en `/mnt/c`.

El `.venv/` de Windows queda **abandonado**, no borrado, para no romper nada que
dependa de él.

## Rationale
- El bloqueo real era la ausencia de entorno, no el shell. Cualquiera de las tres
  opciones lo habría resuelto; se elige por criterios de futuro, no por desbloqueo.
- WSL da acceso a las piezas del ecosistema que Windows no soporta bien. La línea
  de investigación (extracción de activaciones, hooks sobre el residual stream,
  posible paso a `vLLM` o cuantización) tiene alta probabilidad de necesitarlas.
- El venv en ext4 y no en `/mnt/c` es la diferencia entre segundos y minutos:
  instalar decenas de miles de archivos pequeños sobre el bridge 9p es del orden
  de 10x más lento, y algunos symlinks de pip se comportan mal ahí.
- El arranque de pip sin sudo mantiene el setup reproducible en sesiones
  automatizadas, que no pueden responder a una petición de contraseña.

## Riesgo conocido
El disco ext4 de WSL está al **94% de ocupación (64 GiB libres)**. El presupuesto
previsto es ~8 GiB de venv + ~15 GiB de pesos de Qwen 2.5 7B en bf16 ≈ 23 GiB,
que cabe pero deja poco margen. Si se añade un segundo modelo o checkpoints de
activaciones sin podar, habrá que mover la caché de HF o ampliar el VHDX.

## Predicción
Si la decisión es correcta:

- `scripts/setup_wsl_env.sh` termina con `torch.cuda.is_available() == True` y
  `bf16 == True` sobre la RTX 4090.
- Qwen 2.5 7B en bf16 (~15 GiB) carga en los 24 GiB de VRAM sin cuantización ni
  offload para la extracción de activaciones con `batch_size` pequeño.
- No vuelve a aparecer un bloqueo de tipo "no sé qué intérprete usar": hay un
  único intérprete canónico, `~/.venvs/niel_landa/bin/python`.

## Enlaces
- motivated_by: [Q-001]
- habilita: [EXP-001]
- supersede: null
