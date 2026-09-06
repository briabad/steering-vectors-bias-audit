## Why

La puerta de existencia abrió: sobre los pools completos de BBQ hay **1063 ítems
distintos** (formato `chat`) en los que Qwen 2.5 7B Instruct rellena un hueco de
información con un prior de grupo. Es el material de contraste que `Q-001` necesita, y
más del doble de los 400 prompts que `SOA-004` empleó para derivar sus vectores.

Este change construye la dirección latente $v_{\text{op}}$ a partir de ese contraste y
mide si existe: H1 de `EXP-002/hypothesis.md`. **No** interviene sobre el modelo — eso es
H2 y va en un change posterior.

El diseño no es evidente y está pre-registrado con detalle en `EXP-002/hypothesis.md`
Addendum 5, porque el contraste ingenuo produciría un vector de tema en lugar de un
vector de operación. Ver `design.md` para el porqué.

Trazabilidad: implementa `FEAT-012` (extractor de activaciones) y abre `FEAT-013` hacia
la construcción de la dirección. Motivado por `Q-001`, `DEC-005` y `SOA-004`; el
protocolo de contraste procede de `EXP-002/hypothesis.md` Addenda 3, 4 y 5.

## What Changes

- **Extractor de activaciones** reutilizable: carga Qwen 2.5 7B en bf16 y captura los 29
  estados ocultos en la posición del último token, bajo `torch.no_grad()`. Compartido con
  los changes de steering y guardarraíl.
- **Constructor del contraste emparejado**: 1:1 por categoría y plantilla
  (`question_index`), a partir de las observaciones ya medidas en
  `EXP-002/output/existence_gate_raw.jsonl`. No requiere repetir la inferencia de la
  puerta.
- **Tres construcciones de la dirección** según `SOA-004`: reading vector, contrastivo y
  PCA con reescalado por la norma real de las activaciones.
- **Selección de capa** con un corte reservado de ajuste, congelada antes de tocar el
  holdout.
- **Evaluación de H1**: AUC en holdout contra los baselines léxico y de capa 0, test de
  permutación con reajuste dentro de cada permutación, tamaño de efecto e intervalo
  bootstrap sobre ítems.
- **Protocolo del residuo**: $v_A$ (752 pares por plantilla) frente a $v_B$ (311 por
  categoría, conjunto disjunto), con coseno anclado entre un suelo aleatorio y un techo
  de mitades disjuntas.
- **Reserva de transferencia**: `Race_x_gender`, `Gender_identity` y `Sexual_orientation`
  quedan intactas — no intervienen en ajuste, ni en selección de capa, ni en ninguna
  decisión de este change.

**Fuera de alcance**, explícitamente: la intervención causal (barrido de $\alpha$, KL,
entropía), la prueba de transferencia H3, el diccionario de conceptos, y el bloque de
geometría. Cada uno es un change posterior.

## Capabilities

### New Capabilities
- `control-vector`: extraer activaciones del residual stream, construir direcciones
  latentes a partir de un contraste emparejado, seleccionar capa sin contaminar el
  holdout, y evaluar si la dirección separa clases por encima de baselines declarados.

### Modified Capabilities
_(ninguna — `bbq-existence-gate` no cambia; este change consume su salida cruda)_

## Impact

**Código nuevo**
- `src/bbq_gate/domain/` — entidades del contraste emparejado y álgebra de direcciones
  (construcción, normalización, proyección, coseno), sin I/O.
- `src/bbq_gate/infrastructure/activations.py` — adaptador de extracción con
  `transformers`.
- `src/bbq_gate/application/` — casos de uso de emparejamiento, construcción y
  evaluación.
- `scripts/06_build_operation_vector.py` — punto de entrada.
- `tests/` — unitarios con el extractor mockeado en su frontera.

**Datos**
- Consume `EXP-002/output/existence_gate_raw.jsonl` (287.976 observaciones ya medidas).
- Escribe activaciones en fp16 y `EXP-002/output/operation_vector.json`.

**Dependencias**: ninguna nueva.

**Coste**: ~1500 ítems del contraste emparejado (752 pares) más el residuo; una pasada de
inferencia sin gradientes sobre frases cortas. Minutos de GPU, muy por debajo de las 2 h
de la puerta. Almacenamiento: ~1500 × 29 × 3584 × 2 bytes ≈ **310 MB**.

**Riesgo funcional**: un AUC bajo es un resultado válido de H1, no un fallo del change.
