## Why

`EXP-002` quiere aprender una dirección latente que separe los casos en que Qwen 2.5 7B
Instruct **razona apoyándose en un estereotipo** de aquellos en que se abstiene
correctamente. Ese contraste es **conductual**: depende de que el modelo efectivamente
se equivoque. Si Qwen responde casi siempre "no se puede determinar" en los contextos
ambiguos de BBQ, no hay evento de sesgo que contrastar y el experimento no tiene sujeto.

Medir eso cuesta una pasada de inferencia sin gradientes sobre 3680 frases cortas
(media 284 caracteres) y **no requiere extraer activaciones ni construir vectores**. Es
la puerta más barata del plan y decide si tiene sentido gastar el resto. Se hace ahora,
antes de escribir nada de extracción.

Trazabilidad: implementa `FEAT-013` (paso 0). Motivado por `Q-001`, `DEC-005`
(BBQ como corpus de razonamiento) y `DEC-002` (entorno). Criterios y trampas
pre-registrados en `data/experiments/EXP-002_bbq_stereotype_direction/hypothesis.md`,
Addenda 1 y 2.

**Evidencia exploratoria previa.** Un barrido con muestras de 150 ítems por categoría
(`scripts/03_bbq_category_sweep.py`, salida en
`data/experiments/EXP-002_bbq_stereotype_direction/output/category_sweep.json`) mostró
que ninguna categoría reúne por separado sesgo direccional y masa suficientes, y que
**el signo de `s_amb` se invierte entre ejes**: positivo en religión, discapacidad,
orientación sexual, apariencia, nacionalidad y edad; negativo en raza y clase social.
Ese barrido es exploratorio y no sustituye a la medición rigurosa que este change
construye.

## What Changes

- **Primer código de aplicación del repositorio.** Hasta ahora solo existían
  `scripts/` (operacionales) y `tests/validate_graph.py` (sanidad del workspace). Este
  change crea `src/` con la estructura por capas de `docs/architecture.md`.
- Carga de BBQ (`oskarvanderwal/bbq`) sobre **las 11 categorías con sus pools
  completos**, resolviendo la opción "unknown" **por ítem** vía `answer_info`, nunca por
  posición ni por texto.
- Resolución del rol de cada opción (abstención / estereotipada / anti-estereotipada)
  con un emparejador que sobrevive a los tres modos de fallo observados: tags
  compuestos, negaciones `non*`, y códigos de grupo frente a etiquetas textuales. Los
  ítems intersecccionales donde ambas opciones comparten el grupo se **descartan
  explícitamente y se cuentan**.
- Puntuación de las tres opciones por **logits**, no por generación libre, con dos
  formatos de prompt fijados de antemano y las opciones permutadas.
- Cálculo de las métricas de la puerta: `s_amb` (sesgo direccional) **por categoría y
  agrupado**, `rho_unk` (ceguera a la ambigüedad) y accuracy en `disambig` (comprensión
  de la tarea, verificada por categoría).
- Conteo en **ítems distintos**, no en observaciones, e incertidumbre estimada
  **agrupando por ítem**.
- Persistencia del resultado en `data/experiments/EXP-002_bbq_stereotype_direction/output/`
  con semilla, versiones y parámetros.
- **No** se modifican pesos del modelo. **No** se extraen activaciones. **No** se
  construye ningún vector de control. Eso es trabajo de changes posteriores.

## Capabilities

### New Capabilities
- `bbq-existence-gate`: cargar BBQ resolviendo sus opciones semánticamente, puntuar las
  respuestas de un modelo causal por logits sobre un conjunto restringido de opciones, y
  calcular las métricas que deciden si existe un contraste conductual de estereotipo
  utilizable.

### Modified Capabilities
_(ninguna — `openspec/specs/` está vacío; ésta es la primera capacidad del proyecto)_

## Impact

**Código nuevo**
- `src/bbq_gate/domain/` — entidades puras y cálculo de métricas, sin I/O.
- `src/bbq_gate/application/` — el caso de uso que orquesta carga, puntuación y métricas.
- `src/bbq_gate/infrastructure/` — adaptador de Hugging Face y adaptador de scoring con
  `transformers`.
- `scripts/02_bbq_existence_gate.py` — punto de entrada operacional.
- `tests/unit/bbq_gate/` — tests con el proveedor externo mockeado en su frontera
  (`docs/testing.md` §2-5).

**Datos y artefactos**
- Escribe `data/experiments/EXP-002_bbq_stereotype_direction/output/existence_gate.json`.
- No toca `data/index.md` ni la capa de conocimiento: eso se consolida al escribir
  `results.md`, fuera de este change.

**Dependencias**
- Ninguna nueva. `torch`, `transformers` y `datasets` ya están en `requirements.txt`
  y verificados en el entorno de `DEC-002`.

**Coste**
- Primera ejecución descarga ~15 GiB de pesos de Qwen 2.5 7B. El disco ext4 de WSL está
  al 94 % (64 GiB libres); cabe, pero es la restricción a vigilar.
- Inferencia: 3680 ítems cortos en bf16 sobre la RTX 4090, minutos.

**Riesgo funcional**
- Un resultado negativo (`s_amb < 0.20`) **no** es un fallo del change: es la respuesta
  que la puerta existe para dar, y ahorra todo el trabajo posterior.
