# data/ — Capa de conocimiento

Guía operativa mínima para la capa de conocimiento (Q/EXP/DEC/SOA). Ver
`METHODOLOGY.md §2-3` para los principios y tipos completos.

## Estructura

```
data/
├── 00_context/         # Nivel 0 de dominio: qué es este proyecto (ver README.md)
├── index.md             # Nivel 0 de conocimiento: mapa completo Q/EXP/DEC/SOA
├── questions/           # Q-XXX
│   └── Q-001_<slug>.md
├── experiments/         # EXP-XXX
│   └── EXP-001_<slug>/
│       ├── manifest.md      # metadata + enlaces tipados + qué cargar (Nivel 1)
│       ├── hypothesis.md    # pre-registrado ANTES de ejecutar (Nivel 2)
│       ├── results.md       # contraste contra la hipótesis (Nivel 2)
│       └── output/          # crudos del experimento
├── decisions/            # DEC-XXX
│   └── DEC-001_<slug>.md
└── references/            # SOA-XXX
    └── SOA-001_<slug>.md
```

## IDs

- Formato `<TIPO>-NNN` con `NNN` de 3 dígitos, secuencial dentro de su tipo,
  nunca reutilizado.
- El ID es estable de por vida del artefacto. No renombres IDs existentes.

## Reglas

- **Enlaza, no copies.** Si un artefacto necesita el contenido de otro, lo
  referencia por ID (`[[DEC-002]]` o `Ver DEC-002`), no lo pega.
- **Nivel 0 (`index.md`) siempre en contexto.** Nivel 1 (manifest) se abre
  para decidir. Nivel 2 (hypothesis/results, cuerpo completo de una
  decisión) solo bajo demanda.
- **Toda `EXP` con estado `done` necesita `hypothesis.md` pre-registrado y
  `results.md` con conclusión explícita** (confirma/refuta la `Q`). Sin
  hipótesis previa no hay experimento, hay exploración — está bien, pero se
  documenta como tal, no como `EXP`.
- **Toda `DEC` necesita `trigger`, `alternatives`, `rationale` y
  `prediction`** — qué disparó la decisión, qué otras opciones se
  consideraron, por qué se eligió esta, y qué predice que pasará si es
  correcta (permite revisar después si la predicción se cumplió).
- **Toda `SOA` necesita `source`, `summary` y `relevance`** — de dónde viene
  el conocimiento externo, qué dice, y por qué importa para este proyecto.
- Verificación automática de integridad del grafo:
  `python tests/validate_graph.py` (exit 0 = sano; `--strict` para que los
  nodos aislados también fallen).

## Cuándo crear cada tipo en este proyecto

Este es un backend de producción, no un workspace de investigación pura —
en la práctica, la mayoría de artefactos serán `DEC` (decisiones
arquitectónicas: por qué el merge incremental se resuelve así, por qué el
grafo de escalas usa un checkpointer independiente del de Tier 1) y `SOA`
(conocimiento externo relevante: limitaciones descubiertas de Pipecat, del
STT de Google, comportamiento del protocolo RTVI). `Q`/`EXP` aplican cuando
hay una pregunta genuina con incertidumbre a resolver empíricamente (p. ej.
"¿qué configuración de VAD reduce mejor los falsos cortes de habla?").
