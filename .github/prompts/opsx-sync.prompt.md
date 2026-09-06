---
description: "Sincroniza las specs delta de un change con las specs principales (source of truth) sin archivar."
model: ['Claude Sonnet 5 (copilot)', 'Gemini 3.6 Flash (copilot)']
tools: [read, edit, search, execute]
---

# /opsx-sync

Sincroniza las specs delta de un change hacia las specs principales en
`openspec/specs/<capability>/spec.md`, sin archivar el change. Es una operación
**dirigida por el agente**: lees el delta y editas la spec principal aplicando un
merge inteligente (p. ej. añadir un escenario sin copiar el requisito entero).

Sigue la skill `openspec-sync-specs` (`.claude/skills/openspec-sync-specs/SKILL.md`):

1. `openspec status --change "<nombre>" --json`; usa `artifactPaths.specs.existingOutputPaths`.
2. Por cada delta: aplica `ADDED`/`MODIFIED`/`REMOVED`/`RENAMED` a la spec principal,
   preservando el contenido no mencionado. Operación idempotente.
3. Resume qué capacidades se actualizaron y qué requisitos cambiaron.

El change sigue activo tras sincronizar; archívalo con `/opsx-archive` al cerrar.

Change a sincronizar: ${input}
