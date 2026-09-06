---
description: "Archiva un change completado: verifica artefactos/tareas, sincroniza specs delta y mueve el change a archive."
model: ['Claude Sonnet 5 (copilot)', 'Gemini 3.6 Flash (copilot)']
tools: [read, edit, search, execute]
---

# /opsx-archive

Archiva un change **completado**, tras la verificación del `reviewer`.

Sigue la skill `openspec-archive-change` (`.claude/skills/openspec-archive-change/SKILL.md`):

1. `openspec status --change "<nombre>" --json`: confirma artefactos `done`.
2. Comprueba `tasks.md`: todas las tareas `- [x]` (avisa si quedan pendientes).
3. Evalúa las specs delta: si hay cambios sin sincronizar, sincroniza primero
   (`/opsx-sync`) para mergear en `openspec/specs/`.
4. Mueve el change a `archive/YYYY-MM-DD-<nombre>/` bajo el directorio de changes.
5. Actualiza `work/feature_list.json` (`status: "done"`, `trace.implements`) y, si
   surgieron decisiones/experimentos, consolida `data/` e `data/index.md` (delega la
   redacción en el agente `scribe`).

Muestra un resumen del archivo y si las specs se sincronizaron.

Change a archivar: ${input}
