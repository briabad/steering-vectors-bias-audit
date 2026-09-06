---
description: "Implementa las tareas de un change OpenSpec ya validado y aprobado. Delega en el agente implementer."
model: ['Claude Sonnet 5 (copilot)', 'Gemini 3.6 Flash (copilot)']
tools: [read, edit, search, execute, agent]
---

# /opsx-apply

Implementa las tareas de un change de OpenSpec **ya validado y aprobado por un humano**.

Delega en el agente **`implementer`** (modelo de desarrollo). El `implementer` sigue la
skill `openspec-apply-change` (`.claude/skills/openspec-apply-change/SKILL.md`):

1. `openspec status --change "<id>" --json` e `openspec instructions apply --change "<id>" --json`.
2. Lee **todos** los `contextFiles` (proposal, design, specs, tasks).
3. Implementa **tarea a tarea**, con cambios mínimos, marcando `- [ ]` → `- [x]`.
4. Se autoverifica contra los `acceptance`; rellena `trace.implements`.
5. Pausa ante ambigüedad o si aparece un fallo de diseño (no parchea el spec por su cuenta).

Al terminar, lanza `/opsx-` de verificación con el agente `reviewer` antes de archivar.

Change a implementar: ${input}
