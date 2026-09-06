---
description: "Crea un change OpenSpec completo (proposal + design + specs delta + tasks) listo para validar. Delega en el agente proposer."
model: ['Claude Opus 4.8 (copilot)', 'GPT-5.6 Terra (copilot)', 'Claude Sonnet 5 (copilot)']
tools: [read, search, execute, agent]
---

# /opsx-propose

Genera una propuesta de change en un paso, con todos los artefactos necesarios para
implementar.

Delega en el agente **`proposer`** (modelo de razonamiento). El `proposer` sigue la
skill `openspec-propose` (`.claude/skills/openspec-propose/SKILL.md`):

1. `openspec new change "<nombre-kebab>"`.
2. `openspec status --change "<nombre>" --json` para el orden de artefactos.
3. Por cada artefacto listo: `openspec instructions <id> --change "<nombre>" --json`,
   **decide el contenido** según `template`/`instruction`, y **dicta la redacción al
   agente `scribe`** (ruta + contenido literal). No copies `context`/`rules` al archivo.
4. Vincula `FEAT-NNN`/`DEC-XXX`/`Q-XXX` por ID.
5. `openspec validate "<nombre>" --strict`.

Al terminar, recuerda que hace falta **aprobación humana** antes de `/opsx-apply`.

Feature a proponer: ${input}
