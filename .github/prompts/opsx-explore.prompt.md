---
description: "Explora una idea o problema antes de proponer un change (modo pensamiento, no implementación)."
model: ['Claude Opus 4.8 (copilot)', 'GPT-5.6 Terra (copilot)', 'Claude Sonnet 5 (copilot)']
tools: [read, search, execute, agent]
---

# /opsx-explore

Entra en **modo exploración** para pensar una idea, investigar un problema o clarificar
requisitos **antes** de proponer un change. Piensa en profundidad, usa diagramas ASCII
y sigue la conversación. **No implementes**: puedes leer código y crear artefactos
OpenSpec si te lo piden, pero nunca escribes código de aplicación.

Sigue la skill `openspec-explore` (en `.claude/skills/openspec-explore/SKILL.md`) para
la mecánica. Comienza comprobando el contexto:

```bash
openspec list --json
```

Para investigación de código acotada y barata, delega en el agente **`explorer`** con
una pregunta concreta y pídele que devuelva una **referencia a archivo**, no el volcado.

Cuando la idea cristalice, ofrece pasar a `/opsx-propose` (no fuerces la formalización).

Idea a explorar: ${input}
