# Flujo de trabajo con el sistema de agentes

> Guía breve de **cómo trabajar** con los agentes de `.github/`. El detalle de cada
> rol está en [.github/copilot-instructions.md](.github/copilot-instructions.md) y en
> `.github/agents/*.agent.md`. El *porqué* del workspace está en [METHODOLOGY.md](METHODOLOGY.md).

## Los 6 agentes

| Agente | Para qué | Modelo (fallback) |
| ------ | -------- | ----------------- |
| `Leader` | Orquesta, descompone, valida. **Rol por defecto.** No codifica. | Opus 4.8 · GPT-5.6-terra · Sonnet 5 |
| `Proposer` | Diseña el change OpenSpec; **dicta** la redacción a `Scribe`. | Opus 4.8 · GPT-5.6-terra · Sonnet 5 |
| `Implementer` | Implementa un change validado, tarea a tarea. | Sonnet 5 · Gemini 3.6 Flash |
| `Reviewer` | Verifica acceptance, checkpoints, lint/tests/validate. | Sonnet 5 · Gemini 3.6 Flash |
| `Explorer` | Investiga en solo-lectura; devuelve una referencia a archivo. | Haiku 4.5 · GPT-5 mini · MAI Code 1 Flash |
| `Scribe` | Escribe/formatea MD exactamente como se le dicta. | MAI Code 1 Flash · GPT-5 mini · Haiku |

**Idea de coste**: los modelos caros **razonan** (qué construir, cómo, trazabilidad);
los baratos **ejecutan trabajo mecánico** (buscar, redactar MD).

## El ciclo (una feature)

```text
Explorer(opc.) → Proposer → [APROBACIÓN HUMANA] → Implementer → Reviewer → archivar
   investiga       diseña          gate               codifica     verifica   (sync specs)
```

1. **Explora** (si el requisito es difuso): `/opsx-explore "idea"` o invoca `Explorer`.
2. **Propón**: `/opsx-propose "feature"` → `Proposer` crea `openspec/changes/<id>/`.
3. **Valida**: `openspec validate <id> --strict`.
4. **Aprueba** (humano): revisa proposal/specs/tasks antes de implementar.
5. **Implementa**: `/opsx-apply <id>` → `Implementer` ejecuta `tasks.md`.
6. **Verifica**: invoca `Reviewer` (acceptance + checkpoints C2–C6 + validate).
7. **Cierra**: `/opsx-sync <id>` y `/opsx-archive <id>`; actualiza `work/feature_list.json`.

## Uso diario

- **Empieza por `Leader`**: descríbele la tarea; él decide si explorar, proponer o
  responder. Para preguntas de lectura pura, responde él mismo sin subagentes.
- **Invoca un agente concreto** desde el selector de agentes del chat (icono de agente)
  o escribiendo `@` + nombre. Los prompts `/opsx-*` son atajos del flujo.
- **Delegación**: `Leader` y `Proposer` reparten el trabajo; el resto no encadena
  (evita bucles). `Proposer` dicta a `Scribe`; nadie llama a `Reviewer` salvo el `Leader`/tú.

## Reglas duras

- Una sola feature/change a la vez.
- No implementar sin un change OpenSpec **validado y aprobado por un humano**.
- `Leader`/`Proposer` no editan `src/`. `Leader` no marca features como `done`.
- Los subagentes escriben en archivos y devuelven **referencias**, no vuelcan contexto.

## Verlos en el IDE

Los agentes se descubren desde el `.github/` de la **carpeta raíz** del workspace.
Para que aparezcan estos 6 (y no solo los de la raíz anterior), abre
`CIAM_Real_Time/` como carpeta: **Archivo → Abrir carpeta →
`CIAM_Real_Time`**. Alternativas (multi-root / settings) en
[.github/copilot-instructions.md](.github/copilot-instructions.md) §6.
