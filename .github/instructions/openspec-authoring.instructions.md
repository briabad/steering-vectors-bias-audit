---
description: "Reglas para redactar artefactos OpenSpec (proposal, design, specs delta, tasks) dentro de openspec/changes."
applyTo: "openspec/changes/**/*.md"
---

# Autoría de artefactos OpenSpec

Aplica al escribir cualquier archivo dentro de `openspec/changes/`. El schema es
`spec-driven` (ver `openspec/config.yaml`).

## Principios

- Un change = una unidad de comportamiento. **Un cambio por sesión**.
- El contenido lo **decide** `proposer`; la redacción la puede **materializar** `scribe`.
- Enlaza por ID (`FEAT-NNN`, `DEC-XXX`, `Q-XXX`); **nunca copies** contexto de otros
  archivos.
- No incluyas los bloques `context`/`rules` que devuelve `openspec instructions`:
  son restricciones para el agente, no contenido del artefacto.

## Artefactos y su contenido

- **proposal.md** — *qué y por qué*: motivación, alcance, impacto. Vincula la
  `FEAT-NNN` de `work/feature_list.json` y los `DEC-XXX`/`Q-XXX` que lo motivan.
- **design.md** — *cómo*: decisiones técnicas menores. Las decisiones significativas
  van a un `DEC-XXX` en `data/decisions/`, no aquí.
- **specs/<capability>/spec.md** — delta de requisitos. Usa las secciones
  `## ADDED Requirements`, `## MODIFIED Requirements`, `## REMOVED Requirements`,
  `## RENAMED Requirements`. Cada requisito lleva `### Requirement: <nombre>` y al
  menos un `#### Scenario:` con pasos `- **WHEN** ...` / `- **THEN** ...`.
- **tasks.md** — checklist atómico `- [ ]`. Cada tarea implementable y verificable de
  forma aislada; se marca `- [x]` al completarse.

## Verificación

Todo change debe pasar antes de implementarse:

```bash
openspec validate <change-id> --strict
```
