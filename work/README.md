# work/ — Capa de trabajo

Guía operativa mínima para `feature_list.json`. Ver METHODOLOGY.md §1 (capa 4)
y AGENTS.md §4.

## Schema de una feature

```jsonc
{
  "id": "FEAT-NNN",
  "title": "...",
  "status": "pending | in_progress | blocked | deferred | done | dropped",
  "description": "...",
  "acceptance": ["criterio verificable 1", "..."],
  "trace": {
    "motivated_by": ["DEC-XXX", "Q-XXX"],   // por qué existe esta feature
    "implements": []                          // qué se tocó, se rellena al cerrar
  }
}
```

## Estados
- `pending` — lista para empezar.
- `in_progress` — alguien la está ejecutando.
- `blocked` — no se puede empezar porque falta una decisión o una dependencia. El campo
  `description` debe decir **qué** la bloquea.
- `deferred` — se *podría* hacer, pero queda fuera del alcance de la tanda actual por
  coste o por secuenciación. Distinto de `blocked`: no falta nada, se ha decidido no
  hacerla todavía.
- `done` / `dropped` — cerrada, cumpliendo o abandonando sus acceptance.

## Reglas
- `next_id` en el archivo indica el próximo ID libre; incrementar al crear una feature.
- Una feature solo pasa a `done` cuando **todos** sus `acceptance` se cumplen.
- El leader nunca marca `done` (regla dura de AGENTS.md) — lo hace el implementer
  tras autoverificarse, o el leader lo confirma tras revisar.
- Si la feature añade/modifica comportamiento del sistema, debe tener un
  change de OpenSpec vinculado en `openspec/changes/FEAT-NNN-<descripcion>/`.
