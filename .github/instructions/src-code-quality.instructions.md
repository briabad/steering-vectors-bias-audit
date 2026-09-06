---
description: "Calidad de código para el código de aplicación en src/ y sus tests."
applyTo: "src/**,tests/**"
---

# Calidad de código (src/ y tests/)

Aplica al escribir o modificar código de aplicación. Solo el `implementer` edita
aquí, y siempre contra un change de OpenSpec validado. Ver `docs/architecture.md`,
`docs/conventions.md` y `docs/testing.md` para el detalle portable.

## Reglas mínimas (checkpoint C5)

- Type hints en Python (o tipado estricto en TS). Nada sin tipar.
- Sigue las capas y la estructura de `docs/conventions.md`; sin dependencias circulares.
- Cambios mínimos y enfocados a la tarea de `tasks.md`; sin refactors oportunistas.
- DRY: no dupliques lógica; enlaza/reutiliza.
- Sin secretos ni credenciales hardcodeadas.
- Mockea en la frontera del proveedor externo (LLM, bus de eventos); deja correr la
  lógica propia real (ver `docs/testing.md`).

## Antes de cerrar

```bash
make lint          # o el linter/formatter equivalente
make test          # o el runner de tests equivalente
python tests/validate_graph.py   # sanidad del arnés, exit 0
```

Cobertura objetivo > 70% en el código tocado (ideal > 80%).
