# CHECKPOINTS — Evaluación del estado sano

> En sistemas multi-agente no se evalúa el camino, se evalúa el destino. Estos son
> los checkpoints objetivos que un juez (humano o IA) puede usar para decidir si el
> proyecto está sano.

---

## C1 — El arnés está completo

- [ ] Existen las bases: METHODOLOGY.md, AGENTS.md, CHECKPOINTS.md, docs/architecture.md, docs/conventions.md.
- [ ] Existe la capa de conocimiento: data/index.md, data/README.md, data/00_context/, data/_templates/.
- [ ] Existe la capa de trabajo: work/feature_list.json (JSON válido) y work/README.md.
- [ ] Existe infraestructura de calidad: linter/formatter configurado, hooks de pre-commit, CI/CD definido.
- [ ] Existe el directorio de tests con al menos validate_graph.py (sanidad del grafo).

## C2 — El estado del trabajo es coherente

- [ ] Como mucho **una** feature en `in_progress` en feature_list.json.
- [ ] Toda feature `done` cumple sus acceptance criteria y tiene trace.implements relleno.
- [ ] No hay IDs FEAT-NNN reutilizados; next_id es mayor que todos los usados.
- [ ] Toda feature tiene trazabilidad: trace.motivated_by y/o trace.enables poblados cuando aplica.

## C3 — La base de conocimiento está sana

- [ ] Cada fila de data/index.md apunta a un artefacto que existe en disco, y cada artefacto en disco tiene su fila en el índice (sin huérfanos).
- [ ] Todo EXP con estado `done` tiene hypothesis.md (pre-registrado), results.md con resultados y una conclusión que dice si confirma/refuta su Q.
- [ ] Toda DEC tiene trigger, alternatives, rationale y prediction.
- [ ] Toda SOA tiene source, summary y relevance.

## C4 — La trazabilidad es real

- [ ] Ningún ID enlazado (Q/EXP/DEC/SOA/FEAT) referencia algo inexistente.
- [ ] Las features que nacen de o habilitan conocimiento tienen trace.motivated_by / trace.enables poblados.
- [ ] Ningún artefacto copia contenido de otro: se enlaza por ID.

> **C3 y C4 se auto-chequean** con `python tests/validate_graph.py` (exit 0 = sano).
> Usa `--strict` para que los nodos aislados también fallen.

## C5 — La calidad del código está garantizada

- [ ] Todo código en `src/` pasa el linter/formatter configurado (ruff, black, eslint, etc.).
- [ ] Todo código en `src/` tiene tests unitarios con cobertura > 70% (umbral mínimo; ideal > 80%).
- [ ] Los tests de integración pasan (si aplica).
- [ ] No hay código sin tipo (Python: type hints; TS: strict mode).
- [ ] No hay dependencias circulares detectadas por herramientas de análisis estático.
- [ ] No hay secretos o credenciales hardcodeadas (validado con herramientas como `git-secrets` o `truffleHog`).

## C6 — La sesión se cerró bien

- [ ] No quedan artefactos a medias ni hipótesis sin su experimento.
- [ ] data/index.md refleja lo creado/cerrado en la sesión.
- [ ] work/feature_list.json refleja el estado actual del trabajo.
- [ ] No hay archivos temporales sospechosos (*.tmp, __pycache__ fuera de ignore).
- [ ] El commit de cierre tiene mensaje descriptivo que referencia la feature trabajada.

---

## Tests: tres clases

### Sanidad del workspace (activa)
Viven en `tests/` y crecen con el proyecto, una por cada invariante que valga la pena proteger.
- Hoy: `validate_graph.py` — verifica la metodología y la estructura, no la aplicación.
- Añadir: `validate_code_quality.py` — verifica que el código cumple conventions.md.
- Añadir: `validate_dependencies.py` — verifica que no hay dependencias circulares ni vulnerabilidades conocidas.

### Tests unitarios de la aplicación (YA APLICAN — desde 2026-09-06 hay código en `src/`)
Viven en `tests/unit/` y prueban funciones/módulos individuales de `src/`.

### Tests de integración (diferidos hasta que haya código)
Viven en `tests/integration/` y prueban flujos completos con dependencias reales o test doubles.

> Hasta que exista código en `src/`, la verificación de una feature es el cumplimiento de sus acceptance criteria. Cuando el código lo justifique, añadir el checkpoint **C7** de cobertura de tests.

---

## Cómo usar este archivo

Un juez (humano o el agente revisor) recorre cada checkbox, marca [x] o [ ], y **rechaza el cierre de sesión** si quedan vacíos en C1–C6.

La evaluación debe ser **objetiva y repetible**: cada checkbox debe poder verificarse con un comando o una inspección determinista.
