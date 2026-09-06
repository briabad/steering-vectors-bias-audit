# AGENTS.md — Mapa de navegación para agentes de IA

> Punto de entrada para cualquier agente que trabaje en este repositorio. NO es
> una biblia de reglas: es un **mapa**. Lee solo lo que necesites cuando lo
> necesites (divulgación progresiva). El *porqué* del sistema está en
> [METHODOLOGY.md](METHODOLOGY.md).

> Este workspace es de **desarrollo de software de inteligencia artificial** con
> trazabilidad de conocimiento. El fin es producir código de calidad; el
> conocimiento estructurado es el soporte.

---

## 1. Antes de empezar (obligatorio)

1. Lee [METHODOLOGY.md](METHODOLOGY.md) — el modelo de capas y los dos principios.
2. Lee [data/00_context/](data/00_context/README.md) — qué es este proyecto.
3. Lee [data/index.md](data/index.md) — el estado del conocimiento.
4. Lee [work/feature_list.json](work/feature_list.json) — y elige **una** tarea con estado pending.
5. Lee [docs/architecture.md](docs/architecture.md) — principios arquitectónicos del proyecto.
6. Lee [docs/conventions.md](docs/conventions.md) — estilo, nombres, estructura de código.

No trabajes en más de una feature a la vez.

---

## 2. Mapa del repositorio

| Archivo / carpeta | Qué contiene | Cuándo leerlo |
|-------------------|--------------|---------------|
| METHODOLOGY.md | Cómo trabajamos: capas, trazabilidad, ciclo de investigación y desarrollo | Siempre, al empezar |
| data/00_context/ | El proyecto principal: spec del dominio y flujo operativo | Para entender el dominio |
| data/index.md | Mapa Nivel 0 del conocimiento (Q / EXP / DEC / SOA) | Siempre, al empezar |
| data/README.md | Guía operativa: IDs, schemas, reglas de la capa de conocimiento | Al crear un artefacto |
| data/_templates/ | Plantillas para copiar (question, experiment, decision, ...) | Al crear un artefacto |
| work/feature_list.json | Tareas/features con estado y trazabilidad | Siempre, al empezar |
| work/README.md | Schema de features e historias; cómo dar trazabilidad | Antes de crear/cerrar una tarea |
| openspec/project.md | Contexto global del proyecto para OpenSpec (stack, reglas) | Si usas OpenSpec |
| openspec/specs/ | Source of truth del sistema (capacidades actuales) | Antes de modificar comportamiento existente |
| openspec/changes/ | Changes activos (proposal, specs delta, tasks, design) | Siempre que trabajes en una feature |
| docs/architecture.md | Base portable: qué significa buena arquitectura | Antes de implementar |
| docs/conventions.md | Base portable: estilo, nombres, estructura de código | Antes de escribir código |
| docs/testing.md | Estrategia de testing: unitarios, integración, E2E, calidad de modelos | Antes de escribir tests |
| docs/versioning.md | Estrategia de versionado: semver, branching, releases | Antes de versionar |
| CHECKPOINTS.md | Criterios objetivos de "estado sano" para un juez | Para auto-evaluarte |
| .claude/agents/ | Definiciones de subagentes (líder, implementador, revisor) | Si orquestas trabajo |
| tests/ | Tests de sanidad del workspace (validate_graph.py, ...) y tests de aplicación | Antes de cerrar sesión |
| src/ | Código del proyecto | Para implementar |
| openapi/ | Especificaciones OpenAPI (spec-first, cuando aplica) | Si la feature expone API pública |

---

## 3. Reglas duras (no negociables)

### 3.1. Una sola tarea a la vez
No mezcles cambios de varias features en una sesión.

### 3.2. Documenta mientras trabajas
Actualiza trace.implements de la feature y el index.md del conocimiento conforme avanzas.

### 3.3. Enlaza por ID, nunca copies contexto
Entre archivos. Si necesitas referenciar algo, usa su ID estable.

### 3.4. Verificación obligatoria

**Verificar es ejecutar y leer la salida. No es razonar que debería funcionar.**

Una afirmación de que algo está hecho debe poder respaldarse con una salida de comando
o un fichero en disco. En concreto, no cuentan como verificación: que el código "esté
en su sitio", que la arquitectura sea correcta, o que los tests "estén listos". Si vas
a decir que los tests pasan, corre `pytest` y pega la línea de resumen.

Y al revés, para quien coordina: **la plausibilidad de un informe no es evidencia**.
Un informe bien estructurado que afirma haber completado tareas se comprueba mirando el
disco — `grep -c '\[x\]'` sobre `tasks.md`, la existencia del fichero de salida, y la
suite de tests ejecutada de nuevo.

Una tarea se cierra cuando cumple **todos** sus acceptance criteria. Antes de cerrar sesión:
- Corre `python tests/validate_graph.py` (sanidad del grafo).
- Corre el linter/formatter (`make lint` o equivalente).
- Corre los tests unitarios (`make test` o equivalente).
- Verifica cobertura de tests si aplica.
- Si la feature usa OpenSpec: `openspec validate <change-id> --strict`.
- Si la feature expone una API con spec-first: `make validate-spec` (oasdiff).

### 3.5. No inventes
Si no sabes algo, búscalo en data/00_context/ o docs/ antes de inventarlo.

### 3.6. Calidad mínima del código
Todo código mergeado a `main` debe cumplir:
- [ ] Pasar linter y formatter sin warnings (ver docs/conventions.md §3.1).
- [ ] Tener tests unitarios con cobertura > 70% (ver docs/testing.md §4.5).
- [ ] Tener type hints (Python) o tipado estricto (TypeScript) (ver docs/conventions.md §3.2).
- [ ] No tener dependencias circulares (ver docs/architecture.md §8.2).
- [ ] No tener código duplicado (DRY).
- [ ] Seguir las convenciones de naming de docs/conventions.md §3.4.
- [ ] Seguir la estructura de archivos multi-dominio de docs/conventions.md §2.
- [ ] Si usa OpenSpec: specs delta validadas y tasks completadas.
- [ ] Si expone API pública: spec OpenAPI en `openapi/` y validación estructural (oasdiff).

---

## 4. Cómo elegir o crear una tarea

**Elegir:**
```
feature_list.json → status == "pending" → menor id → status = "in_progress"
```

**Crear:**
```
toma next_id → FEAT-NNN → escribe title, acceptance y trace ANTES de empezar.
```
(Schema completo en work/README.md)

**Si la feature añade/modifica comportamiento del sistema:**
```
# Crear change de OpenSpec vinculado a FEAT-NNN
mkdir openspec/changes/FEAT-NNN-<descripcion>
# Crear proposal.md, specs/<capability>/spec.md (delta), tasks.md
# (design.md optativo)
```

---

## 5. Cierre de sesión

Antes de terminar:

1. **La tarea cumple sus acceptance** → status: "done" (o "dropped" si se abandona).
2. **Rellena trace.implements** con lo que se tocó.
3. **Ejecuta calidad:**
   - `make lint` (o equivalente) → debe pasar sin errores.
   - `make test` (o equivalente) → todos los tests deben pasar.
   - `python tests/validate_graph.py` → exit 0.
   - `openspec validate <change-id> --strict` (si aplica OpenSpec).
   - `make validate-spec` (si aplica OpenAPI spec-first).
4. **Archivar change de OpenSpec** (si aplica):
   - `openspec archive <change-id>` → specs delta se mergean en `openspec/specs/`.
5. **Consolida el conocimiento:** si surgieron decisiones, experimentos o estado del arte, crea/actualiza sus artefactos y la fila correspondiente en data/index.md.
6. **Commit:** mensaje descriptivo que referencia FEAT-NNN (ver docs/conventions.md §6.2).
7. No dejes artefactos a medias ni IDs huérfanos.

---

## 6. Si te bloqueas

- Relee la sección relevante de data/00_context/ o docs/.
- Si una herramienta no hace lo que esperas, **no inventes un workaround**: documenta el bloqueo en la feature (campo libre / status sin avanzar) y para.
- Si el bloqueo es arquitectónico, crea un DEC-XXX para documentar la decisión pendiente.
