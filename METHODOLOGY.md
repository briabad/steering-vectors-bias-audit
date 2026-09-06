# METHODOLOGY — Bases de trabajo del workspace

> Este archivo define **cómo trabajamos**, no qué construimos. Es la capa más
> estable del proyecto: cambia rara vez y se copia tal cual a otros proyectos.
> El objetivo último del workspace es **generar software de calidad con trazabilidad**,
> usando conocimiento estructurado como soporte. Estas son las reglas que lo hacen posible.

---

## 1. El modelo de cuatro capas (separadas por portabilidad)

El workspace se organiza por **cuánto cambia cada cosa al cambiar de proyecto**, no por "código vs. documentación".

| Capa | Qué es | Cambia cuando... | Archivos |
|------|--------|------------------|----------|
| **1. Harness portable** | *Cómo* trabajamos: arquitectura, convenciones, roles, criterios de cierre, calidad | Casi nunca. Se reutiliza entre proyectos | METHODOLOGY.md, AGENTS.md, CHECKPOINTS.md, docs/architecture.md, docs/conventions.md, docs/testing.md, docs/versioning.md, .claude/agents/ |
| **2. Core del proyecto** | *Qué* es este proyecto: spec, reglas, módulos reales, dependencias | Se **reemplaza entero** al cambiar de proyecto | data/00_context/, código del agente, pyproject.toml, requirements.txt |
| **3. Capa de conocimiento** | *Qué aprendemos y por qué*: experimentos, decisiones, estado del arte | **Crece continuamente** dentro del proyecto | data/ (Q / EXP / DEC / SOA) |
| **4. Capa de trabajo** | *Qué tareas se ejecutan y qué tocaron*: features / historias con trazabilidad | Continuo; cruza con la capa 3 | work/feature_list.json, openspec/changes/ |

**Regla de oro:** la capa 1 nunca menciona el proyecto concreto. No dice "Orbit Wars" ni "torch". Cuando una política depende del proyecto (p. ej. "¿se permiten dependencias externas?"), vive en la capa 2, no en architecture.md. Ver §5.

---

## 2. Dos principios rectores

### Principio A — Economía de contexto por divulgación progresiva

Nada se lee entero por defecto. La información se estructura en **tres niveles** para no saturar la ventana de contexto:

```
Nivel 0   data/index.md      ← SIEMPRE en contexto. El mapa: IDs + hook + estado.
Nivel 1   <artefacto>/manifest    ← pequeño. Metadata + enlaces tipados + qué cargar.
Nivel 2   hypothesis.md, results… ← se cargan BAJO DEMANDA, solo si el manifest dice que aplican.
```

Un agente lee el índice, sigue un enlace, abre el manifest y *desde ahí* decide qué cuerpos cargar. Nunca arrastra el Nivel 2 completo.

### Principio B — Trazabilidad como grafo de enlaces tipados

Toda decisión futura debe poder rastrearse hasta los resultados que la sustentan. Cada artefacto tiene un **ID estable** y **relaciones tipadas**:

```
Q-001  (pregunta / paradigma)
  └─ motiva → EXP-001 (experimento)
       ├─ usa → DEC-002 (decisión metodológica)
       ├─ apoyado_en → SOA-003 (estado del arte)
       └─ produce → results → confirma | refuta → Q-001

DEC-005 supersede → DEC-002
```

Los enlaces son **manuales pero validables**: ningún ID referenciado debe faltar, ningún artefacto debe quedar huérfano. (El validador automático llega en una fase posterior; por ahora la disciplina es humana.)

---

## 3. Los tipos de artefacto

| ID | Tipo | Carpeta | Para qué |
|----|------|---------|----------|
| Q-XXX | Pregunta / paradigma | data/questions/ | La pregunta raíz que guía una línea de trabajo |
| EXP-XXX | Experimento | data/experiments/ | La **unidad atómica** del conocimiento |
| DEC-XXX | Decisión | data/decisions/ | Una elección metodológica con su porqué |
| SOA-XXX | Estado del arte | data/references/ | Conocimiento destilado externo (no se repite, se enlaza) |

TOT-XXX (train of thought refinado) se añade **solo si** una decisión necesita una justificación larga que no cabe en el campo rationale. No se crea por defecto.

---

## 4. El ciclo de desarrollo (workflow)

### 4.1. Ciclo de investigación (cuando aplica)

1. **Plantear** — una Q-XXX enmarca la pregunta. Apóyala en SOA-XXX si aplica.
2. **Pre-registrar** — antes de correr nada, EXP-XXX/hypothesis.md declara la hipótesis **y un criterio de éxito cuantitativo**. Esto separa *investigar* de *trastear*: la predicción se escribe antes de ver el resultado (anti-HARKing).
3. **Ejecutar** — el código (cuando exista) vive en `src/`, no en `data/`. Los crudos del experimento van a EXP-XXX/output/.
4. **Documentar** — results.md contrasta el resultado contra la predicción registrada. Concluye: ¿confirma o refuta la Q?
5. **Consolidar (cierre de sesión)** — se crean/actualizan los DEC y SOA que surgieron, se actualiza index.md, y se descarta el buffer crudo de la sesión. La conversación viva es memoria volátil (STM); solo la síntesis se persiste (LTM).

### 4.2. Ciclo de desarrollo de software

1. **Planificar** — una feature FEAT-XXX define el scope, acceptance criteria y trazabilidad en work/feature_list.json.
2. **Especificar** — para features que añaden/modifican comportamiento del sistema, crear un change en `openspec/changes/FEAT-XXX/` con:
   - `proposal.md`: por qué, alcance, impacto (vincula a DEC/Q si aplica).
   - `specs/<capability>/spec.md`: delta de requisitos (ADDED/MODIFIED/REMOVED) con escenarios GIVEN/WHEN/THEN.
   - `tasks.md`: checklist de implementación atómica.
   - `design.md` (optativo): decisiones técnicas menores que no justifican DEC-XXX.
3. **Diseñar** — antes de tocar código, documentar decisiones arquitectónicas en DEC-XXX si son significativas, o en `design.md` si son técnicas menores.
4. **Implementar** — el código vive en `src/`, los tests en `tests/`. El agente lee `tasks.md` y ejecuta paso a paso, marcando tareas completadas.
5. **Verificar** — ejecutar el test suite completo. Los tests unitarios, de integración, de contrato (si aplica OpenAPI) y de sanidad del workspace deben pasar.
6. **Revisar** — validar contra acceptance criteria de la feature, specs delta, y checkpoints C2–C6.
7. **Consolidar** — actualizar feature_list.json (status: done), archivar el change de OpenSpec (merge specs delta en source of truth), documentar decisiones en DEC si surgieron, y actualizar el índice de conocimiento.

---

## 5. Encaje con el harness existente

- **docs/architecture.md** (capa 1): principios arquitectónicos portable. Define el patrón de capas horizontales (`application`/`domain`/`repository`/`instrumentation`) para un backend de un único dominio orquestado con agentes LLM, inversión de dependencias, y cuándo migrar a un modelo multi-dominio (`shared/`+`domains/`, ver §9 de ese documento).
- **docs/conventions.md** (capa 1): estilo de código portable. Define reglas de ubicación de código dentro de esas capas, convenciones de Python, de orquestación agéntica (prompts, agentes, grafos) y de Git. Ver `docs/estructura_archivos.md` para el árbol de referencia completo.
- **docs/testing.md** (capa 1): estrategia de testing portable. Define pirámide de testing, y sobre todo el principio de **mockear en la frontera del proveedor externo** (LLM, bus de eventos) dejando correr la lógica propia real — ver §2-5 de ese documento.
- **docs/versioning.md** (capa 1): estrategia de versionado portable. Define SemVer, versionado de prompts como contrato funcional (no hay modelos entrenados propios que versionar), branching, changelog.
- **OpenSpec** (capa 4): framework de Spec-Driven Development que estructura cada feature en proposal + specs delta + tasks. Complementa `work/feature_list.json` añadiendo especificación de comportamiento. Ver openspec/changes/ para changes activos y openspec/specs/ para source of truth.
- **Roles de agente** (.claude/agents/): el leader (modelo fuerte) hace la **consolidación** —decide qué es conocimiento y mantiene el grafo—; el implementer (modelo ligero) ejecuta y vuelca crudos, pero no decide qué se persiste como conocimiento.
- **CHECKPOINTS.md**: evalúa la salud del harness de código **y** la salud de la base de conocimiento.

### Skills de OpenSpec disponibles

El agente tiene estas 5 skills de OpenSpec:

| Skill | Uso | Cuándo |
|-------|-----|--------|
| `openspec-explore` | Explorar ideas antes de proponer | Cuando no estás seguro |
| `openspec-propose` | Generar proposal + specs + tasks | Cuando ya sabes qué hacer |
| `openspec-apply-change` | Implementar plan aprobado | Después de validación humana |
| `openspec-sync-specs` | Actualizar specs sin archivar | Durante desarrollo |
| `openspec-archive-change` | Cerrar y archivar cambio | Al finalizar feature |

**Regla**: Nunca uses `apply-change` sin que un humano haya validado el `propose` primero.
---

## 6. La prueba de fuego

¿Puedes entender el estado del proyecto leyendo **solo** data/index.md y siguiendo enlaces? Si sí, el sistema funciona.

¿Puedes ejecutar `make test` (o equivalente) y obtener un reporte de sanidad del workspace + tests de aplicación? Si sí, el sistema de calidad funciona.

¿Puedes ejecutar `openspec list` y ver qué changes están activos con sus specs y tasks? Si sí, el sistema de especificación funciona.

Si llenar los artefactos se siente como papeleo, está mal diseñado: recórtalo.
