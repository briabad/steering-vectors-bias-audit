# CONVENTIONS — Estilo, nombres y estructura de código

> Base portable (capa 1). Define **cómo** escribimos código, no **qué** escribimos.
> Se adapta por proyecto en la capa 2 si hay conflictos (ej: "solo stdlib" vs. "usar torch").

---

## 1. Principios generales

### 1.1. Claridad sobre astucia
El código se lee más veces de las que se escribe. Prioriza legibilidad. Si necesitas un comentario para explicar qué hace una línea, refactoriza.

### 1.2. Explicito sobre implícito
No uses magia. No uses metaprogramación innecesaria. Cada comportamiento debe ser trazable a una línea de código legible.

### 1.3. Fallar rápido y ruidoso
Las precondiciones se validan con assertions o excepciones, no con silencios. Los errores deben ser informativos y contener contexto suficiente para debuggear.

### 1.4. Un solo nivel de abstracción por función
Si una función hace "alto nivel" y "bajo nivel", extrae el bajo nivel.

### 1.5. Un dominio no conoce a otro dominio
Los dominios en `src/domains/` son autocontenidos. Si necesitan comunicarse, lo hacen vía `shared/` (eventos, interfaces) o HTTP, nunca importando directamente.

---

## 2. Estructura de archivos (Multi-dominio)

```
project-root/
├── src/
│   ├── shared/                    # Código transversal (building blocks + infra)
│   │   ├── domain/                # Entity base, ValueObject base, Event base, Repository protocol
│   │   ├── infrastructure/          # DB, cache, messaging, storage, observability
│   │   ├── application/             # Casos de uso compartidos (raro)
│   │   ├── api/                     # Middleware, DI container, validators, pagination
│   │   ├── ml/                      # Base model, trainer, evaluator, registry, feature store
│   │   └── config/                  # Settings globales (Pydantic), environments/
│   │
│   ├── domains/                     # Cada dominio = una aplicación autocontenida
│   │   ├── <dominio_1>/
│   │   │   ├── domain/              # Entities, ValueObjects, Services, Events, Repositories (protocols)
│   │   │   ├── application/         # Use cases, Commands, Queries (orquesta domain + infra)
│   │   │   ├── infrastructure/      # Implementaciones concretas: persistence, ml models, external APIs
│   │   │   ├── api/                 # Router, Controller, Schemas (request/response), Dependencies
│   │   │   └── config.py            # Config específica del dominio
│   │   └── <dominio_2>/
│   │       └── ... (misma estructura)
│   │
│   ├── api_gateway/                 # Punto de entrada único (opcional, monta routers de dominios)
│   │   ├── main.py                  # FastAPI/Flask app principal
│   │   ├── routers.py               # Incluye routers de cada dominio
│   │   ├── middleware/              # Auth, rate limiting, logging global
│   │   └── lifespan.py              # Startup/shutdown events
│   │
│   └── config.py                    # Configuración raíz (importa shared.config)
│
├── tests/
│   ├── unit/
│   │   ├── shared/                  # Tests de building blocks
│   │   └── domains/
│   │       ├── <dominio_1>/       # Mirror exacto de src/domains/<dominio_1>/
│   │       │   ├── domain/          # Tests puros (sin I/O, sin mocks)
│   │       │   ├── application/     # Mocks de infraestructura
│   │       │   └── infrastructure/  # Tests con I/O real (testcontainers)
│   │       └── <dominio_2>/
│   ├── integration/                 # Tests de API completa (TestClient)
│   ├── e2e/                         # Tests end-to-end
│   └── fixtures/                    # Datos de prueba, mocks, stubs
│
├── docs/                            # Documentación del proyecto
├── data/                            # Capa de conocimiento (METHODOLOGY.md)
├── work/                            # Capa de trabajo (features, tareas)
├── scripts/                         # Scripts operacionales (no código de app)
├── .claude/                         # Configuración de agentes de IA
├── pyproject.toml                   # Dependencias, configuración de herramientas
├── Makefile                         # Comandos estándar (build, test, lint, format)
└── README.md                        # Punto de entrada humano
```

### 2.1. Reglas de ubicación

| ¿Qué? | ¿Dónde? | ¿Por qué? |
|-------|---------|-----------|
| Entidad de negocio pura | `domains/X/domain/entities/` | Sin dependencias externas |
| Protocolo de repositorio | `domains/X/domain/repositories/` | El dominio define el contrato |
| Implementación de repositorio | `domains/X/infrastructure/persistence/` | La infra cumple el contrato |
| Endpoint HTTP | `domains/X/api/router.py` | Interfaz pública del dominio |
| Validación de request | `domains/X/api/schemas/` | Pydantic models |
| Orquestación de caso de uso | `domains/X/application/use_cases/` | Coordina domain + infra |
| Modelo de ML específico | `domains/X/infrastructure/ml/` | Implementa `shared/ml/base_model.py` |
| Logger estructurado | `shared/infrastructure/observability/` | Transversal a todos los dominios |
| Config global | `shared/config/settings.py` | Un solo punto de verdad |
| Config de dominio | `domains/X/config.py` | Override o extensión de la global |

---

## 3. Convenciones de Python (predeterminado)

### 3.1. Estilo
- **PEP 8** como base.
- **Black** para formateo (line-length 88).
- **Ruff** para linting (reemplaza flake8, isort, pydocstyle).
- **mypy** para type checking (strict mode).

### 3.2. Type hints
- Todo código nuevo debe tener type hints.
- Usa `from __future__ import annotations` para compatibilidad.
- Evita `Any`; usa `TypeVar` o protocolos cuando necesites genericidad.

### 3.3. Docstrings
- Toda función pública tiene docstring en formato **Google style**.
- Incluye: descripción, Args, Returns, Raises (si aplica).

### 3.4. Nombres
- `snake_case` para funciones, variables, módulos.
- `PascalCase` para clases, enums, type aliases.
- `SCREAMING_SNAKE_CASE` para constantes.
- Prefijos descriptivos:
  - `is_`, `has_`, `should_` para booleanos.
  - `get_`, `set_`, `compute_`, `validate_` para funciones con efectos claros.

### 3.5. Manejo de errores
- Usa excepciones personalizadas que hereden de una base del proyecto.
- Nunca captures `Exception` genérico sin re-raise o logging adecuado.
- Los mensajes de error deben incluir contexto: qué se intentó, con qué datos, qué falló.

### 3.6. Dependencias
- Todas las dependencias van en `pyproject.toml`.
- No uses dependencias sin justificar en un DEC o en la documentación del proyecto.
- Versiones pinnadas para reproducibilidad (`poetry.lock` o `uv.lock`).

---

## 4. Convenciones de testing

### 4.1. Estructura de tests
- **Mirror exacto** de `src/`: si `src/domains/sentiment/domain/entities/text_sample.py`, entonces `tests/unit/domains/sentiment/domain/entities/test_text_sample.py`.
- Un archivo de test por archivo de código.
- Una función de test por comportamiento (no por función).

### 4.2. Nomenclatura de tests
```
test_<funcion>_<condicion>_<resultado_esperado>
# Ej: test_train_model_empty_dataset_raises_value_error
```

### 4.3. Fixtures
- Usa `pytest.fixture` para setup/teardown reutilizable.
- Los fixtures van en `tests/fixtures/` o en `conftest.py` por módulo.
- Fixtures por dominio: `tests/unit/domains/sentiment/conftest.py`.

### 4.4. Mocks
- Usa `unittest.mock` o `pytest-mock`.
- Nunca mocks lo que no controlas (APIs externas): usa test doubles o VCR.

### 4.5. Cobertura mínima
- **70%** como umbral mínimo (falla el CI si baja).
- **80%** como objetivo.
- La cobertura se mide con `pytest-cov`.

---

## 5. Convenciones de IA / ML

### 5.1. Reproducibilidad
- Seeds fijas para todo lo aleatorio (numpy, torch, random).
- Configuraciones versionadas (YAML/JSON) que se guardan con los artefactos.
- Determinismo documentado: qué es determinista y qué no.

### 5.2. Versionado de modelos
- Cada modelo entrenado tiene un ID único (hash de config + timestamp).
- Los artefactos se guardan en un registro (MLflow, W&B, o directorio versionado).
- Nunca sobreescribas un modelo entrenado.

### 5.3. Evaluación
- Métricas calculadas con funciones puras y testeadas.
- Comparación contra baseline siempre documentada.
- Resultados de evaluación persistidos como artefactos EXP-XXX.

### 5.4. Datos
- No commitees datasets crudos. Usa DVC o referencias externas.
- Los pipelines de datos deben ser reproducibles y versionados.

---

## 6. Convenciones de versionado (Git)

### 6.1. Branching
- `main`: siempre estable, tests pasando.
- `feat/FEAT-NNN-descripcion`: features en desarrollo.
- `fix/FEAT-NNN-descripcion`: fixes.
- `exp/EXP-NNN-descripcion`: experimentos (no mergean a main directamente).

### 6.2. Commits
- Formato: `<tipo>(<scope>): <descripcion>` (Conventional Commits).
- Tipos: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `exp`.
- El cuerpo explica **por qué**, no **qué** (eso está en el diff).
- Referencia la feature: `feat(training): implement early stopping FEAT-012`.

### 6.3. Pull Requests
- Título: `[FEAT-NNN] Descripción breve`.
- Descripción: qué cambia, por qué, cómo se valida.
- Checklist de CHECKPOINTS.md C5 aplicado.

---

## 7. Herramientas obligatorias

| Herramienta | Propósito | Configuración |
|-------------|-----------|---------------|
| Black | Formateo | `pyproject.toml` |
| Ruff | Linting + import sorting | `pyproject.toml` |
| mypy | Type checking | `pyproject.toml` (strict) |
| pytest | Testing | `pyproject.toml` |
| pytest-cov | Cobertura | Umbral 70% |
| pre-commit | Hooks pre-commit | `.pre-commit-config.yaml` |
| git-secrets | Prevención de secretos | `.gitallowed` |

---

## 8. Anti-patrones prohibidos

- ❌ Código sin tests.
- ❌ `print()` en código de producción (usa logging estructurado).
- ❌ Variables de entorno leídas en cualquier lado (centraliza en `shared/config/`).
- ❌ Dependencias circulares.
- ❌ Código muerto (funciones no usadas, imports no usados).
- ❌ Hardcodear paths, URLs, o credenciales.
- ❌ Ignorar warnings del linter (fix o `# noqa` con justificación).
- ❌ Mergear a `main` sin PR review (humano o agente revisor).
- ❌ Importar directamente de `domains/B/` desde `domains/A/` (usa `shared/` o eventos).
- ❌ Lógica de negocio en controllers (delega a use cases).
- ❌ Commitear pesos de modelos o datasets crudos en Git.
