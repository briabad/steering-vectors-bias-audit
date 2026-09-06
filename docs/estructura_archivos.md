# ESTRUCTURA DE ARCHIVOS — Propuesta v2: Multi-dominio + API-first

> Esta propuesta organiza `src/` para soportar múltiples dominios/aplicaciones
> de IA, cada uno con su propia API, lógica de negocio y ciclo de vida,
> manteniendo código compartido en `shared/`.

---

## 1. Principios de organización

### 1.1. Un dominio = una aplicación autocontenida
Cada dominio (ej: "recomendación", "clasificación de imágenes", "NLP") tiene:
- Su propia lógica de negocio (entities, use cases).
- Su propia API (routers, controllers, schemas).
- Su propio pipeline de datos y entrenamiento.
- Su propio servicio de inferencia.

### 1.2. Shared = código transversal, no negocio
Lo que es genuinamente compartido entre dominios vive en `shared/`.
Lo que es específico de un dominio NO escapa de su carpeta.

### 1.3. API como contrato
La API de cada dominio es su interfaz pública. Todo lo demás es privado.
Los otros dominios no importan directamente de `internal/` de otro dominio.
Si necesitan comunicarse, lo hacen vía la API (HTTP, eventos, o interfaces explícitas).

### 1.4. Inversión de dependencias
- `shared/` no depende de ningún dominio.
- Los dominios pueden depender de `shared/`.
- Los dominios no dependen entre sí (o lo hacen vía interfaces en `shared/`).

---

## 2. Estructura propuesta

```
project-root/
│
├── src/
│   ├── __init__.py
│   │
│   ├── shared/                          # Código transversal (capa horizontal)
│   │   ├── __init__.py
│   │   ├── domain/                      # Building blocks compartidos
│   │   │   ├── __init__.py
│   │   │   ├── entity.py                # BaseEntity, ID generation
│   │   │   ├── value_object.py          # BaseValueObject, validation
│   │   │   ├── event.py                 # DomainEvent base
│   │   │   ├── repository.py            # Repository protocol (interface)
│   │   │   └── exception.py             # DomainException base
│   │   │
│   │   ├── infrastructure/              # Implementaciones compartidas
│   │   │   ├── __init__.py
│   │   │   ├── database/                # Conexiones, ORM, migrations
│   │   │   │   ├── __init__.py
│   │   │   │   ├── connection.py
│   │   │   │   └── migrations/
│   │   │   ├── cache/                   # Redis, memcached
│   │   │   ├── messaging/               # Kafka, RabbitMQ, event bus
│   │   │   ├── storage/                 # S3, GCS, local filesystem
│   │   │   └── observability/           # Logging, metrics, tracing
│   │   │       ├── __init__.py
│   │   │       ├── logger.py            # Logger estructurado configurado
│   │   │       ├── metrics.py           # Métricas (Prometheus, etc.)
│   │   │       └── tracer.py            # Distributed tracing
│   │   │
│   │   ├── application/                 # Casos de uso compartidos (raro)
│   │   │   └── __init__.py
│   │   │
│   │   ├── api/                         # Utilidades de API compartidas
│   │   │   ├── __init__.py
│   │   │   ├── dependencies.py          # Dependency injection container
│   │   │   ├── middleware/              # CORS, auth, rate limiting, logging
│   │   │   ├── exception_handlers.py    # Manejo global de excepciones
│   │   │   ├── pagination.py            # Esquemas de paginación
│   │   │   ├── response.py              # Response wrapper estándar
│   │   │   └── validators.py            # Validadores reutilizables
│   │   │
│   │   ├── ml/                          # Infraestructura de ML compartida
│   │   │   ├── __init__.py
│   │   │   ├── base_model.py            # Clase base para modelos de IA
│   │   │   ├── base_trainer.py          # Clase base para entrenamiento
│   │   │   ├── base_evaluator.py        # Clase base para evaluación
│   │   │   ├── registry.py              # Registro de modelos (MLflow wrapper)
│   │   │   ├── feature_store.py         # Feature store compartido
│   │   │   └── monitoring/              # Drift detection, data quality
│   │   │       ├── __init__.py
│   │   │       ├── drift_detector.py
│   │   │       └── data_quality.py
│   │   │
│   │   └── config/                      # Configuración global
│   │       ├── __init__.py
│   │       ├── settings.py              # Pydantic Settings con validación
│   │       └── environments/            # Overrides por entorno
│   │           ├── dev.yaml
│   │           ├── staging.yaml
│   │           └── prod.yaml
│   │
│   ├── domains/                         # Cada dominio es una aplicación
│   │   ├── __init__.py
│   │   │
│   │   ├── sentiment_analysis/          # Ejemplo: Dominio 1
│   │   │   ├── __init__.py
│   │   │   │
│   │   │   ├── domain/                  # Lógica de negocio pura
│   │   │   │   ├── __init__.py
│   │   │   │   ├── entities/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── text_sample.py   # Value objects / entities
│   │   │   │   │   └── prediction.py
│   │   │   │   ├── repositories/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── text_repository.py  # Protocol (interface)
│   │   │   │   ├── services/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── sentiment_scorer.py # Reglas de negocio puras
│   │   │   │   └── events/
│   │   │   │       ├── __init__.py
│   │   │   │       └── prediction_made.py
│   │   │   │
│   │   │   ├── application/             # Casos de uso (orquesta domain + infra)
│   │   │   │   ├── __init__.py
│   │   │   │   ├── commands/            # Write operations (CQRS opcional)
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── train_model.py
│   │   │   │   ├── queries/             # Read operations
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── get_prediction.py
│   │   │   │   └── use_cases/
│   │   │   │       ├── __init__.py
│   │   │   │       └── analyze_sentiment.py
│   │   │   │
│   │   │   ├── infrastructure/          # Implementaciones concretas
│   │   │   │   ├── __init__.py
│   │   │   │   ├── persistence/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── sql_text_repository.py  # Implementa el protocol
│   │   │   │   ├── ml/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── model.py         # Modelo PyTorch/TF específico
│   │   │   │   │   ├── trainer.py       # Trainer específico
│   │   │   │   │   └── pipeline.py      # Pipeline de datos específico
│   │   │   │   └── external/
│   │   │   │       ├── __init__.py
│   │   │   │       └── some_api_client.py
│   │   │   │
│   │   │   ├── api/                     # Interfaz pública (FastAPI/Flask)
│   │   │   │   ├── __init__.py
│   │   │   │   ├── router.py            # Define rutas del dominio
│   │   │   │   ├── controller.py        # Recibe request, delega a use case
│   │   │   │   ├── schemas/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── request.py       # Pydantic request models
│   │   │   │   │   └── response.py      # Pydantic response models
│   │   │   │   └── dependencies.py      # DI específico del dominio
│   │   │   │
│   │   │   └── config.py                # Config específica del dominio
│   │   │
│   │   ├── image_classification/        # Ejemplo: Dominio 2
│   │   │   ├── domain/
│   │   │   ├── application/
│   │   │   ├── infrastructure/
│   │   │   ├── api/
│   │   │   └── config.py
│   │   │
│   │   └── recommendation/              # Ejemplo: Dominio 3
│   │       ├── domain/
│   │       ├── application/
│   │       ├── infrastructure/
│   │       ├── api/
│   │       └── config.py
│   │
│   ├── api_gateway/                     # Punto de entrada único (opcional)
│   │   ├── __init__.py
│   │   ├── main.py                      # FastAPI/Flask app principal
│   │   ├── routers.py                   # Incluye routers de cada dominio
│   │   ├── middleware/
│   │   │   ├── __init__.py
│   │   │   ├── authentication.py
│   │   │   ├── rate_limiting.py
│   │   │   └── request_logging.py
│   │   └── lifespan.py                  # Startup/shutdown events
│   │
│   └── config.py                        # Configuración raíz (importa shared.config)
│
├── tests/
│   ├── unit/
│   │   ├── shared/                      # Tests de código compartido
│   │   │   ├── test_domain_entity.py
│   │   │   └── test_infrastructure_logger.py
│   │   ├── domains/
│   │   │   ├── sentiment_analysis/      # Mirror de src/domains/sentiment_analysis/
│   │   │   │   ├── domain/
│   │   │   │   ├── application/
│   │   │   │   └── infrastructure/
│   │   │   └── image_classification/
│   │   └── api_gateway/
│   │
│   ├── integration/
│   │   ├── test_sentiment_api.py        # Test de API completa
│   │   ├── test_image_api.py
│   │   └── test_cross_domain.py         # Si dominios interactúan
│   │
│   ├── e2e/
│   │   └── test_full_pipeline.py
│   │
│   └── fixtures/
│       ├── datasets/
│       │   ├── sentiment_sample.csv
│       │   └── image_sample.png
│       └── models/
│           └── tiny_model.pt
│
├── scripts/
│   ├── train_sentiment.py               # Script operacional
│   ├── evaluate_sentiment.py
│   └── deploy_model.py
│
├── docs/
│   ├── architecture.md
│   ├── conventions.md
│   ├── testing.md
│   └── versioning.md
│
├── data/                                # Capa de conocimiento (METHODOLOGY.md)
├── work/                                # Capa de trabajo
├── .claude/                             # Agentes de IA
├── pyproject.toml
├── Makefile
└── README.md
```

---

## 3. Flujo de una request (ejemplo: análisis de sentimiento)

```
HTTP Request
    ↓
┌─────────────────────────────────────┐
│  api_gateway/main.py                │  ← Punto de entrada, middleware global
│  (CORS, auth, rate limiting)        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  domains/sentiment_analysis/api/    │
│  ├── router.py  →  /sentiment/     │  ← Define rutas
│  ├── controller.py                  │  ← Valida request, extrae params
│  └── schemas/request.py             │  ← Pydantic validation
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  domains/sentiment_analysis/        │
│  application/use_cases/           │
│  analyze_sentiment.py               │  ← Orquesta: llama a domain + infra
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  domains/sentiment_analysis/domain/ │
│  ├── services/sentiment_scorer.py   │  ← Lógica de negocio pura
│  └── entities/prediction.py         │  ← Value objects
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  domains/sentiment_analysis/        │
│  infrastructure/ml/model.py         │  ← Modelo PyTorch/TF
│  infrastructure/persistence/        │  ← Guarda resultado en DB
└─────────────────────────────────────┘
    ↓
HTTP Response (schemas/response.py)
```

---

## 4. Decisiones clave y por qué

### 4.1. ¿`shared/domain/` o cada dominio tiene su propio domain?

**Respuesta:** Ambos.
- `shared/domain/` tiene **building blocks** (Entity base, Event base, Repository protocol).
- Cada dominio extiende estos building blocks con sus entidades concretas.
- Las entidades de negocio específicas NO van a `shared/`.

### 4.2. ¿Cada dominio tiene su propia API?

**Respuesta:** Sí.
- Cada dominio expone su propio `router.py`.
- El `api_gateway` los monta todos en un único punto de entrada.
- Esto permite:
  - Desplegar dominios independientemente (microservicios futuro).
  - Testear APIs de dominio aisladamente.
  - Versionar APIs por dominio (`/v1/sentiment/`, `/v2/sentiment/`).

### 4.3. ¿Qué pasa si un dominio necesita algo de otro dominio?

**Respuesta:** Tres opciones, en orden de preferencia:

| Opción | Cuándo usar | Implementación |
|--------|-------------|----------------|
| **1. Eventos** | Comunicación asíncrona, acoplamiento débil | `shared/infrastructure/messaging/` + DomainEvent |
| **2. Interface en `shared/`** | Necesitas llamada síncrona | Protocol en `shared/domain/` + implementación en dominio B |
| **3. HTTP interno** | Dominios ya desplegados separadamente | Cliente en `infrastructure/external/` |

**Nunca:** Importar directamente de `domains/B/internal/` desde `domains/A/`.

### 4.4. ¿Dónde van los modelos de ML entrenados?

**Respuesta:**
- Código del modelo: `domains/X/infrastructure/ml/model.py`.
- Pesos/artefactos: en un registro externo (MLflow, S3) referenciado por versión.
- Nunca commitear pesos en Git.

### 4.5. ¿Dónde va el entrenamiento?

**Respuesta:**
- Clase base: `shared/ml/base_trainer.py`.
- Implementación específica: `domains/X/infrastructure/ml/trainer.py`.
- Script de entrenamiento: `scripts/train_X.py` (o `domains/X/application/commands/train_model.py`).

---

## 5. Comparativa: Monolito vs. Multi-repo

| Aspecto | Monolito (esta propuesta) | Multi-repo |
|---------|---------------------------|------------|
| Código compartido | Fácil (`shared/`) | Necesita package interno o copia |
| CI/CD | Un pipeline, más lento | Pipelines independientes, más rápido |
| Despliegue | Todo junto o por dominio | Por dominio nativamente |
| Contexto de agente | Un repo, más info | Menos contexto por repo |
| Equipo | Un equipo o varios en mismo repo | Un equipo por repo |
| **Recomendación** | **Start here** | Cuando un dominio crezca mucho o tenga equipo propio |

> **Regla:** empieza monolito modular. Extrae a repo independiente cuando un dominio tenga >5 desarrolladores o ciclo de release propio.

---

## 6. Anti-patrones a evitar

- ❌ `utils/` global con funciones de todo tipo → usa `shared/` con namespaces claros.
- ❌ Importar de `domains/B/` desde `domains/A/` → usa eventos o interfaces en `shared/`.
- ❌ Lógica de negocio en controllers → delega a use cases.
- ❌ Queries SQL en use cases → usa repositories con protocolos.
- ❌ Modelos de ML sin versionado → usa `shared/ml/registry.py`.
- ❌ Config esparcida → centraliza en `shared/config/` + `domains/X/config.py`.

---

## 7. Tests: mirror estructural

```
tests/
├── unit/
│   ├── shared/              ← Testea building blocks
│   └── domains/
│       ├── sentiment_analysis/
│       │   ├── domain/      ← Tests puros, sin mocks de I/O
│       │   ├── application/ ← Mocks de infraestructura
│       │   └── infrastructure/ ← Tests con DB/cache reales (testcontainers)
│       └── image_classification/
├── integration/
│   ├── test_sentiment_api.py    ← FastAPI TestClient
│   └── test_cross_domain.py     ← Eventos entre dominios
└── e2e/
    └── test_full_training_pipeline.py
```

**Regla:** cada test debe saber en qué capa está y qué puede usar.
- Unit domain → sin I/O, sin mocks (funciones puras).
- Unit application → mocks de infraestructura.
- Unit infrastructure → I/O real o testcontainers.
- Integration → API completa con DB real.
