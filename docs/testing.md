# TESTING — Estrategia de testing para proyectos de IA

> Define **cómo** probamos el software, no **qué** probamos. Se adapta en la capa 2.

---

## 1. Pirámide de testing

```
         ┌─────────┐
         │   E2E   │  ← Pocos, lentos, alto valor de confianza
         │ (slow)  │
        ┌┴─────────┴┐
        │ Integration │  ← Medianos, medianamente lentos
        │   (med)     │
       ┌┴─────────────┴┐
       │     Unit      │  ← Muchos, rápidos, baratos
       │    (fast)     │
      ┌┴───────────────┴┐
      │  Static Analysis│  ← Lint, type check, format (ms)
      │    (instant)    │
      └─────────────────┘
```

**Regla de oro:** la mayoría de los tests deben ser unitarios. Los E2E son la punta de la pirámide.

---

## 2. Tests unitarios

### 2.1. Qué probar
- Funciones puras (dominio): 100% de cobertura objetivo.
- Transformaciones de datos: entra X, sale Y.
- Reglas de negocio: casos límite, edge cases.
- Utilidades compartidas.

### 2.2. Qué NO probar
- Frameworks (no testees PyTorch, testea TU código que usa PyTorch).
- I/O (mock o test doubles).
- Código generado automáticamente.

### 2.3. Patrones
- **AAA**: Arrange → Act → Assert.
- **Given-When-Then** para tests de comportamiento.
- Un assert por test (ideal) o un tema coherente por test.

### 2.4. Fixtures y parametrización
```python
@pytest.mark.parametrize("input,expected", [
    ("hello", 5),
    ("world", 5),
    ("", 0),
])
def test_string_length(input, expected):
    assert len(input) == expected
```

---

## 3. Tests de integración

### 3.1. Qué probar
- Flujos que cruzan capas (API → Aplicación → Dominio → Infraestructura).
- Persistencia (DB real o testcontainer).
- Integraciones con servicios externos (con test doubles o VCR).

### 3.2. Base de datos
- Usa **testcontainers** para DB reales en tests.
- O usa **SQLite en memoria** para tests rápidos (si la lógica es agnóstica al motor).
- Cada test debe ser independiente: setup/teardown limpio.

### 3.3. APIs externas
- **VCR.py** para grabar/reproducir respuestas HTTP.
- **MockServer** o **WireMock** para tests más complejos.
- Nunca llames APIs reales en tests (lentos, frágiles, costosos).

---

## 4. Tests end-to-end (E2E)

### 4.1. Cuándo usar
- Flujos críticos del usuario (happy path).
- Regresiones que los tests unitarios no capturan.
- Validación de despliegue.

### 4.2. Herramientas por tipo de app

| Tipo de app | Herramienta recomendada |
|-------------|------------------------|
| API REST | pytest + requests / httpx |
| API con contratos | Pact (contract testing) |
| CLI | pytest + subprocess / click.testing |
| Web app | Playwright / Selenium |
| ML pipeline | pytest + pipeline runner completo |

### 4.3. Tests de pipelines de ML
- Ejecutar el pipeline completo con datos de prueba pequeños.
- Verificar que los artefactos se generan correctamente.
- Verificar que las métricas están en rangos esperados.

---

## 5. Testing de calidad de modelos de IA

### 5.1. Métricas de rendimiento
- Tests que verifican que las métricas (accuracy, F1, BLEU, etc.) superan umbrales.
- Baselines versionados: cada modelo nuevo debe superar o igualar el baseline.
- Regresión: si una métrica baja más del 5%, el test falla.

### 5.2. Tests de robustez
- Inputs adversariales (perturbaciones pequeñas).
- Inputs fuera de distribución (OOD).
- Inputs malformados (deben manejarse gracefulmente).

### 5.3. Tests de fairness y sesgo
- Verificar paridad entre grupos demográficos.
- Verificar que el modelo no reproduce sesgos del dataset.

### 5.4. Evaluación con LLM-as-Judge
- Para tareas generativas, usar un LLM como juez con prompts estructurados.
- Verificar consistencia del juez (múltiples runs, varianza baja).
- Documentar el prompt del juez como parte del test.

---

## 6. Tests de sanidad del workspace

### 6.1. validate_graph.py
- Verifica que el grafo de conocimiento (Q/EXP/DEC/SOA/FEAT) es consistente.
- Sin huérfanos, sin ciclos, sin IDs duplicados.
- Corre en CI en cada push.

### 6.2. validate_code_quality.py
- Verifica que el código cumple conventions.md.
- Verifica que no hay imports no usados, funciones no documentadas, etc.

### 6.3. validate_dependencies.py
- Verifica que no hay dependencias circulares.
- Verifica que no hay vulnerabilidades conocidas (safety, pip-audit).

---

## 7. Cobertura

### 7.1. Umbral
- **70%** mínimo (falla CI).
- **80%** objetivo.
- **90%+** para código crítico (dominio, inferencia).

### 7.2. Qué NO cuenta
- Código de tests.
- Código de infraestructura pura (a veces).
- Código generado.

### 7.3. Reporte
- Generar reporte HTML de cobertura en CI.
- Publicar badge en README.

---

## 8. CI/CD y testing

### 8.1. Pipeline de CI
```
lint → type-check → unit-tests → integration-tests → e2e-tests (opcional)
  ↓         ↓            ↓              ↓                ↓
 fail     fail         fail           fail             warn
```

### 8.2. Pre-commit hooks
- Black, Ruff, mypy corren en pre-commit.
- Tests unitarios rápidos (< 30s) pueden correr en pre-commit.

### 8.3. Nightly builds
- Tests E2E lentos.
- Evaluación de modelos con datasets completos.
- Benchmarks de rendimiento.

---

## 9. Anti-patrones de testing

- ❌ Tests que dependen del orden de ejecución.
- ❌ Tests que modifican estado global.
- ❌ Tests que acceden a la red sin mocking.
- ❌ Tests con sleeps (usar polling o eventos).
- ❌ Tests que verifican implementación en lugar de comportamiento.
- ❌ "Test para tener cobertura" sin valor de verificación real.
