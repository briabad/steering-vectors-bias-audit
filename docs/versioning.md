# VERSIONING — Estrategia de versionado

> Define **cómo** versionamos el software y los artefactos, no **qué** versionamos.

---

## 1. Versionado semántico (SemVer)

```
MAJOR.MINOR.PATCH
```

- **MAJOR**: cambios incompatibles con versiones anteriores (API breaking changes).
- **MINOR**: nuevas funcionalidades, backwards compatible.
- **PATCH**: fixes, backwards compatible.

### 1.1. Reglas
- La versión se define en `pyproject.toml` (Python) o `package.json` (Node).
- Cada release tiene un tag Git: `v1.2.3`.
- El CHANGELOG.md se actualiza con cada release.

### 1.2. Pre-releases
- `v1.2.3-alpha.1`, `v1.2.3-beta.2`, `v1.2.3-rc.1`.
- Los pre-releases no se instalan por defecto (`pip install` los ignora).

---

## 2. Versionado de modelos de IA

### 2.1. Esquema
```
<model-name>-v<semver>+<git-short-sha>
# Ej: sentiment-v2.1.0+a3f4d2e
```

### 2.2. Metadatos obligatorios
Cada modelo versionado debe incluir:
- Git commit del código que lo entrenó.
- Configuración completa (YAML/JSON).
- Dataset usado (referencia, no el dataset completo).
- Métricas de evaluación.
- Dependencias (versión exacta de cada librería).

### 2.3. Registro
- MLflow, Weights & Biases, o directorio versionado en artefactos.
- Nunca sobreescribir un modelo versionado.
- Los modelos en producción se referencian por versión exacta.

---

## 3. Branching strategy

### 3.1. Git Flow simplificado
```
main          ●────●────●────●────●────●  (producción, siempre estable)
               \    \    \    \    feat/FEAT-001  ●────●                  (feature branch)
feat/FEAT-002       ●────●────●       (feature branch)
fix/FEAT-003               ●────●     (hotfix branch)
```

### 3.2. Reglas
- `main` siempre pasa todos los tests.
- Los PRs a `main` requieren review (humano o agente revisor).
- Los PRs deben estar actualizados con `main` antes de mergear.
- Squash merge para features, merge commit para releases.

### 3.3. Feature branches
- Nombre: `feat/FEAT-NNN-breve-descripcion`.
- Vida útil: mientras la feature está en desarrollo.
- Se borran tras mergear.

### 3.4. Hotfix branches
- Nombre: `fix/FEAT-NNN-breve-descripcion`.
- Se crean desde `main`, se mergean a `main` y a la rama de desarrollo activa.

---

## 4. Versionado de datasets

### 4.1. DVC (Data Version Control)
- Los datasets grandes se versionan con DVC, no con Git.
- Cada versión de dataset tiene un hash y un tag.
- El pipeline de entrenamiento referencia el dataset por versión exacta.

### 4.2. Metadatos
- Fuente, fecha de obtención, preprocesamiento aplicado.
- Licencia y restricciones de uso.
- Estadísticas descriptivas.

---

## 5. Versionado de experimentos

### 5.1. Esquema
Cada experimento (EXP-XXX) se vincula a:
- Código: commit Git.
- Config: archivo versionado.
- Dataset: versión DVC o referencia.
- Modelo: versión del modelo resultante.

### 5.2. Reproducibilidad
- Un experimento debe ser reproducible ejecutando un solo comando.
- El comando y sus dependencias se documentan en el manifest del experimento.

---

## 6. Changelog

### 6.1. Formato (Keep a Changelog)
```markdown
## [Unreleased]

## [1.2.0] - 2026-06-22
### Added
- Nueva feature de evaluación de modelos (FEAT-015).

### Changed
- Mejorada la eficiencia del pipeline de datos (FEAT-014).

### Fixed
- Bug en el cálculo de métricas F1 (FEAT-013).

### Deprecated
- API v1, será removida en v2.0.0.
```

### 6.2. Reglas
- Cada PR que añade funcionalidad debe actualizar el CHANGELOG.
- Las categorías son: Added, Changed, Deprecated, Removed, Fixed, Security.
- Las versiones se enlazan a los tags Git.

---

## 7. Releases

### 7.1. Proceso
1. Actualizar versión en `pyproject.toml`.
2. Actualizar CHANGELOG.md.
3. Crear PR de release.
4. Review y aprobación.
5. Mergear a `main`.
6. Crear tag Git: `git tag -a v1.2.0 -m "Release v1.2.0"`.
7. Push del tag: `git push origin v1.2.0`.
8. Crear release en GitHub/GitLab con notas del CHANGELOG.

### 7.2. Automatización
- GitHub Actions / GitLab CI puede automatizar el proceso.
- El CI crea el tag y la release tras mergear el PR de release.

---

## 8. Versionado de dependencias

### 8.1. Direct dependencies
- Versiones pinnadas en `pyproject.toml`.
- Lock file (`poetry.lock`, `uv.lock`, `package-lock.json`) commiteado.

### 8.2. Security
- `pip-audit` o `safety` en CI para detectar vulnerabilidades.
- Dependabot o Renovate para actualizaciones automáticas.

### 8.3. Compatibilidad
- Las dependencias de desarrollo (dev) separadas de las de producción.
- Las dependencias opcionales documentadas con extras.
