# Índice de conocimiento (Nivel 0)

> Mapa completo de la capa de conocimiento. Se lee siempre, entero, al
> empezar una sesión (ver `METHODOLOGY.md §2 Principio A`). Cada fila enlaza
> a un artefacto en disco; ningún artefacto en disco debe faltar aquí
> (huérfano) y ninguna fila debe apuntar a un artefacto inexistente —
> verificable con `python tests/validate_graph.py`.

## Preguntas (Q)

| ID | Hook (una frase) | Estado | Artefacto |
|----|-------------------|--------|-----------|
| Q-001 | ¿Puede una dirección latente detectar y moderar razonamientos sesgados sin tocar los pesos del modelo? | open | [questions/Q-001_hateval_vector_flow.md](questions/Q-001_hateval_vector_flow.md) |

## Experimentos (EXP)

| ID | Hook (una frase) | Estado | Q relacionada | Artefacto |
|----|-------------------|--------|----------------|-----------|
| EXP-001 | Buscar una dirección latente discriminativa y métricas geométricas de flujo que separen odio de neutral en HatEval con Qwen 2.5 7B. | superseded_by EXP-002/003 | Q-001 | [experiments/EXP-001_hateval_vector_flow/manifest.md](experiments/EXP-001_hateval_vector_flow/manifest.md) |
| EXP-002 | Puerta de existencia superada: 1063 ítems de contraste conductual en BBQ; el sesgo direccional se cancela entre ejes y el alineamiento cubre unos 60× más que otros. | puerta hecha; vector pendiente | Q-001 | [experiments/EXP-002_bbq_stereotype_direction/manifest.md](experiments/EXP-002_bbq_stereotype_direction/manifest.md) |
| EXP-003 | PoC: diccionario de 5 conceptos de daño sin SAE + reglas lógicas que reduzcan el sobre-bloqueo del guardarraíl. | planned | Q-001 | [experiments/EXP-003_concept_dictionary_guardrail/manifest.md](experiments/EXP-003_concept_dictionary_guardrail/manifest.md) |

## Decisiones (DEC)

| ID | Hook (una frase) | Supersede | Artefacto |
|----|-------------------|-----------|-----------|
| DEC-001 | Qwen 2.5 7B Instruct + HatEval como base del primer experimento, sin modificar pesos. | — | [decisions/DEC-001_model_dataset_choice.md](decisions/DEC-001_model_dataset_choice.md) |
| DEC-002 | Entorno de ejecución en WSL/Ubuntu 22.04 con venv en ext4 y pip arrancado sin sudo. | — | [decisions/DEC-002_execution_environment.md](decisions/DEC-002_execution_environment.md) |
| DEC-003 | No cambiamos de dataset: HatEval entrena la dirección, HateCheck la falsa fuera de distribución. | — | [decisions/DEC-003_evaluation_protocol.md](decisions/DEC-003_evaluation_protocol.md) |
| DEC-004 | El eje del flujo es el paso de razonamiento (capa como secundario); la curvatura es de Menger, no `np.cross`. | — | [decisions/DEC-004_flow_axis_and_curvature.md](decisions/DEC-004_flow_axis_and_curvature.md) |
| DEC-005 | Tres patas (HateCheck / BBQ / HatEval), BBQ como corpus de razonamiento, y diccionario de conceptos sin SAE. | — | [decisions/DEC-005_three_leg_architecture.md](decisions/DEC-005_three_leg_architecture.md) |

## Estado del arte (SOA)

| ID | Hook (una frase) | Relevancia | Artefacto |
|----|-------------------|------------|-----------|
| SOA-001 | En HatEval un TF-IDF ya da AUC 0.834 en dev, la señal es léxica (hashtags de campaña) y hay 225 tweets filtrados de train a test. | Alta — invalida el criterio de éxito original de EXP-001 | [references/SOA-001_hateval_artifacts.md](references/SOA-001_hateval_artifacts.md) |
| SOA-002 | HateCheck: 3728 tests funcionales con pares mínimos ya construidos; el baseline léxico de HatEval se desploma ahí a AUC 0.5875. | Alta — es el instrumento de falsación de EXP-001 | [references/SOA-002_hatecheck_instrument.md](references/SOA-002_hatecheck_instrument.md) |
| SOA-003 | Dherin et al.: una pasada con contexto ≡ pasada sin contexto + actualización de pesos de rango 1; su parte vectorial *es* un steering vector. | Alta — fundamento teórico de intervenir sin tocar pesos | [references/SOA-003_implicit_dynamics_icl.md](references/SOA-003_implicit_dynamics_icl.md) |
| SOA-004 | Højer et al.: extracción en último token, tres construcciones de vector de control, aplicación en capa media, validación por KL/entropía/probabilidad. | Alta — es el mecanismo operativo | [references/SOA-004_representation_engineering.md](references/SOA-004_representation_engineering.md) |
| SOA-005 | Helff et al. (AR): conceptos como tuplas (nombre, representación, umbral) y reglas lógicas encima; no hay SAE para Qwen 7B. | Alta — capa de explicabilidad y guardarraíl | [references/SOA-005_activation_reasoning.md](references/SOA-005_activation_reasoning.md) |
| SOA-006 | Zhou et al.: razonamiento como flujo sobre pasos; curvatura de Menger, válida en cualquier dimensión. | Alta — fija el eje t y arregla la curvatura | [references/SOA-006_geometry_of_reasoning.md](references/SOA-006_geometry_of_reasoning.md) |
| SOA-007 | Chen et al.: persona vectors en Qwen2.5-7B-Instruct con la misma intervención aditiva; la proyección previa a generar predice el rasgo (r 0.75–0.83), pero sobre todo entre tipos de prompt. | Alta — mismo modelo; capa elegida por efectividad causal y límite de la monitorización por proyección | [references/SOA-007_persona_vectors.md](references/SOA-007_persona_vectors.md) |
| _(pendiente — sin destilar: `lapo.pdf`, `owls_like_numbers.pdf`, `2026.findings-acl.707.pdf`, `2604.28192v3.pdf`, `NeurIPS-2025-consistent-paths…`)_ | | | |

---

## Cómo añadir una fila

1. Copia la plantilla correspondiente de `data/_templates/`.
2. Asigna el siguiente ID libre de su tipo (no reutilices IDs, aunque el
   artefacto se elimine después).
3. Rellena el artefacto en su carpeta (`data/questions/`,
   `data/experiments/`, `data/decisions/`, `data/references/`).
4. Añade la fila aquí con el hook (una frase, no un resumen largo — el
   detalle vive en el artefacto).
5. Enlaza relaciones tipadas dentro del propio artefacto (`motivated_by`,
   `apoyado_en`, `supersede`, etc.), no aquí.
