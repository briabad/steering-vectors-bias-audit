---
id: SOA-004
type: state_of_the_art
---

# SOA-004 — Højer et al.: cómo se construye y se aplica un vector de control

## Source
Højer, Jarvis, Heinrich (IT University of Copenhagen). *Improving Reasoning Performance
in Large Language Models via Representation Engineering*. ICLR 2025, arXiv:2504.19483.
Código: `github.com/bertramhojer/improve-reasoning-iclr-2025`.
Local: `data/00_context/IMPROVING REASONING PERFORMANCE IN LARGE.pdf` (18 pp).

## Summary

**Extracción.** Se captura el residual stream **después de cada capa, en la posición
del último token** de un ejemplo. Da $H_\ell(P_i)$ por capa $\ell$ y ejemplo $i$. Usan
ejemplos *few-shot* al extraer, en parte porque los LLM son aprendices en contexto.

**Tres construcciones del vector**, por calidad creciente:

| | fórmula | nota |
|---|---|---|
| reading vector (ec. 2) | $c_\ell = \frac{1}{\lvert P\rvert}\sum_i H_\ell(P_i)$ | la peor; sin contraste |
| contrastivo (ec. 3) | $c_\ell = \frac{1}{\lvert P^\pm\rvert}\sum_i \left(H_\ell(P_i^+) - H_\ell(P_i^-)\right)$ | media de diferencias **emparejadas** |
| PCA (ec. 4) | $c_\ell = \mathrm{PCA}\left(\{H_\ell(P_i^+) - H_\ell(P_i^-)\}\right)_{(1)}$ | primera componente |

**Aplicación (ec. 1).** $x_{\ell+1} = \mathrm{LayerNorm}(y_\ell + \mathrm{MLP}(y_\ell)) + c_\ell\cdot\alpha$.
Solo en la **capa media** — citan a Templeton et al. 2024 para justificar que basta una.
Barren $\alpha\in[-1,1]$ en pasos de 0.1, y $[-3,3]$ en Mistral-7B-Instruct.

**Un vector, dos direcciones.** *"The control vector can thus be scaled to induce the
desired behavior or its opposite."* No hacen falta dos vectores: $\alpha>0$ induce,
$\alpha<0$ evita.

**Trampa de escala.** Con reading vector, $\alpha=1$ equivale a sumar una activación
completa porque $\lVert c_\ell\rVert \simeq \lVert H_\ell\rVert$. Con PCA
$\lVert c_\ell\rVert = 1$, así que **reescalan el vector según la norma real de las
activaciones extraídas** (§A.1). Sin ese reescalado, $\alpha$ no es comparable entre
métodos.

**Protocolo.** 400 prompts para derivar los vectores en GSM8K. Splits train/test
**estratificados**. El vector se deriva en train y se evalúa aplicándolo en test.

**El prompt negativo es su parte frágil.** Para "mal razonamiento" probaron: (1) pedir
al modelo que responda mal — descartado, producir la respuesta incorrecta a propósito
puede ser *buen* razonamiento; (2) usar casos donde el modelo falla — ambiguo, a veces
la respuesta no es incorrecta sino solo distinta del token esperado; (3) **cadenas
aleatorias de 75 caracteres A-z** como "punto de referencia". Usan (2) y (3).

**Métricas de validación — no usan AUC.** Evalúan causalmente:

- divergencia KL entre la distribución de logits con y sin intervención (ec. 5)
- entropía en función de $\alpha$ (ec. 6)
- masa de probabilidad en la respuesta correcta (ec. 7-8)

Y aplican un **criterio de coherencia**: si baja la entropía *y* sube la probabilidad
del token correcto *y* sube la accuracy, la intervención funciona. Si las métricas no
se alinean — p. ej. baja la entropía pero la accuracy no mejora — *"it might indicate
that we are affecting the model in unintended ways"*.

## Relevance

1. **Es el mecanismo operativo de toda la línea.** Extracción en último token, tres
   construcciones, aplicación en capa media, barrido de $\alpha$.

2. **Su punto débil es nuestra ventaja.** Su contraste negativo es un apaño porque
   "mal razonamiento" no tiene definición limpia. En estereotipos y odio el contraste
   es natural: [DEC-005] lo deriva del **comportamiento** del modelo en BBQ
   (responde estereotipo vs responde "unknown"), que es su esquema (2) sin la
   ambigüedad que a ellos les obligó a usar ruido aleatorio.

3. **La triangulación KL/entropía/probabilidad sustituye al frente de Pareto** que
   habíamos planteado para el steering. Es mejor criterio y viene validado.

4. **El reescalado de norma es la misma trampa** que aparece en la curvatura de
   [SOA-006]: en alta dimensión la norma domina si no se controla.

## Enlaces
- mecanismo_de: [Q-001]
- usado_por: [EXP-002, EXP-003]
- relacionado: [SOA-003, SOA-005, SOA-006]
