---
id: SOA-007
type: state_of_the_art
---

# SOA-007 — Chen et al.: persona vectors, y el límite de la monitorización por proyección

## Source
Chen, Arditi, Sleight, Evans, Lindsey (Anthropic Fellows Program / UT Austin /
Constellation / Truthful AI / UC Berkeley). *Persona Vectors: Monitoring and Controlling
Character Traits in Language Models*. arXiv:2507.21509v3, 5 sep 2025 (preprint, 63 pp).
Código: `github.com/safety-research/persona_vectors`.
Local: `data/00_context/persona_vector.pdf`.

> **Relevancia máxima para este trabajo: usan Qwen2.5-7B-Instruct**, el mismo modelo que
> nosotros, con la misma forma de intervención. Los resultados son directamente comparables.

## Summary

**Qué es un persona vector.** Una dirección en el espacio de activaciones que subyace a un
rasgo de carácter — *evil*, *sycophancy*, propensión a *hallucinate*. Se extrae con un
*pipeline* automatizado que solo requiere el nombre del rasgo y una descripción en lenguaje
natural.

**Extracción.** Un LLM frontera (Claude 3.7 Sonnet) genera 5 pares de *system prompts*
contrastivos (uno que induce el rasgo, otro que lo suprime), 40 preguntas de evaluación
divididas en conjunto de extracción y de evaluación, y una rúbrica. Se generan 10
*rollouts* por pregunta y prompt, se filtran por puntuación de expresión del rasgo emitida
por un juez (GPT-4.1-mini, escala 0–100, umbral 50), y el vector es la **diferencia de
medias de activaciones** entre respuestas que exhiben el rasgo y las que no.

**Dónde extraen, y por qué importa para nosotros.** Promedian sobre los **tokens de la
respuesta**, no del prompt. Reportan explícitamente (Apéndice A.3) que *"response tokens
yield more effective steering directions than alternative positions such as prompt tokens"*.
Nosotros extraemos en el último token del prompt.

**Selección de capa — y ésta es la lección metodológica que más nos costó.** Obtienen un
vector candidato por capa y **eligen la capa probando la efectividad del steering**, no el
AUC de la sonda (Apéndice B.4). Nuestro error de elegir la capa 22 por la meseta del AUC
emparejado —cuando la 20 generaliza mucho mejor— es exactamente lo que este criterio evita.

**Intervención.** Idéntica a la nuestra:

$$h_\ell \leftarrow h_\ell + \alpha\, v_\ell \quad\text{en cada paso de decodificación}$$

y para mitigar, $h_\ell \leftarrow h_\ell - \alpha\, v_\ell$. Un solo vector en ambos
sentidos, como en `SOA-004`.

## El resultado que más nos toca: monitorización por proyección

§3.3. Proyectan la activación del **último token del prompt** —el inmediatamente anterior a
la respuesta— sobre el persona vector, y la correlacionan con la expresión del rasgo en la
respuesta que aún no se ha generado.

$$r = 0.75\text{–}0.83$$

*"suggesting that persona vectors can be useful for monitoring prompt-induced behavioral
shifts before text generation occurs."*

**Pero el aviso siguiente es el que explica nuestro resultado negativo de la compuerta:**

> *"These correlations arise primarily from distinguishing between different prompt types
> (e.g., trait-encouraging vs trait-discouraging system prompts), with more modest
> correlations when controlling for prompt type (Appendix C.2). This indicates the persona
> vectors are effective for detecting clear and explicit prompt-induced shifts, but may be
> **less reliable for more subtle behavioral changes in deployment settings**."*

Es decir: su $r=0.75$–$0.83$ mide sobre todo *"¿este prompt pide ser malvado o pide ser
útil?"*. Controlando el tipo de prompt, la correlación cae.

### Cómo se relaciona con nuestro hallazgo

Nuestro caso es precisamente el escenario que ellos declaran incierto: **no hay manipulación
de system prompt**, todos los ítems son del mismo tipo, y la variación conductual es sutil
—el mismo modelo, ante prompts del mismo formato, a veces abandona la respuesta correcta y a
veces no—.

| | persona vectors | este trabajo |
|---|---|---|
| variación entre condiciones | *system prompt* explícito, de suprimir a inducir | ninguna: mismo formato en todos los ítems |
| señal reportada | $r = 0.75$–$0.83$ entre tipos de prompt | AUC 0.61–0.63 fuera de muestra |
| al controlar el contexto | *"more modest correlations"* | AUC 0.76 dentro de pares emparejados |

Los dos patrones apuntan a lo mismo desde lados opuestos: **una proyección puede parecer
predictiva porque rastrea una propiedad gruesa de la entrada, no el desenlace conductual
concreto**. Ellos lo ven al controlar el tipo de prompt; nosotros al salir del diseño
emparejado.

La diferencia a nuestro favor: nuestra señal **sobrevive** el control de contexto (0.76
dentro de pares idénticos en categoría, plantilla y polaridad), que es el control que
degrada la suya. La diferencia en contra: fuera de ese control queda en 0.61–0.63, lift
1.2–1.3, insuficiente para una compuerta.

## Otras aportaciones (fuera de nuestro alcance actual pero relevantes como continuación)

- **Predicción de deriva por finetuning.** El desplazamiento de activaciones a lo largo del
  persona vector durante el finetuning correlaciona con la expresión del rasgo posterior
  ($r = 0.76$–$0.97$), por encima de las líneas base entre rasgos ($r = 0.34$–$0.86$).
- **Steering preventivo.** Amplificar el vector *durante* el finetuning evita la deriva,
  en vez de corregirla después.
- **Marcado de datos de entrenamiento** antes de entrenar, por proyección, a nivel de
  conjunto y de muestra individual, incluyendo casos que escapan al filtrado por LLM.
- **Advertencia sobre correlación entre rasgos.** Los rasgos negativos (y, sorprendentemente,
  el humor) se desplazan juntos y en sentido opuesto al optimismo. Los persona vectors
  correlacionan entre sí, lo que limita la interpretación de cualquier vector aislado.

## Qué tomamos y qué no

**Se toma:**
- La confirmación de que la intervención aditiva en el residual stream funciona en
  Qwen2.5-7B-Instruct, nuestro modelo.
- **El criterio de selección de capa por efectividad causal**, no por AUC. Corrige un error
  nuestro documentado en `results.md` §III.12.
- El aviso de §3.3 como explicación externa de nuestro negativo en la compuerta.

**No se toma:**
- El *pipeline* automatizado por LLM juez: nuestra etiqueta es conductual y objetiva (la
  respuesta correcta de BBQ en `ambig` es la abstención), no una puntuación de rúbrica.
- La extracción en tokens de respuesta: nuestra pregunta es específicamente si el estado
  **previo** a la generación anticipa el desenlace, luego el último token del prompt es el
  punto correcto para nosotros. Su hallazgo sugiere, eso sí, que un vector extraído de
  tokens de respuesta podría ser **más efectivo para steering** — línea abierta.

## Enlaces
- método de construcción e intervención: [[SOA-004]]
- fundamento teórico de intervenir sin pesos: [[SOA-003]]
- resultados propios que este trabajo contextualiza:
  `EXP-002/results.md` §III.4 (causal), §III.9 (señal en el prompt), §III.12 (compuerta)
