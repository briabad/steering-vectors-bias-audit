---
id: SOA-005
type: state_of_the_art
---

# SOA-005 — Helff et al.: reglas lógicas sobre activaciones latentes

## Source
Helff, Härle, Stammer, Friedrich, Brack, Wüst, Shindo, Schramowski, Kersting
(TU Darmstadt / hessian.AI / Aleph Alpha / Meta FAIR / DFKI).
*ActivationReasoning: Logical Reasoning in Latent Activation Spaces*. ICLR 2026.
Código y dataset: `github.com/ml-research/ActivationReasoning`.
Local: `data/00_context/ICLR-2026-activationreasoning-...pdf` (24 pp).

## Summary

**Tres etapas.**
1. *Finding latent representations* — se identifican representaciones de concepto en el
   espacio latente y se recogen en un **diccionario**.
2. *Activating propositions* — en inferencia, las activaciones por token se detectan y
   se tratan como **unidades proposicionales**.
3. *Logical reasoning* — se aplican **reglas** sobre las proposiciones activadas para
   inferir estructuras de orden superior, componer conceptos nuevos y dirigir el
   comportamiento del modelo.

**Formalización del concepto.** Un concepto es la tupla $c = (n_c, r_c, \tau_c)$:
- $n_c$ — identificador semántico (p. ej. "Bridge")
- $r_c: \mathcal{L}\to\mathbb{R}_{\geq 0}$ — la **función de representación**, que mapea
  el código latente $\ell_t = E(h_t)$ a una puntuación de activación $a(c,t) := r_c(\ell_t)$
- $\tau_c \in \mathbb{R}_{\geq 0}$ — umbral blando de activación

El diccionario es $\mathcal{D} = \{c_i\}_{i=1}^{N}$.

**Tres formas de representación**: $\mathcal{R}_\text{single}$ (un concepto = una
feature), $\mathcal{R}_\text{multi}$ (un concepto = $k$ features agregadas con pesos),
$\mathcal{R}_\text{relation}$ (un concepto = relaciones entre features).

**Composición.** Conceptos ausentes del diccionario se derivan de la activación
conjunta de otros: `Bridge ∧ San Francisco ∧ USA → Golden Gate Bridge`.

**El panel de guardarraíl (Fig. 2, "Constitutional Reasoning")** — es literalmente
nuestro caso de uso:
```
Weapon ∧ ¬Police  → Unsafe
¬Weapon ∨ Police  → Safe
Drug ∧ Medical    → Safe
```
`"A police officer carries a gun"` → Safe ✓ · `"A criminal carries a gun"` → Unsafe ✓ ·
`"A patient receives morphine in a hospital"` → Safe ✓

**Evaluación**: PrOntoQA (multi-hop), Rail2Country (abstracción), ProverQA (lenguaje
natural), y **BeaverTails** (seguridad sensible al contexto).

## Restricción práctica crítica

Usan **SAEs** para obtener $r_c$. **No existe un SAE público para Qwen 2.5 7B**
(verificado 2026-09-05 contra la API de Hugging Face): hay SAEs para Qwen 2.5 1.5B y
14B (`huypn16`, no oficiales, <10 descargas) y Qwen 1.5 0.5B. Las suites maduras son
**Gemma Scope** (Gemma-2 2B/9B, oficial) y **Llama Scope** (Llama-3.1-8B).

**Pero el paper dice explícitamente que el marco es agnóstico al método:** *"The
framework remains method-agnostic and can directly integrate future advances in
interpretability."* El SAE es solo una forma de obtener $r_c$. **Un vector de control
contrastivo de [SOA-004] es otra**: $r_c(\ell_t) = \langle h_t, v_c\rangle$ con umbral
$\tau_c$. Esa es la salida adoptada en [DEC-005].

## Relevance

1. **Es la capa de explicabilidad que Q-001 pide.** Convierte una proyección escalar en
   una proposición con nombre, y las proposiciones en reglas auditables. Cuando el
   guardarraíl bloquea, dice **qué concepto disparó**.

2. **El diccionario de conceptos es el artefacto reutilizable.** Sobrevive al
   experimento concreto: es la "semilla" para métodos de seguridad posteriores.

3. **Ojo con el salto que NO da el paper**: sus reglas están **escritas a mano** por
   expertos. Inducir reglas desde datos es programación lógica inductiva, un problema
   distinto y mucho más duro. Primera pasada: reglas a mano sobre conceptos
   descubiertos. La inducción queda como fase estirada.

## Enlaces
- capa_de_explicabilidad_de: [Q-001]
- usado_por: [EXP-003]
- depende_de: [SOA-004]
- restringido_por: [DEC-005]
