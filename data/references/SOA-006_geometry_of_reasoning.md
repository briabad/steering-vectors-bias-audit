---
id: SOA-006
type: state_of_the_art
---

# SOA-006 — Zhou et al.: el razonamiento como flujo, y la curvatura que sí funciona

## Source
Zhou, Wang, Yin, Zhou, Zhang (Duke University). *The Geometry of Reasoning: Flowing
Logics in Representation Space*. ICLR 2026.
Código: `github.com/MasterZhou1/Reasoning-Flow` · Dataset:
`huggingface.co/datasets/MasterZhou/Reasoning-Flow` (verificado, existe).
Local: `data/00_context/THE_GEOMETRY-OF-REASONING-FLOWING-LOGICS-IN.pdf` (29 pp).

## Summary

**Tesis.** El razonamiento de un LLM se modela como **flujos**: trayectorias de
embeddings que evolucionan por donde va la lógica. Dos afirmaciones teóricas:
(1) el razonamiento corresponde a flujos suaves en el espacio de representación;
(2) **los enunciados lógicos actúan como controladores locales de la velocidad de esos
flujos**.

**El eje temporal es el paso de razonamiento** (Def. 3.1): CoT es un proceso iterativo
que genera $\mathcal{U} = (u_1,\dots,u_T)$. No es la capa.

**Operador de representación** (Def. 3.2): $\mathcal{E}: \mathcal{V}^*\times\mathcal{I}\to\mathbb{R}^d$,
donde el índice $\iota$ especifica el tipo de representación — posición de token, regla
de pooling, o estado interno de una capa. En la práctica lo instancian con un encoder
preentrenado (Qwen3 Embedding, `text-embedding-3-large`) **o extrayendo hidden states
directamente del LLM**, con mean pooling o el estado del último token.

**Curvatura de Menger** (Def. 3.4) — la pieza que nos hacía falta. Para tres puntos
distintos $x_1,x_2,x_3\in\mathbb{R}^n$, es el recíproco del radio del círculo único que
pasa por los tres:

$$c(x_1,x_2,x_3) = \frac{1}{R(x_1,x_2,x_3)} = \frac{4A}{\lVert x_1-x_2\rVert\,\lVert x_2-x_3\rVert\,\lVert x_3-x_1\rVert}$$

con $A$ el área del triángulo, obtenible por Herón desde las tres distancias.
**Se calcula solo con distancias y vale en cualquier dimensión.** La eligen porque
captura a la vez desviación angular y variación de distancia, lo que la hace adecuada
para trayectorias representadas como embeddings discretos.

**El diseño experimental clave.** Construyen un dataset de lógica formal que
**desacopla estructura lógica de superficie semántica**: las *mismas* proposiciones de
deducción natural con *distintos portadores semánticos*. Eso permite probar si el
modelo internaliza la lógica más allá de la forma superficial.

**Experimentos en las familias Qwen y LLaMA** — sugieren una regularidad
representacional posiblemente universal, independiente de la receta de entrenamiento.

## Relevance

1. **Resuelve el bug de curvatura.** `scripts/01_hateval_latent_pipeline.py:108` usa
   `np.cross`, que solo admite 2 o 3 dimensiones y en numpy 2.2.6 ya ni existe para
   2-d. Menger es el reemplazo correcto y no requiere producto vectorial. Ver [DEC-004].

2. **Fija el eje $t$.** El eje bueno es el **paso de razonamiento**, no la capa. Eso
   obliga a *generar* CoT sobre datasets de una sola respuesta como BBQ, lo que
   encarece el bloque de geometría respecto al resto. Ver [DEC-004].

3. **Su control es el que llevamos exigiendo desde [SOA-001].** Misma lógica, distinto
   portador semántico. En BBQ eso se traduce en: misma plantilla, distinto grupo
   protegido.

4. **Aviso de confusión**: trayectorias más largas tienen más ocasión de curvarse. Si
   se compara curvatura entre clases sin normalizar por longitud, se mide verbosidad,
   no geometría. Es el mismo problema de norma que aparece en el reescalado de PCA de
   [SOA-004].

## Enlaces
- medicion_de: [Q-001]
- resuelve_bug_en: [DEC-004]
- relacionado: [SOA-004, SOA-001]
