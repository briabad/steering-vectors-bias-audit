---
id: DEC-003
type: decision
supersede: null
---

# DEC-003 — No cambiamos de dataset: HatEval entrena, HateCheck falsa

## Trigger
[SOA-001] mostró que en HatEval un TF-IDF puro alcanza AUC 0.834 en `dev`, por encima
del criterio de éxito pre-registrado de EXP-001 (AUC ≥ 0.75), con coeficientes
dominados por hashtags de campaña (`buildthatwall`, `maga`, `nodaca`) e insultos de
diccionario (`bitch`, `whore`, `hoe`). Surgió entonces la pregunta de si convenía
abandonar HatEval y usar otro dataset.

## Alternativas consideradas

- **Opción A — sustituir HatEval por otro corpus de odio.** Candidatos verificados
  como existentes y cargables: `tdavidson/hate_speech_offensive`,
  `ucberkeley-dlab/measuring-hate-speech`, `allenai/social_bias_frames`,
  `google/civil_comments`, `tweets-hate-speech-detection/…`.
- **Opción B — mantener HatEval como entrenamiento y añadir un segundo dataset como
  evaluación fuera de distribución.**
- **Opción C — mantener HatEval sola y confiar en pares contrafactuales escritos a
  mano** (el plan original de FEAT-007).

Nota: "POLAR", mencionado como Opción C en [DEC-001], no se encontró en Hugging Face
bajo ese nombre ni como `LLM-Beetle/POLAR`. Queda pendiente de identificar si se
quiere reconsiderar.

## Decisión
**Opción B.** HatEval se mantiene como corpus de **entrenamiento** de la dirección
latente. Se añade **HateCheck** (`Paul/hatecheck`, ver [SOA-002]) como **instrumento
de evaluación y falsación**, fuera de distribución.

Protocolo de evaluación, fijado antes de extraer ninguna activación:

| rol | fuente | n | uso |
|---|---|---|---|
| ajuste | HatEval `train` | 9000 | única fuente para estimar µ_hate − µ_neutral |
| validación en distribución | HatEval `dev` | 1000 | solape 0 con train; comparación contra baseline léxico |
| **falsación** | **HateCheck** `test` | **3728** | criterio principal; nunca interviene en el ajuste |
| contrafactual | HateCheck agrupado por `templ_id` | 442 plantillas | dispersión intra-plantilla del score |
| reservado | HatEval `test` | 3000 | solo con los 225 solapes eliminados, y reportando ambos números |

**Listones congelados** (medidos en [SOA-001] y [SOA-002], no se recalculan):

- baseline léxico en HatEval `dev`: **AUC 0.8342**
- baseline léxico en HateCheck: **AUC 0.5875**
- dispersión intra-plantilla del baseline: **media 0.2313, máx 0.6422**

## Rationale

- **Cambiar de dataset no elimina el artefacto, lo reubica.** El atajo léxico no es un
  defecto de HatEval: es consecuencia de cómo se construyen estos corpus, muestreando
  tweets por palabra clave y hashtag, lo que hornea el término dentro de la etiqueta.
  Cualquier corpus de la Opción A traería su propia versión del mismo problema, y
  habríamos gastado el presupuesto en descubrirla otra vez.

- **Lo que falta no es otro corpus de entrenamiento, es un instrumento de medida.**
  HateCheck está diseñado a propósito para separar "el modelo lee la palabra" de "el
  modelo entiende la frase": incluye odio implícito, odio negado, contra-discurso que
  cita odio, homónimos de slur y profanidad inocua.

- **La evidencia es contundente y ya está medida.** El mismo baseline pasa de AUC
  0.834 en HatEval `dev` a **0.5875** en HateCheck, con accuracy 20 puntos por debajo
  de la clase mayoritaria. En el par mínimo `"I hate women."` (hateful, p=0.518) vs
  `"Let me be clear: I don't hate women."` (non-hateful, p=0.560) el ranking sale
  **invertido**. Ese hueco de 0.25 de AUC es exactamente el espacio donde la
  hipótesis de Q-001 puede demostrarse o refutarse.

- **Sale gratis en coste de anotación.** La Opción C requería escribir pares mínimos a
  mano, con el sesgo del que los escribe. HateCheck trae 442 plantillas × 7 grupos
  protegidos ya validadas, con `focus_words` marcando el span exacto.

- **Convierte un número en un diagnóstico.** El desglose por `functionality` dice qué
  tipo de razonamiento falla, no solo cuánto. Es lo que hace falta para las reglas
  interpretables de FEAT-009.

## Predicción
Si la hipótesis de [Q-001] es correcta:

- La dirección latente supera **AUC 0.65** en HateCheck (frente al 0.5875 léxico), sin
  haber visto un solo ejemplo de HateCheck durante el ajuste.
- La mejora se concentra en las funcionalidades composicionales — `negate_neg_nh`,
  `counter_quote_nh`, `negate_pos_h`, `derog_impl_h` — donde el léxico saca entre
  0.30 y 0.78 de accuracy.
- La dispersión intra-plantilla baja de 0.2313, es decir, el score depende menos del
  grupo protegido que el del baseline.

Si en cambio la dirección latente sube en HatEval `dev` pero se queda en ~0.59 en
HateCheck, la conclusión es que codifica vocabulario, y Q-001 queda **refutada** en su
forma actual. Ese resultado negativo también es publicable y hay que registrarlo.

## Enlaces
- motivated_by: [Q-001]
- apoyado_en: [SOA-001, SOA-002]
- modifica_alcance_de: [DEC-001]
- supersede: null
