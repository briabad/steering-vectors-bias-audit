---
id: DEC-005
type: decision
supersede: null
---

# DEC-005 — Arquitectura de tres patas, BBQ como corpus de razonamiento, y diccionario de conceptos sin SAE

## Trigger
Tres presiones simultáneas:

1. [SOA-001] y [SOA-002] mostraron que HatEval mide **detección de contenido** con
   señal léxica, no razonamiento sesgado. Llegué a proponer archivarlo.
2. El objetivo declarado del proyecto incluye **sembrar métodos de seguridad y
   guardarraíles** — identificar contenido malicioso, discriminación, misoginia. Eso
   necesita precisamente detección de contenido. Archivar HatEval habría amputado esa
   mitad.
3. [SOA-005] (ActivationReasoning) es la capa de explicabilidad que `Q-001` pide, pero
   depende de SAEs, y **no hay SAE público para Qwen 2.5 7B**.

## Decisión

### 1. Tres patas, tres preguntas distintas

| corpus | qué es | pregunta que contesta | rol |
|---|---|---|---|
| **HateCheck** (`Paul/hatecheck`) | 3728 tests funcionales por plantilla, 29 `functionality`, 7 grupos | ¿el concepto es **preciso**? | fuente del diccionario + banco de guardarraíl |
| **BBQ** (`oskarvanderwal/bbq`) | QA de estereotipos, 11 categorías, `ambig`/`disambig` 50/50 | ¿el **razonamiento** se apoya en el estereotipo? | corpus de razonamiento |
| **HatEval** | 9000 tweets reales, ruidosos | ¿el concepto **sobrevive** fuera de plantilla? | pata "en la vida real" |

HatEval **no se archiva**: se degrada de corpus principal a prueba de transferencia.
Sigue siendo el único texto salvaje que tenemos, y sin él no se puede afirmar que los
conceptos no sean un artefacto de plantilla.

### 2. BBQ como corpus de razonamiento

Sustituye a HatEval en el papel principal para el bloque de razonamiento, por tres
propiedades de diseño que HatEval no tiene:

- En `ambig` la respuesta correcta es siempre "no se puede determinar". **Responder otra
  cosa *es* el sesgo** — el evento es conductual, no una etiqueta de anotador.
- `ambig` y `disambig` comparten casi todo el vocabulario y tienen respuestas correctas
  distintas: el atajo léxico de [SOA-001] es **estructuralmente incapaz** de resolverlo.
- `disambig` es contra-estereotípico: la evidencia apunta al grupo no estereotipado.
  Contrafactual incorporado.

Además es opción múltiple, así que se mide con logits sobre un conjunto restringido de
respuestas, igual que [SOA-004].

### 3. El diccionario de conceptos se construye con vectores de control, no con SAEs

**Verificado 2026-09-05** contra la API de Hugging Face: no existe SAE público para
Qwen 2.5 7B. Hay para Qwen 2.5 1.5B y 14B (`huypn16`, no oficiales, <10 descargas) y
Qwen 1.5 0.5B. Las suites maduras cubren otros modelos: Gemma Scope (Gemma-2 2B/9B) y
Llama Scope (Llama-3.1-8B).

Alternativas: (a) cambiar de modelo a Gemma-2-9B, rompiendo [DEC-001]; (b) entrenar un
SAE propio, semanas de trabajo y cómputo; (c) **obtener $r_c$ de otra forma**.

Se elige (c). [SOA-005] define un concepto como $(n_c, r_c, \tau_c)$ y declara el marco
*"method-agnostic"*. Instanciamos:

$$r_c(h_t) = \langle h_t,\ \hat v_c\rangle, \qquad \hat v_c = \frac{v_c}{\lVert v_c\rVert}$$

donde $v_c$ es un vector de control contrastivo al estilo [SOA-004] entrenado para ese
concepto, y $\tau_c$ se calibra en train. Corresponde a $\mathcal{R}_\text{single}$ del
paper con la feature sustituida por una dirección supervisada.

**Los conceptos y sus contrastes salen de las columnas de HateCheck:**

| concepto | positivos | negativos que lo hacen preciso |
|---|---|---|
| deshumanización | `derog_dehum_h` | `ident_neutral_nh`, `ident_pos_nh` |
| amenaza | `threat_dir_h`, `threat_norm_h` | `target_obj_nh` |
| slur | `slur_h` | `slur_reclaimed_nh`, `slur_homonym_nh` |
| contra-discurso | `counter_quote_nh`, `counter_ref_nh` | `derog_neg_emote_h` |
| profanidad | `profanity_h` | `profanity_nh` |
| negación | `negate_pos_h` | `negate_neg_nh` |

### 4. Reglas escritas a mano en la primera pasada

[SOA-005] escribe sus reglas a mano. Inducirlas desde datos es programación lógica
inductiva: otro problema, mucho más duro. Primera pasada = reglas a mano sobre conceptos
descubiertos. La inducción queda como fase estirada (FEAT-009).

### 5. Alcance de la PoC

Dimensionado para **decidir viabilidad, no para publicar**:

- **EXP-002** (razonamiento): **las 11 categorías de BBQ con sus pools completos**
  (ver Addendum 1 más abajo — la restricción original a `Age` no sobrevivió a la
  medición).
- **EXP-003** (guardarraíl): **cinco conceptos**, HateCheck completo (3728 frases cortas
  son minutos de GPU; el límite es el alcance, no el cómputo).
- **Fuera de la PoC**: el bloque de geometría (requiere generar CoT, ver [DEC-004]) y la
  inducción de reglas.

---

## Addendum 1 (2026-09-05) — el alcance de EXP-002 pasa de una categoría a las once

> Escrito tras el barrido exploratorio de `scripts/03_bbq_category_sweep.py`, cuya salida
> vive en `data/experiments/EXP-002_bbq_stereotype_direction/output/category_sweep.json`.
> No invalida ninguna hipótesis pre-registrada: el contraste no se había construido aún.

**Motivo.** Se midió `s_amb` sobre 150 ítems por categoría, promediando 3 permutaciones
de opciones y con los dos formatos de prompt. Ninguna categoría por separado reúne a la
vez sesgo direccional suficiente y masa suficiente:

| categoría (formato chat) | `rho_unk` | `s_amb` | n estereotipada (obs) |
|---|---|---|---|
| Religion | 0.180 | **0.481** | 60 |
| Disability_status | 0.107 | **0.583** | 38 |
| Sexual_orientation | 0.038 | 0.647 | 14 |
| Physical_appearance | 0.060 | 0.407 | 19 |
| Nationality | 0.051 | 0.217 | 14 |
| Age | 0.264 | 0.176 | 70 |
| Race_ethnicity | 0.044 | −0.200 | 8 |
| Race_x_gender | 0.011 | −0.600 | 1 |
| SES | 0.022 | −0.800 | 1 |
| Race_x_SES | 0.024 | **−1.000** | 0 |

`Age` es la única con masa y su `s_amb` queda por debajo del umbral; las que superan el
umbral no llegan a 100 ítems distintos ni usando su pool entero. Agrupadas sí: ~450
ítems distintos con `s_amb` conjunto ≈ 0.36 sobre las de signo positivo, ≈ 0.24
incluyéndolas todas.

**Además, agrupar es lo correcto**: [Q-001] pregunta por una dirección de *razonamiento
sesgado*, no de *sesgo de edad*.

**Hallazgo colateral que merece seguimiento.** El signo se invierte por eje. En religión,
discapacidad, orientación sexual, apariencia, nacionalidad y edad el modelo falla
**hacia** el estereotipo; en raza y clase social falla **en contra** — en `Race_x_SES`
eligió la opción estereotipada 0 veces de 11. La explicación más simple es sobrecorrección
por RLHF sobre el eje más entrenado en seguridad. Si se confirma con los pools completos,
implica que **una dirección latente única de "estereotipo" puede no existir**: el
fenómeno sería dependiente del eje. Es material para un artefacto propio.

**Nota de calidad de la medición.** Las cifras de arriba cuentan *observaciones*
(ítem × permutación), no ítems distintos. Al repetirse cada ítem tres veces, las
observaciones están correlacionadas y el `n` aparente exagera la precisión. La corrida
definitiva debe contar ítems distintos y agrupar por ítem al estimar incertidumbre.

## Rationale
- Detección de contenido y razonamiento sesgado no compiten: son las dos capas que un
  guardarraíl necesita — filtro sobre el *input* y restricción sobre el *proceso*.
- El puente técnico es el diccionario: HateCheck aporta el vocabulario de conceptos
  sobre el que las reglas de [SOA-005] operan. Sin él, AR no tiene de dónde sacar los
  nombres.
- El diccionario es además **el artefacto que sobrevive al experimento**: reutilizable,
  auditable, y cuando el guardarraíl bloquea dice qué concepto disparó. Es la "semilla"
  que el proyecto persigue.
- La medición de [SOA-002] justifica el objetivo: el baseline falla **en las dos
  direcciones** (recall 0.388, precisión 0.752) y sus falsos positivos silencian
  contra-discurso (0.225), uso reapropiado (0.247) y afirmaciones **positivas** sobre
  grupos protegidos (`ident_pos_nh` 0.365). Bajar eso manteniendo recall es la
  contribución, y no está resuelto en sistemas desplegados.

## Predicción
- Los baselines léxicos sobre BBQ quedarán **cerca del azar**, por construcción. Si no
  lo están, hay fuga en el diseño y hay que encontrarla antes de seguir.
- Los conceptos derivados de HateCheck separarán sus propios positivos/negativos con
  AUC > 0.8 en held-out — son contrastes limpios y templados.
- La transferencia a HatEval será **peor** que dentro de HateCheck: es la prueba de que
  no son artefacto de plantilla, y un descenso moderado es el resultado esperado, no un
  fallo.

## Enlaces
- motivated_by: [Q-001]
- apoyado_en: [SOA-002, SOA-005, SOA-004, SOA-001]
- modifica_alcance_de: [DEC-003, DEC-001]
- supersede: null
