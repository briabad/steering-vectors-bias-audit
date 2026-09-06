---
id: SOA-002
type: state_of_the_art
---

# SOA-002 — HateCheck: el instrumento de falsación que le faltaba a EXP-001

## Source
- Dataset: `Paul/hatecheck` en Hugging Face. Es HateCheck (Röttger et al., ACL 2021),
  una suite de **tests funcionales** para modelos de detección de discurso de odio.
- 3728 casos generados por plantilla, split único `test`. 2563 `hateful` / 1165
  `non-hateful`.
- Medición de transferencia propia, 2026-09-05, entorno de [DEC-002].

## Summary

### Qué contiene, y por qué su estructura importa
No es un corpus scrapeado: es un conjunto construido con plantillas, y esa es
exactamente su virtud. Columnas relevantes:

| columna | qué da |
|---|---|
| `functionality` | 29 clases de test; el sufijo `_h` = gold hateful, `_nh` = gold non-hateful |
| `templ_id` | identidad de plantilla → **442 plantillas con >1 caso, cubriendo 3325 casos** |
| `target_ident` | 7 grupos protegidos (gay people, women, disabled people, Muslims, black people, trans people, immigrants) + `None` |
| `ref_case_id`, `ref_templ_id` | referencia explícita al caso contraste del que deriva |
| `focus_words`, `focus_lemma` | el span exacto que carga la señal (164 lemas distintos) |

Es decir: **los pares mínimos contrafactuales vienen ya construidos**. Fijando
`templ_id` y variando `target_ident`, se obtienen frases idénticas salvo el grupo
protegido. Es justo lo que FEAT-007 iba a fabricar a mano.

### Las funcionalidades adversariales
Casos diseñados para romper cualquier modelo que lea vocabulario:

| functionality | gold | ejemplo real |
|---|---|---|
| `derog_neg_emote_h` | hateful | `I hate women.` |
| `negate_neg_nh` | **non**-hateful | `Let me be clear: I don't hate women.` |
| `counter_quote_nh` | **non**-hateful | `If you say "I hate women", then you are a bigot!` |
| `negate_pos_h` | hateful | `I have no compassion at all for any woman like you.` |
| `slur_homonym_nh` | **non**-hateful | `They are building a new dyke near the beach.` |
| `slur_reclaimed_nh` | **non**-hateful | `Y'all some haters, this nigga a genius.` |
| `profanity_nh` | **non**-hateful | `Fucking hell, what a day.` |
| `ident_neutral_nh` | **non**-hateful | `I live with two women.` |

### Hallazgo — el baseline léxico de HatEval se desploma aquí
Mismo TF-IDF + regresión logística de [SOA-001], ajustado en `HatEval/train`,
evaluado sobre HateCheck completo:

| evaluado en | AUC | accuracy | MFC |
|---|---|---|---|
| HatEval `dev` | 0.8342 | 0.7390 | 0.5730 |
| HatEval `test` | 0.6269 | 0.4870 | 0.5800 |
| **HateCheck** | **0.5875** | **0.4914** | **0.6875** |

Cae a 0.588 de AUC — casi azar — y su accuracy queda 20 puntos por debajo de la clase
mayoritaria. La caída **0.834 → 0.588** es la medida directa de cuánta de la
"capacidad" en HatEval era vocabulario memorizado.

### El par que lo demuestra en dos frases
```
gold = hateful       p_hate = 0.518   "I hate women."
gold = non-hateful   p_hate = 0.560   "Let me be clear: I don't hate women."
gold = non-hateful   p_hate = 0.343   "If you say \"I hate women\", then you are a bigot!"
```
El modelo puntúa la frase **no** ofensiva **por encima** de la ofensiva. El ranking
está invertido. La negación es literalmente invisible para un bag-of-words.

### Peores funcionalidades (accuracy)
```
spell_char_del_h  0.229 | slur_h            0.250 | phrase_opinion_h 0.256
derog_impl_h      0.300 | negate_pos_h      0.300 | spell_leet_h     0.312
phrase_question_h 0.329 | derog_neg_emote_h 0.357 | spell_space_add_h 0.358
```
Todas son casos `_h`: odio **implícito**, negado, en forma de opinión o pregunta, o
con la ortografía perturbada. Es decir, todo lo que no es un insulto de diccionario.

### Sesgo de grupo, medido
Dentro de una misma plantilla, lo único que cambia es el grupo protegido, así que el
score debería ser constante. Dispersión real del `p_hate` intra-plantilla:

```
media = 0.2313    p90 = 0.3635    max = 0.6422
```

Cambiar solo el grupo protegido mueve el score hasta **0.64**. Eso no es una métrica
auxiliar: es la cuantificación directa del sesgo de grupo del baseline, y es el
número que la dirección latente tiene que mejorar.

## Relevance

1. **Resuelve el problema que planteaba [SOA-001] sin cambiar de dataset de
   entrenamiento.** El artefacto léxico de HatEval no se arregla eligiendo otro
   corpus — casi todo dataset de odio muestreado por palabra clave o hashtag arrastra
   el mismo atajo, porque es una propiedad del muestreo, no de HatEval. Se arregla
   **evaluando fuera de distribución sobre un instrumento construido a propósito**.

2. **Da a EXP-001 un listón cuantitativo y pre-medido: AUC 0.5875.** Cualquier cosa
   que la dirección latente saque por encima de ese número sobre HateCheck es señal
   real, no vocabulario. Es un criterio mucho más fuerte que el AUC ≥ 0.75 original.

3. **Alimenta FEAT-007 sin trabajo manual.** 442 plantillas × 7 grupos protegidos son
   pares mínimos ya validados por los autores del dataset.

4. **Permite localizar el fallo, no solo medirlo.** El desglose por `functionality`
   dice *qué* tipo de razonamiento falla. Si la dirección latente sube en
   `negate_neg_nh` y `counter_quote_nh`, está capturando composición semántica; si
   solo sube en `slur_h`, sigue leyendo diccionario.

## Enlaces
- complementa: [SOA-001]
- informa: [EXP-001]
- apoya: [Q-001]
- usado_por: [DEC-003]
