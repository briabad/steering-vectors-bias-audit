---
id: SOA-001
type: state_of_the_art
---

# SOA-001 — HatEval: señal léxica dominante, shift train→test y contaminación entre splits

## Source
- Dataset: `hs-knowledge/hateval_enriched` en Hugging Face — es HatEval (SemEval-2019
  Task 5, inglés) enriquecido con entity linking contra Google Knowledge Graph
  (columnas `ner_output`, `entities`, no usadas por nosotros).
- Medición propia sobre el dataset completo, ejecutada el 2026-09-05 con el entorno
  de DEC-002 (scikit-learn 1.7.2). Reproducible: TF-IDF (1-2 gramas, `min_df=2`,
  `sublinear_tf`) + regresión logística (`C=1.0`), ajustado **solo** en `train`.

## Summary

### Composición real
| split | n | OK | HATEFUL | ratio hateful |
|-------|-----|------|---------|------|
| train | 9000 | 5217 | 3783 | 0.420 |
| dev   | 1000 | 573  | 427   | 0.427 |
| test  | 3000 | 1740 | 1260  | 0.420 |

Las proporciones de clase son casi idénticas en los tres splits. El problema **no**
es desbalance.

### Hallazgo 1 — un bag-of-words ya supera el criterio de éxito pre-registrado
Ajustando solo TF-IDF + regresión logística, sin ningún modelo neuronal:

| evaluado en | AUC | accuracy | baseline de clase mayoritaria |
|---|---|---|---|
| dev  | **0.8342** | 0.7390 | 0.5730 |
| test | **0.6269** | 0.4870 | 0.5800 |

El criterio 1 de `EXP-001/hypothesis.md` pide **AUC ≥ 0.75** para la dirección
latente. En `dev`, el léxico puro da **0.834**. Es decir: tal como está escrito, el
criterio se cumple sin usar el modelo, y por tanto **no discrimina la hipótesis**.

### Hallazgo 2 — shift train→test brutal
La misma función pasa de AUC 0.834 en `dev` a **0.627** en `test`, y su accuracy
(0.487) cae **por debajo** del baseline de clase mayoritaria (0.580). Es la anomalía
conocida de HatEval: el test set se construyó con una distribución distinta a la de
train/dev, y en la competición original muchos sistemas quedaron por debajo del MFC.

Consecuencia directa: **el split de evaluación cambia la conclusión en 0.21 de AUC**.
Cualquier número que reportemos sin decir sobre qué split se midió es ininterpretable.

### Hallazgo 3 — contaminación entre splits
- `train ∩ test` = **225 tweets con texto exactamente idéntico**.
- `train ∩ dev` = 0, `dev ∩ test` = 0.
- `test` tiene además 28 duplicados internos (3000 filas, 2972 textos únicos).

### Hallazgo 4 — la señal léxica es un artefacto de campaña, no de semántica
Coeficientes más fuertes hacia HATEFUL:

```
bitch(8.04)  buildthatwall(7.28)  womensuck(4.86)  illegal(3.91)
buildthewall(3.38)  whore(3.35)  maga(3.19)  bitches(3.18)  hoe(3.10)
nodaca(2.94)  illegals(2.91)  her(2.40)  europe(2.25)  she(2.15)
stoptheinvasion(2.11)
```

Son hashtags de campaña política e insultos de diccionario. Aparecen incluso los
pronombres `her` y `she` — la mitad "misoginia" de HatEval hace que el género
gramatical solo ya cargue señal. Ejemplo real etiquetado `HATEFUL` en train:

> `Hurray, saving us $$$ in so many ways @potus @realDonaldTrump #LockThemUp #BuildTheWall #EndDACA #BoycottNFL #BoycottNike`

No hay insulto: la etiqueta la sostienen los hashtags.

## Relevance

Impacta las cuatro piezas centrales de la línea de investigación:

1. **Invalida el criterio de éxito de EXP-001 tal como está escrito.** "AUC ≥ 0.75"
   debe sustituirse por una afirmación relativa: *la dirección latente supera al
   baseline léxico sobre el mismo split*. Ver addendum en `EXP-001/hypothesis.md`.

2. **Un vector de diferencia de medias capturará esto de forma trivial.** Los
   embeddings de `#BuildTheWall` y `bitch` van a dominar µ_hate − µ_neutral. Un AUC
   alto en `dev` sería, por defecto, evidencia de que el modelo lee hashtags — no de
   que exista una firma latente de sesgo.

3. **Los pares contrafactuales (FEAT-007) dejan de ser validación opcional y pasan a
   ser el control principal.** Son la única forma de separar "el modelo detecta la
   palabra" de "el modelo detecta el sesgo".

4. **Obliga a fijar el protocolo de evaluación antes de mirar nada:** `dev` es el
   split de validación (limpio, sin solape); `test` se reserva y, si se usa, hay que
   deduplicar los 225 solapes con train y reportar los dos números por separado.

## Enlaces
- informa: [EXP-001]
- apoya: [Q-001]
- relacionado: [DEC-001]
