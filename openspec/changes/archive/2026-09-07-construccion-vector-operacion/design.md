## Context

Ver `proposal.md — Why` para la motivación y `specs/control-vector/spec.md` para los
requisitos. El diseño del contraste está pre-registrado en
`data/experiments/EXP-002_bbq_stereotype_direction/hypothesis.md`, Addenda 3, 4 y 5.

Restricciones que condicionan el diseño:

- **El contraste no requiere inferencia nueva para decidir quién va a cada lado.** Las
  287.976 observaciones de `EXP-002/output/existence_gate_raw.jsonl` ya dicen qué eligió
  el modelo en cada ítem. Solo hace falta una pasada adicional para extraer activaciones
  de los ~1500 ítems seleccionados.
- El código de `bbq-existence-gate` ya provee carga de BBQ, resolución de roles y
  consolidación por mayoría. Se reutiliza, no se duplica.
- Entorno de `DEC-002`: un dispositivo CUDA, 24 GiB, bf16.
- `docs/testing.md §2-5`: mockear en la frontera del proveedor externo. El extractor de
  activaciones es esa frontera.

## Goals / Non-Goals

**Goals**
- Que el álgebra de direcciones y el emparejamiento sean testeables **sin GPU**: ahí
  viven los errores que producirían un vector plausible pero equivocado.
- Persistir las activaciones para que las tres construcciones, la curva por capa y el
  protocolo del residuo salgan de **una sola extracción**.

**Non-Goals**
- No se interviene sobre el modelo. Ningún $\alpha$, ninguna suma al residual stream.
- No se prueba transferencia (H3). Las categorías reservadas solo se declaran y se
  aíslan.
- No se generaliza a otros corpus ni a otros modelos.

## Decisions

### D1 — El contraste se deriva del crudo ya medido, no de una nueva pasada

`existence_gate_raw.jsonl` contiene, por observación, el rol elegido. La consolidación
por mayoría de permutaciones ya está implementada y verificada. De ahí sale la partición
$P^+/P^-$ sin volver a ejecutar las 2 horas.

*Alternativa descartada:* reejecutar la evaluación conjuntamente con la extracción.
Duplicaría el coste sin aportar nada: el rol elegido no depende de qué activaciones
guardemos.

**Advertencia heredada**: el campo `item_id` del crudo contiene el `question_index`, no
un identificador único. Está corregido en el loader, pero **el fichero crudo existente
conserva el defecto**. La identidad se reconstruye por posición, con verificación del
ciclo de permutaciones que aborta si el patrón no se cumple — el mismo mecanismo que
`scripts/05_recompute_gate.py`, que debe reutilizarse en lugar de reimplementarse.

### D2 — Emparejamiento por categoría, plantilla **y polaridad**

> **Corregido el 2026-09-06 tras el aborto por fuga.** La versión original de esta
> decisión emparejaba por `(categoría, question_index)` afirmando que eso controlaba el
> vocabulario de la pregunta. **Era falso**: las 343 plantillas del corpus, sin
> excepción, contienen los dos textos de pregunta correspondientes a las polaridades
> `neg` y `nonneg`, que son preguntas opuestas. Con esa clave un TF-IDF alcanzó AUC
> 0.9079 frente a 0.9174 de la dirección, y la tarea 4.6 abortó. Ver
> `EXP-002/hypothesis.md` Addendum 6.

La clave es `(categoría, question_index, question_polarity)`. Deja solo 7 de 686 grupos
(1.0 %) con más de un texto de pregunta; para ese residuo se exige además **texto de
pregunta idéntico**.

Emparejar así controla la categoría **y** la redacción de la pregunta. Coste medido:
564 pares en `chat` (frente a 752 con la clave defectuosa) y 413 en `plano`.

**Lo que sigue sin controlar, declarado**: el contexto difiere en las personas nombradas
—*"un nieto y su abuelo"* vs *"una abuela y su nieta"*—. Un baseline léxico puede
aprender qué parejas hacen que el modelo adivine, y eso es intrínseco: la conducta del
modelo depende de a quién se nombra. La expectativa registrada es que el baseline baje
sustancialmente pero no a 0.5, y que C2 se juzgue por el **margen**, no por su valor
absoluto.

*Alternativa considerada:* estratificación proporcional por categoría. Descartada: iguala
la mezcla temática pero no el vocabulario, y la categoría más escasa la limitaría a 176
ítems por lado.

*Alternativa considerada:* estratificación proporcional por categoría. Descartada: iguala
la mezcla temática pero no el vocabulario, y la categoría más escasa la limitaría a 176
ítems por lado.

Medido sobre la corrida real (`chat`): 752 pares de 1063 posibles, 70.7 %, con las 11
categorías representadas.

### D3 — El residuo se decide midiendo, no eligiendo

Los 311 sin pareja proceden de 11 plantillas de 81: aquellas donde el modelo casi nunca
se abstiene, que son las de los estereotipos más marcados. Descartarlos dejaría el vector
construido con la mitad más tibia del fenómeno.

En vez de decidir a priori, se construyen dos direcciones sobre conjuntos **disjuntos**
—$v_A$ con los 752 pares, $v_B$ con los 311 emparejados por categoría— y se mide su
coseno contra dos anclas:

- **suelo**: coseno esperado entre direcciones aleatorias en $\mathbb{R}^{3584}$, ≈ 0 con
  desviación $1/\sqrt{3584} \approx 0.017$;
- **techo**: coseno entre direcciones construidas con dos mitades disjuntas de los 752,
  que mide la similitud alcanzable por dos muestras del mismo fenómeno a este tamaño.

Umbral declarado: **80 % del techo** para fusionar.

*Por qué disjuntos:* comparar $v_{752}$ con $v_{1063}$ sería inválido — comparten el 71 %
de los ítems y el coseno alto saldría garantizado.

*Límite reconocido:* el coseno dice **si** difieren, no cuál es mejor. El desempate, si
hace falta, es el desempeño sobre las categorías reservadas, y eso pertenece al change de
transferencia.

### D4 — Todas las capas se extraen una vez y se persisten

~1500 ítems × 29 capas × 3584 dims × 2 bytes ≈ **310 MB**. Cabe holgadamente y evita
repetir la extracción para la curva por capa, las tres construcciones y el protocolo del
residuo.

*Alternativa descartada:* extraer solo la capa media. Impediría producir la curva que el
spec exige y obligaría a repetir la pasada.

### D5 — El extractor es un `Protocol` del dominio

Igual que el puntuador del change anterior. El álgebra de direcciones —construcción,
normalización, proyección, coseno, permutación— se testea con activaciones sintéticas y
sin cargar 15 GiB de pesos.

### D6 — El aislamiento de las categorías reservadas se impone en el tipo, no en la disciplina

El cargador del contraste devuelve dos estructuras separadas, y la de transferencia **no
se pasa** a ninguna función de ajuste. Que el aislamiento dependa de recordar filtrar es
justo el tipo de error que este proyecto ya ha cometido.

## Risks / Trade-offs

**El vector puede codificar tema pese al emparejamiento** → El emparejamiento iguala
categoría y plantilla, pero no las personas nombradas. Mitigación: reportar la
composición resultante y, en el change de transferencia, el probe de categoría de C8.

**Selección hacia plantillas donde el modelo a veces se abstiene** → Al emparejar se
descartan preferentemente las plantillas de sesgo más fuerte, que no dejan negativos.
Mitigación: D3 lo cuantifica y, si el residuo no fusiona, pasa a conjunto de evaluación.
Se declara como limitación en `results.md`.

**AUC alto por memorización del corpus** → BBQ es público desde 2022. No es descartable;
se declara.

**El crudo existente tiene el identificador defectuoso** → Mitigado por reconstrucción
posicional verificada (D1). Si el patrón falla, el proceso aborta en lugar de producir
cifras silenciosamente erróneas.

**Un AUC bajo se lea como fallo del change** → Es un resultado válido de H1. Las tareas
deben redactar el veredicto sin interpretarlo.

## Migration Plan

No aplica: código nuevo que extiende `src/bbq_gate/` sin romper contratos existentes.
Reversión: borrar los módulos nuevos, `scripts/06_build_operation_vector.py` y los
artefactos de salida.

## Open Questions

- **Política de agregación sobre tokens.** Se fija el último token, siguiendo `SOA-004`.
  Media sobre tokens es una alternativa razonable que podría explorarse después sin
  cambiar specs ni tareas.
- **Número de permutaciones del test.** 1000 es el valor por defecto; ajustable según
  coste observado, sin efecto sobre el diseño.
