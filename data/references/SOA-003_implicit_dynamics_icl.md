---
id: SOA-003
type: state_of_the_art
---

# SOA-003 — Dherin et al.: el contexto es una actualización de pesos implícita

## Source
Dherin, Munn, Mazzawi, Wunder, Gonzalvo (Google Research). *Learning without training:
The implicit dynamics of in-context learning*. arXiv:2507.16003v4, jun 2026.
Local: `data/00_context/in-context-learning.pdf` (24 pp).

## Summary

**Resultado central.** Una pasada hacia delante **con** contexto es matemáticamente
equivalente a una pasada **sin** contexto pero con los pesos del MLP modificados por
una actualización de **rango 1**. Formalizan el "bloque contextual" (capa de
auto-atención + red neuronal) y prueban que el contexto actúa como un ajuste fino de
bajo rango, único y minimizador de la norma de Frobenius — lo llaman *minimal
token-patch*.

**Qué significa "implícito".** No es un análisis mecanicista del cómputo real. La
actualización de pesos **nunca se calcula en el hardware**: la salida es solo
*funcionalmente equivalente* a la de un modelo cuyos pesos hubieran sido actualizados
así. Es equivalencia matemática, no descripción del circuito.

**El puente con steering (Teorema C.2).** Con skip connections, la actualización
implícita tiene **dos partes**: una de bajo rango sobre la matriz, y **una vectorial**.
Los autores dicen literalmente que la segunda *"has strong similarities to the steering
vectors"* (Ilharco 2022, Hendel 2023, Todd 2023), y que su trabajo *"connects steering
vectors and low-rank matrix edits to the implicit mechanisms of the transformer
architecture"*.

## Relevance

1. **Da fundamento teórico a toda la línea de Q-001.** Intervenir sobre activaciones
   sin tocar pesos no es un truco: es operar sobre el mismo mecanismo por el que un
   prompt cambia el comportamiento. La restricción "sin modificar pesos" de [DEC-001]
   deja de ser una limitación autoimpuesta y pasa a ser una elección de nivel de
   análisis.

2. **Predice un experimento que ninguno de los cuatro papers hizo.** Si la parte
   vectorial de la actualización implícita *es* un steering vector, entonces el vector
   de control derivado por contraste (método de [SOA-004]) y el desplazamiento en el
   residual stream inducido por un prompt few-shot deberían **alinearse**. Es una
   predicción falsable con una medida de similitud coseno. Es el bloque B2 de
   [EXP-002].

3. **Marca el límite de lo que se puede afirmar.** Como la equivalencia es funcional y
   no mecanicista, un resultado de alineación alta no demuestra que el modelo "haga"
   gradiente por dentro. Hay que redactar las conclusiones con ese cuidado.

## Enlaces
- fundamenta: [Q-001]
- predice_experimento_en: [EXP-002]
- relacionado: [SOA-004]
