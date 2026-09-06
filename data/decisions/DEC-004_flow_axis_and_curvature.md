---
id: DEC-004
type: decision
supersede: null
---

# DEC-004 — El eje del flujo es el paso de razonamiento; la curvatura es de Menger

## Trigger
`Q-001` y `EXP-001` hablan de velocidad, curvatura y continuidad de una trayectoria
latente $h_t$ sin decir nunca sobre qué eje corre $t$. Había tres candidatos
incompatibles — capa, token, paso de razonamiento — y sin elegir uno la geometría no se
puede implementar. Era el único bloqueo conceptual abierto (FEAT-006, `blocked`).

En paralelo, `scripts/01_hateval_latent_pipeline.py:108` calcula la curvatura con
`np.cross`, que solo admite vectores de 2 o 3 dimensiones. Con los 3584-d de Qwen falla,
y numpy 2.2.6 (el instalado, ver [DEC-002]) ya eliminó incluso el cross de 2-d.

## Alternativas consideradas
- **A — $t$ = capa.** La trayectoria del residual stream a través de las 28 capas.
  Barata: sale gratis de `output_hidden_states=True`, sin generación. Bien definida y
  comparable entre ejemplos de longitud distinta.
- **B — $t$ = token.** A lo largo de la secuencia, en una capa fija. Longitud variable,
  difícil de comparar entre ejemplos.
- **C — $t$ = paso de razonamiento.** La cadena de pensamiento generada, segmentada en
  pasos. Es lo que hace [SOA-006].

## Decisión

**Eje primario: C, el paso de razonamiento.** Es el eje de [SOA-006], y su afirmación
teórica — *"logical statements act as local controllers of these flows' velocities"* —
solo tiene sentido sobre pasos de razonamiento, no sobre capas. Como BBQ es QA de un
turno, hay que **generar** la cadena y segmentarla.

**Eje secundario: A, la capa.** Se calcula igualmente porque sale gratis de las mismas
activaciones que ya extraemos para el vector de control. No se le atribuye la
interpretación de "razonamiento": es una descripción del cómputo interno, y así se
redacta.

**Se descarta B** como eje principal. Queda disponible como diagnóstico.

**Operador de representación.** Siguiendo la Def. 3.2 de [SOA-006], $\mathcal{E}$ se
instancia extrayendo hidden states del propio Qwen 2.5 7B en la posición del **último
token** de cada paso, en la capa seleccionada — coherente con [SOA-004], que extrae en
último token. No se introduce un encoder externo: añadiría un modelo más que auditar.

**Curvatura: la de Menger** (Def. 3.4 de [SOA-006]), para tres puntos consecutivos:

$$\kappa(x_1,x_2,x_3) = \frac{4A}{\lVert x_1-x_2\rVert\,\lVert x_2-x_3\rVert\,\lVert x_3-x_1\rVert}$$

con $A$ por la fórmula de Herón sobre las tres distancias. Se calcula **solo con
distancias**, vale en dimensión arbitraria y no necesita producto vectorial.
`np.cross` queda prohibido en este repositorio para métricas de flujo.

**Normalizaciones obligatorias.** Toda métrica geométrica se reporta en dos variantes:
sobre $h_t$ crudo y sobre $\hat h_t = h_t/\lVert h_t\rVert$. Y toda comparación entre
clases se normaliza por **longitud de trayectoria** (número de pasos): una cadena más
larga tiene más ocasión de curvarse, y sin ese control se mide verbosidad.

## Rationale
- El paper que define las métricas define también el eje. Usar sus fórmulas sobre otro
  eje sería citarlo mal.
- El eje capa se queda porque su coste marginal es cero y da una segunda lectura, pero
  no puede llamarse "flujo de razonamiento" sin abuso de lenguaje.
- Menger no es una alternativa entre varias: es la única de las consideradas que está
  definida en $\mathbb{R}^{3584}$ sin proyectar antes.
- Las dos normalizaciones no son celo metodológico. En [SOA-004] los autores tuvieron
  que reescalar los vectores de PCA por la norma real de las activaciones; es el mismo
  fenómeno, y aquí sesgaría la comparación entre clases.

## Coste que introduce
Generar CoT multiplica el coste del bloque de geometría respecto al resto: hay que
generar (no solo una pasada hacia delante), y la generación mete varianza de muestreo.
Por eso el bloque de geometría **no** entra en la PoC: ver [DEC-005].

## Predicción
Si la hipótesis de [Q-001] es correcta, sobre BBQ con CoT generada:
- las trayectorias que terminan en respuesta estereotipada muestran curvatura media de
  Menger distinta de las que terminan en "unknown", con tamaño de efecto no trivial
  tras normalizar por longitud;
- la diferencia sobrevive al control de [SOA-006] — misma plantilla, distinto grupo
  protegido.

Si la diferencia desaparece al normalizar por longitud, la señal era verbosidad.

---

## Addendum 1 (2026-09-05) — la geometría agregada por token NO separa pares mínimos lógicos

> Sonda `scripts/04_flow_geometry_probe.py`, salida en
> `data/experiments/EXP-001_hateval_vector_flow/output/flow_geometry_probe.json`.
> **Resultado negativo, y anecdótico**: 4 pares, una sola capa.

### Qué se probó

Si la tesis de [SOA-006] es correcta — *"logical statements act as local controllers of
these flows' velocities"* — entonces un par mínimo que comparta vocabulario y difiera
solo en estructura lógica debería mostrar geometría distinta. Es el **dual** del diseño
de Zhou et al.: ellos fijan la lógica y varían la semántica; aquí se fija la semántica y
se varía la lógica.

Pares tomados de HateCheck: aserción vs negación, aserción vs cita atribuida, negación
de positivo, y slur real vs homónimo.

### Resultado

Sobre el único par con **longitud idéntica** en ambos lados (T=12):

```
"I have no compassion at all for any woman like you."   hateful
"I have a lot of compassion for any woman like you."    non-hateful

razón hateful/non-hateful (normalizado):
    velocidad = 1.010    curvatura = 1.001    curvatura/arco = 0.991
```

**Indistinguibles.** El par slur/homónimo (T=12 vs 11) da razones entre 0.915 y 1.060,
igualmente planas.

Los pares 1 y 2 sí muestran diferencias aparentes (velocidad ×1.16 y ×1.17 normalizada,
×3.09 y ×3.28 en crudo), pero sus longitudes son 4 vs 11 y 5 vs 15 tokens. **Es artefacto
de longitud**, exactamente el confundidor que esta decisión ya obligaba a controlar.

Confirmado de paso el requisito de doble normalización: en crudo la curvatura sale ~0.02
y normalizada ~1.8 — casi dos órdenes de magnitud. Sin normalizar se mide la escala de
activación, no la forma.

### Qué se concluye, y qué NO

**Se descarta**: el eje token con **estadísticos agregados sobre la frase entera** como
método para separar estructura lógica. No sirve.

**No se descarta la hipótesis de [SOA-006]**, por tres razones:

1. n = 4 pares, capa 14 de 28. Es anecdótico.
2. El eje de Zhou es el **paso de razonamiento**, no el token. Se probó una versión
   degradada de su método.
3. **Promediar destruye la señal local**, y esta es la razón de peso. Si la negación
   perturba la trayectoria, lo hace **en la posición de `don't`**: uno o dos pasos.
   Promediar sobre 12 los diluye por un factor ~10. El instrumento estaba mal elegido.

### Consecuencia para FEAT-017

Cuando se aborde el bloque de geometría, la comparación debe ser **posición a posición
con trayectorias alineadas** — localizar dónde divergen — y no por estadístico agregado.
La media sobre la frase queda desaconsejada explícitamente.

## Enlaces
- motivated_by: [Q-001]
- apoyado_en: [SOA-006, SOA-004]
- desbloquea: [EXP-001]
- supersede: null
