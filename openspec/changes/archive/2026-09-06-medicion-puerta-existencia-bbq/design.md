## Context

Ver `proposal.md — Why` para la motivación y `specs/bbq-existence-gate/spec.md` para los
requisitos.

Restricciones que condicionan el diseño:

- **Es el primer código de aplicación del repositorio.** `src/` no existe. Lo que se
  monte aquí fija el patrón para los changes siguientes (extracción de activaciones,
  vectores de control, guardarraíl), que comparten la misma frontera con el proveedor
  externo.
- `docs/conventions.md §2` describe una estructura **multi-dominio** (`src/shared/` +
  `src/domains/`) pensada para un backend con varios dominios y una API HTTP. Este
  proyecto es un workspace de investigación con un solo dominio y sin API.
  `docs/architecture.md §9` contempla explícitamente empezar en el modelo de capas de un
  único dominio y migrar después.
- `docs/testing.md §2-5` exige **mockear en la frontera del proveedor externo** dejando
  correr la lógica propia. Aquí hay dos proveedores externos: Hugging Face Datasets y
  `transformers`. Ninguno debe aparecer en un test unitario.
- Entorno de `DEC-002`: WSL, un solo dispositivo CUDA, 24 GiB, bf16. Sin multi-GPU.

## Goals / Non-Goals

**Goals**
- Que la lógica de resolución de opciones y de cálculo de métricas sea testeable **sin
  GPU y sin red**. Es donde viven las trampas que corromperían el resultado en silencio.
- Que el adaptador de puntuación sea sustituible, porque los changes siguientes lo
  reutilizan y porque un doble determinista es la única forma de testear las métricas.

**Non-Goals**
- No se generaliza a otros corpus. Dentro de BBQ sí se cubren las 11 categorías: el
  barrido exploratorio mostró que ninguna sola reúne sesgo y masa suficientes
  (`DEC-005` Addendum 1).
- No se abstrae para "cualquier tarea de opción múltiple". Se resuelve BBQ.
- No se intenta rescatar los ítems interseccionales donde ambas opciones comparten el
  grupo declarado. Requerirían decidir qué intersección es la estereotipada, que BBQ no
  anota por esta vía. Se descartan y se cuentan.
- No se optimiza el rendimiento más allá de agrupar en lotes: minutos de GPU no
  justifican complejidad.

## Decisions

### D1 — Capas de un solo dominio, no la estructura multi-dominio de conventions.md

`src/bbq_gate/` con `domain/` (entidades y métricas, sin I/O), `application/` (el caso de
uso) e `infrastructure/` (adaptadores de HF y del modelo).

*Alternativa considerada:* seguir literalmente `conventions.md §2` con
`src/shared/` + `src/domains/bbq_gate/`. Descartada: introduce cuatro niveles de
directorio y una capa `shared/` vacía para un único caso de uso. `architecture.md §9`
autoriza el modelo de un solo dominio y migrar cuando aparezca el segundo. El segundo
dominio real llegará con la extracción de activaciones, y entonces se evalúa la
migración.

*Alternativa considerada:* un único script en `scripts/`, como
`00_lexical_baseline.py`. Descartada: aquel no tiene lógica que pueda fallar en
silencio; éste sí (resolución de la opción unknown), y necesita tests unitarios.

### D2 — El puntuador es un `Protocol` del dominio, implementado en infraestructura

`domain/` declara el contrato "dado un texto y una lista de opciones, devuelve una
puntuación por opción". `infrastructure/` lo implementa con `transformers`.

Es lo que hace testeable todo lo demás: un doble determinista permite verificar el
cálculo de `s_amb`, `rho_unk` y la exactitud sin cargar 15 GiB de pesos. Cumple la
inversión de dependencias de `architecture.md` y la regla de mockeo de `testing.md`.

### D3 — La opción "unknown" se resuelve por anotación, y su ausencia es un error duro

`answer_info` etiqueta cada opción con el grupo que nombra, y marca la de abstención con
la etiqueta `unknown`. Es la única fuente fiable.

*Alternativas descartadas:* posición fija (se reparte 34.0/36.3/29.7 en `Age`) y
coincidencia de texto (hay diez redacciones distintas). Ambas fallan **sin error**, que
es el peor modo de fallo posible en una medición.

Cuando la resolución no es posible, se lanza excepción con el identificador del ítem, no
se descarta el ítem. Descartar en silencio sesgaría la muestra hacia los ítems fáciles.
Sigue `conventions.md §1.3` (fallar rápido y ruidoso).

### D4 — Puntuación por verosimilitud de la letra, con verificación por texto completo

Primario: plantilla que presenta las tres opciones etiquetadas y compara la
verosimilitud de los tokens de letra. Verificación: comparar la verosimilitud del texto
completo de cada opción, normalizada por longitud.

*Por qué el primario es la letra:* un único token por opción, sin necesidad de
normalizar por longitud, que es donde se cuelan artefactos. *Por qué se verifica con el
texto:* si ambos métodos discrepan mucho, el resultado depende del método de puntuación
y hay que decirlo, no elegir el que convenga.

### D5 — Dos formatos de prompt declarados en constantes, no configurables por CLI

Los dos formatos se fijan en código y se ejecutan **siempre los dos**. No se expone una
opción para elegir uno.

Es una decisión deliberadamente rígida: si el formato fuese un parámetro, la tentación
de probar hasta que la puerta abra sería estructural. Añadir un tercer formato exige
editar el código, que deja rastro.

### D6 — Una sola pasada, todas las métricas

Se puntúan los 3680 ítems (`ambig` + `disambig`) en una sola ejecución por formato, y de
ahí salen las tres métricas. Cargar el modelo es lo caro; la inferencia sobre frases de
284 caracteres de media no lo es.

### D7 — La salida es un único JSON con las cifras crudas, no solo el veredicto

Se persisten los recuentos (`n_s`, `n_a`, abstenciones, aciertos en `disambig`) además
de las métricas derivadas, para cada formato y cada orden de opciones.

Permite recalcular una métrica si el criterio cambia, sin repetir la inferencia — y el
criterio ya cambió una vez (Addendum 1 de `hypothesis.md`).

### D8 — El emparejador de roles es una cascada ordenada, no una heurística única

Orden obligatorio: **exacto → guarda de negación → partes del tag compuesto →
subcadena**, y regla dura de que exactamente una de las dos opciones no-abstención debe
casar.

Cada paso existe porque su ausencia deja **cero** ítems utilizables en alguna categoría,
verificado empíricamente:

| heurística sola | dónde muere |
|---|---|
| solo exacta | `Race_ethnicity` 0 % — el tag es `F-Black`, el grupo es `Black` |
| solo subcadena | `Age` 0 % — `'old'` casa dentro de `'nonOld'` y marca al revés |
| sin alias de código | `Gender_identity` 89.6 % — declara `['F']`, anota `['woman']` |

La guarda de negación se aplica **antes** que la subcadena y **después** que la
coincidencia exacta, para que un grupo llamado literalmente `nonbinary` no se
autoexcluya.

*Alternativa considerada:* mantener una tabla manual de grupo→etiquetas por categoría.
Descartada: son 11 categorías con vocabularios distintos, la tabla se desincroniza en
cuanto BBQ cambie, y la cascada cubre los tres modos con una regla general verificable
por tests.

### D9 — La unidad estadística es el ítem, no la observación

Cada ítem produce varias observaciones (permutaciones × formatos). El recuento que
gobierna la puerta y el remuestreo para intervalos operan sobre **ítems**; un ítem se
resuelve por mayoría de sus permutaciones y, si no la hay, se marca inestable.

Sin esto, 150 ítems × 3 permutaciones parecen 450 datos independientes y el intervalo
sale ~1.7× más estrecho de lo que debe. En el barrido exploratorio eso es exactamente la
diferencia entre que el `s_amb` de `Age` excluya el cero o lo incluya.

## Risks / Trade-offs

**El modelo puede estar bien alineado y no producir contraste** → No es un fallo del
change. `hypothesis.md` Addendum 1 tiene el plan B escrito (elección forzada sin opción
de abstención, o ampliar a más categorías). La puerta existe para dar también esta
respuesta.

**Contaminación: BBQ es público desde 2022 y puede estar en el entrenamiento de Qwen
2.5** → No se puede descartar en una PoC. Mitigación: registrarlo como limitación en la
salida y en `results.md`. No se finge que no existe.

**Discrepancia entre puntuar por letra y por texto completo** → D4 obliga a medir ambos.
Si discrepan, el resultado se reporta como dependiente del método, que es información
útil, no un fallo.

**Sesgo de posición confundido con sesgo estereotípico** → Verificado que en `Age` la
opción estereotipada se reparte 33.4/31.7/34.9 entre posiciones, así que el efecto
agregado es pequeño. Aun así se ejecuta con orden permutado y se comparan.

**El disco ext4 de WSL está al 94 %** → Los pesos ocupan ~15 GiB de los 64 GiB libres.
Cabe, pero conviene comprobar espacio antes de la primera descarga y avisar si no
alcanza, en vez de fallar a medias.

**Sobre-abstracción del primer módulo del repositorio** → El riesgo real es diseñar para
casos que no existen. Mitigación: D1 y los Non-Goals acotan a BBQ-Age; la generalización
se hace cuando llegue el segundo consumidor, no antes.

## Migration Plan

No aplica: no hay sistema en producción, ni datos que migrar, ni contrato previo que
romper. Es código nuevo en un repositorio sin `src/`.

Reversión: borrar `src/bbq_gate/`, `scripts/02_bbq_existence_gate.py` y el JSON de
salida. Nada más depende de ellos.

## Open Questions

- **Umbral de exactitud en `disambig`.** `hypothesis.md` fija 0.60. Es un valor razonado
  pero no calibrado contra este modelo. Puede ajustarse tras la primera ejecución sin
  cambiar specs ni tareas, siempre que el cambio se registre como addendum.
- **Número de ítems por lote.** Depende de la memoria libre en el momento; se resuelve
  empíricamente al ejecutar y no afecta a ninguna cifra.
