# bbq-existence-gate Specification

## Purpose
Decidir, mediante una única pasada de inferencia sin gradientes, si un modelo causal
exhibe sesgo estereotípico direccional medible en preguntas ambiguas de BBQ. Es la
puerta que determina si existe un contraste conductual sobre el que aprender una
dirección latente, antes de invertir en extracción de activaciones.

## Requirements

### Requirement: Resolución semántica de la opción "no se puede determinar"

El sistema SHALL identificar la opción de respuesta "unknown" de cada ítem de BBQ a
partir de sus metadatos de anotación, y NUNCA por su posición entre las opciones ni por
coincidencia de su texto.

Esto es normativo, no un detalle interno: la opción "unknown" se reparte entre las tres
posiciones (34.0 % / 36.3 % / 29.7 % en la categoría `Age`) y aparece con diez
redacciones distintas. Resolverla por posición o por texto produce métricas corruptas
sin emitir ningún error, lo que invalidaría silenciosamente todo el experimento.

#### Scenario: La opción unknown no está en la última posición
- **WHEN** se procesa un ítem cuya opción "unknown" está en `ans0` o `ans1`
- **THEN** el sistema la identifica correctamente como la opción de abstención
- **AND** no marca `ans2` como abstención

#### Scenario: Redacción no vista de la abstención
- **WHEN** un ítem expresa la abstención con una redacción distinta de las conocidas
  (p. ej. `Not answerable` frente a `Cannot be determined`)
- **THEN** el sistema la identifica igualmente, porque se apoya en la anotación y no en
  el texto

#### Scenario: Ítem sin opción de abstención identificable
- **WHEN** un ítem no permite identificar ninguna opción de abstención
- **THEN** el sistema falla de forma ruidosa indicando el identificador del ítem
- **AND** no lo incluye silenciosamente en el cálculo de métricas

### Requirement: Identificación de la opción estereotipada

El sistema SHALL determinar, para cada ítem ambiguo, cuál de las opciones no-abstención
corresponde al grupo señalado como estereotipado por los metadatos del ítem, y cuál
corresponde al grupo contrario. La correspondencia entre el grupo declarado y la
anotación de la opción SHALL resolverse de forma que sobreviva a las tres
discrepancias observadas en el corpus, y un ítem cuya asignación resulte ambigua
SHALL descartarse explícitamente, nunca asignarse por defecto.

Cada una de estas discrepancias, tratada ingenuamente, deja **cero** ítems utilizables
en al menos una categoría, y lo hace sin emitir ningún error.

#### Scenario: Clasificación de las tres opciones
- **WHEN** se procesa un ítem ambiguo utilizable
- **THEN** el sistema clasifica sus tres opciones en exactamente: abstención,
  estereotipada y anti-estereotipada

#### Scenario: La anotación es un tag compuesto
- **WHEN** el grupo declarado es un componente de la anotación de la opción (por
  ejemplo, un grupo racial dentro de una anotación que combina género y raza)
- **THEN** el sistema reconoce la correspondencia

#### Scenario: La anotación niega el grupo declarado
- **WHEN** la anotación de una opción es la forma negada del grupo declarado (por
  ejemplo, la anotación de "no perteneciente al grupo" frente al grupo)
- **THEN** el sistema NO la clasifica como estereotipada
- **AND** la clasifica como anti-estereotipada

#### Scenario: El grupo se declara con un código y se anota con una palabra
- **WHEN** el grupo declarado es un código abreviado y la anotación de la opción usa el
  término textual equivalente
- **THEN** el sistema reconoce la correspondencia

#### Scenario: Ambas opciones comparten el grupo declarado
- **WHEN** las dos opciones no-abstención corresponden al grupo declarado, porque el
  contraste real del ítem va por un eje distinto del declarado
- **THEN** el sistema descarta el ítem como no utilizable
- **AND** lo contabiliza y lo reporta, sin asignar un rol por defecto

#### Scenario: Ninguna opción corresponde al grupo declarado
- **WHEN** ninguna de las dos opciones no-abstención corresponde al grupo declarado
- **THEN** el sistema descarta el ítem como no utilizable
- **AND** lo contabiliza por separado de los descartes por coincidencia doble

### Requirement: Cobertura declarada por categoría

El sistema SHALL reportar, para cada categoría del corpus, cuántos ítems resultaron
utilizables y cuántos se descartaron por cada motivo.

Sin esta cifra no se puede distinguir "el modelo no tiene sesgo" de "el emparejador
descartó el 70 % del corpus".

#### Scenario: Informe de cobertura
- **WHEN** termina la carga del corpus
- **THEN** el sistema expone por categoría el total de ítems, los utilizables y el
  desglose de descartes por motivo

### Requirement: Puntuación por logits sobre opciones restringidas

El sistema SHALL determinar la respuesta del modelo comparando las verosimilitudes que
asigna a un conjunto cerrado de opciones, y NO mediante generación libre de texto.

La generación libre introduce fallos de parseo por prefacios, rechazos y respuestas
evasivas propias de un modelo alineado por instrucciones, que se confundirían con
ausencia de sesgo.

#### Scenario: Selección de la respuesta
- **WHEN** se evalúa un ítem
- **THEN** la respuesta elegida es la opción con mayor verosimilitud entre las tres
- **AND** el sistema registra la puntuación de las tres, no solo la ganadora

#### Scenario: El modelo tiende a evadir la pregunta
- **WHEN** el modelo, en generación libre, produciría un rechazo o una evasiva
- **THEN** la puntuación restringida devuelve igualmente una opción elegida
- **AND** el resultado no queda sin clasificar

### Requirement: Métricas separadas de ceguera y de sesgo direccional

El sistema SHALL calcular por separado la proporción de ítems ambiguos en que el modelo
no se abstiene (`rho_unk`) y el sesgo direccional entre las respuestas no-abstención
(`s_amb`), y NO SHALL combinarlas en una única cifra.

Son fenómenos distintos: un modelo que elige al azar entre los dos grupos produce
`rho_unk` alto con `s_amb` nulo — no capta la ambigüedad, pero no tiene estereotipo. Una
sola cifra abriría la puerta por el motivo equivocado y el resto del experimento mediría
otra cosa.

#### Scenario: Ceguera sin sesgo
- **WHEN** el modelo elige la opción estereotipada y la anti-estereotipada en
  proporciones aproximadamente iguales
- **THEN** `rho_unk` es alto y `s_amb` es próximo a cero
- **AND** el sistema informa de que no hay contraste de estereotipo utilizable

#### Scenario: Sesgo direccional presente
- **WHEN** el modelo elige la opción estereotipada mucho más que la anti-estereotipada
- **THEN** `s_amb` es sustancialmente positivo
- **AND** el sistema informa del número absoluto de respuestas estereotipadas
  disponibles como contraste

#### Scenario: Modelo bien alineado
- **WHEN** el modelo se abstiene en casi todos los ítems ambiguos
- **THEN** `rho_unk` es bajo
- **AND** el sistema informa de que el contraste conductual no es viable en esta
  categoría

### Requirement: Conteo en ítems distintos y agrupación por ítem

El sistema SHALL expresar los recuentos que gobiernan la puerta en **ítems distintos**,
no en observaciones, y SHALL estimar la incertidumbre de las métricas agrupando por
ítem.

Cada ítem se evalúa varias veces (una por permutación de opciones y por formato de
prompt). Tratar esas repeticiones como independientes exagera la precisión y puede abrir
la puerta con un intervalo que en realidad incluye el cero.

#### Scenario: Un ítem con respuestas discrepantes entre permutaciones
- **WHEN** un mismo ítem recibe roles distintos según la permutación
- **THEN** el sistema lo clasifica por la mayoría de sus permutaciones
- **AND** si no hay mayoría, lo marca como inestable y lo reporta aparte

#### Scenario: Estimación de incertidumbre
- **WHEN** se reporta el sesgo direccional
- **THEN** su intervalo de confianza se obtiene remuestreando ítems, no observaciones

### Requirement: Métricas por categoría además de agregadas

El sistema SHALL reportar el sesgo direccional tanto agregado sobre todo el corpus como
desglosado por categoría.

El signo del sesgo no es homogéneo entre ejes: agregarlo sin desglosar ocultaría que en
unos ejes el modelo falla hacia el estereotipo y en otros en contra, que es un resultado
en sí mismo.

#### Scenario: Signos opuestos entre categorías
- **WHEN** unas categorías presentan sesgo direccional positivo y otras negativo
- **THEN** el informe expone ambos grupos por separado además del valor agregado
- **AND** el valor agregado no se presenta como si describiera a todas por igual

#### Scenario: Una categoría no supera el control de comprensión
- **WHEN** la exactitud sobre ítems desambiguados de una categoría queda bajo el umbral
- **THEN** esa categoría se excluye del agregado
- **AND** la exclusión se reporta explícitamente

### Requirement: Control de comprensión de la tarea

El sistema SHALL medir la exactitud del modelo sobre los ítems desambiguados, donde la
evidencia es explícita, y SHALL declarar el resultado no interpretable si esa exactitud
no supera un umbral configurado.

Sin este control, un modelo que simplemente no entiende el formato produciría métricas
de sesgo que parecen informativas y no lo son.

#### Scenario: El modelo no resuelve los casos con evidencia
- **WHEN** la exactitud sobre ítems desambiguados queda por debajo del umbral
- **THEN** el sistema marca el resultado como no interpretable
- **AND** indica que el problema está en el formato del prompt, no en la ausencia de
  sesgo

#### Scenario: El modelo comprende la tarea
- **WHEN** la exactitud sobre ítems desambiguados supera el umbral
- **THEN** las métricas sobre ítems ambiguos se declaran interpretables

### Requirement: Robustez frente al orden de las opciones

El sistema SHALL permitir evaluar el mismo conjunto de ítems con el orden de las
opciones permutado, y SHALL reportar las métricas de ambas presentaciones.

#### Scenario: Comparación entre órdenes
- **WHEN** se evalúa con el orden original y con el orden permutado
- **THEN** el sistema reporta `s_amb` para ambos
- **AND** expone la diferencia, de modo que un sesgo de posición sea distinguible de un
  sesgo estereotípico

### Requirement: Pluralidad de formatos de prompt fijados de antemano

El sistema SHALL evaluar con al menos dos formatos de prompt declarados antes de la
ejecución, y SHALL reportar las métricas de todos ellos.

Reportar solo el formato más favorable equivaldría a ajustar la propia puerta al
resultado deseado.

#### Scenario: Los formatos discrepan
- **WHEN** dos formatos producen valores de `s_amb` distintos
- **THEN** el sistema reporta ambos sin descartar ninguno
- **AND** señala cuál es el más desfavorable

### Requirement: Reproducibilidad y trazabilidad del resultado

El sistema SHALL persistir el resultado junto a la semilla, el identificador y revisión
del modelo, la configuración del corpus, los formatos de prompt empleados y las
versiones de las bibliotecas relevantes.

#### Scenario: Reejecución
- **WHEN** se vuelve a ejecutar con la misma semilla y configuración
- **THEN** las métricas obtenidas son idénticas

#### Scenario: Auditoría posterior
- **WHEN** alguien abre el artefacto de salida sin acceso a la conversación original
- **THEN** puede determinar qué modelo, qué corpus y qué prompts produjeron cada cifra

### Requirement: No modificación del modelo

El sistema SHALL operar exclusivamente en inferencia y NO SHALL modificar, ajustar ni
persistir ningún peso del modelo.

#### Scenario: Ejecución completa
- **WHEN** termina la medición
- **THEN** los pesos del modelo son idénticos a los cargados
- **AND** no se ha escrito ningún checkpoint
