# control-vector Specification

## Purpose
Construir direcciones latentes en el residual stream de un modelo causal a partir de un
contraste conductual emparejado, y determinar si esa dirección separa las clases del
contraste por encima de baselines declarados, sin modificar los pesos del modelo.

## Requirements

### Requirement: Contraste emparejado 1:1 con el enunciado controlado

El sistema SHALL emparejar cada ítem del conjunto positivo con exactamente un ítem del
conjunto negativo que comparta **categoría, plantilla y polaridad de la pregunta**, de
modo que ambos miembros del par presenten **el mismo enunciado**, y SHALL descartar los
ítems negativos no emparejados.

Compartir plantilla no basta: una misma plantilla contiene los enunciados de ambas
polaridades, que son preguntas opuestas con vocabulario distinto. Emparejar sin controlar
la polaridad permite que un clasificador léxico aprenda el enunciado como sustituto de la
etiqueta.

#### Scenario: Enunciados opuestos dentro de la misma plantilla
- **WHEN** dos ítems comparten categoría y plantilla pero difieren en polaridad
- **THEN** el sistema NO los empareja entre sí

#### Scenario: Plantilla y polaridad no fijan el enunciado
- **WHEN** un grupo de categoría, plantilla y polaridad contiene más de un enunciado
- **THEN** el emparejamiento exige además que el enunciado sea idéntico

La composición de los dos conjuntos debe quedar igualada **por construcción**, no por
muestreo. Un contraste sin emparejar reproduce el desequilibrio de $\rho_{\text{unk}}$
entre ejes —un factor 60— y la diferencia de medias codificaría el tema en lugar de la
operación.

#### Scenario: Plantilla con negativos suficientes
- **WHEN** una plantilla tiene tantos o más ítems negativos que positivos
- **THEN** todos sus ítems positivos quedan emparejados
- **AND** los ítems negativos sobrantes se descartan

#### Scenario: Plantilla con negativos insuficientes
- **WHEN** una plantilla tiene menos ítems negativos que positivos
- **THEN** se forman tantas parejas como ítems negativos haya
- **AND** los ítems positivos restantes se registran como residuo, contabilizados por
  plantilla

#### Scenario: Plantilla sin ningún ítem negativo
- **WHEN** una plantilla no tiene ítems negativos
- **THEN** no se forma ninguna pareja
- **AND** todos sus ítems positivos pasan al residuo

#### Scenario: Composición resultante
- **WHEN** termina el emparejamiento
- **THEN** la distribución por categoría de los dos conjuntos es idéntica
- **AND** el sistema expone esa distribución para su verificación

### Requirement: Extracción de activaciones sin modificar el modelo

El sistema SHALL capturar el estado oculto de todas las capas en la posición del último
token de cada ejemplo, operando exclusivamente en inferencia.

#### Scenario: Extracción completa
- **WHEN** se procesa un ejemplo
- **THEN** se obtiene un vector por cada capa, incluida la de embeddings
- **AND** cada vector corresponde a la posición del último token

#### Scenario: Integridad del modelo
- **WHEN** termina la extracción
- **THEN** los pesos del modelo son idénticos a los cargados
- **AND** no se ha escrito ningún checkpoint

#### Scenario: Trazabilidad de la salida
- **WHEN** se persisten las activaciones
- **THEN** el artefacto registra modelo, revisión, precisión, política de posición y
  semilla

### Requirement: Tres construcciones de la dirección con escala comparable

El sistema SHALL derivar la dirección por las tres vías de la literatura de referencia
—media simple, media de diferencias emparejadas, y primera componente principal de esas
diferencias— y SHALL reescalar las que salgan normalizadas según la norma real de las
activaciones, de modo que su escala sea comparable con las demás.

Sin ese reescalado, el parámetro de intervención no significa lo mismo entre métodos.

#### Scenario: Comparabilidad de escala
- **WHEN** se construyen las tres direcciones sobre el mismo contraste
- **THEN** sus normas son del mismo orden de magnitud
- **AND** el factor de reescalado aplicado queda registrado

#### Scenario: Diferencias emparejadas
- **WHEN** se emplea una construcción basada en contraste
- **THEN** las diferencias se calculan **por pareja**, no entre medias de conjuntos sueltos

### Requirement: Selección de capa sin contaminar la evaluación

El sistema SHALL elegir la capa usando un subconjunto reservado del conjunto de ajuste, y
SHALL congelar esa elección antes de evaluar sobre el holdout.

Elegir la mejor de las capas disponibles mirando el holdout son tantas comparaciones como
capas, y el resultado quedaría inflado por selección.

#### Scenario: Orden de las decisiones
- **WHEN** se selecciona la capa
- **THEN** la decisión usa solo datos del conjunto de ajuste
- **AND** el holdout no se consulta hasta que la capa está fijada

#### Scenario: Registro de la curva
- **WHEN** se reporta el resultado
- **THEN** se expone el rendimiento en todas las capas evaluadas, no solo en la elegida

### Requirement: Evaluación relativa a baselines declarados

El sistema SHALL comparar la capacidad discriminativa de la dirección contra un baseline
léxico y contra la capa de embeddings, sobre el mismo holdout, y NO SHALL presentar una
cifra absoluta como evidencia por sí sola.

La capa de embeddings no ha atravesado ningún bloque del modelo: si una capa intermedia
no la supera, la separación no involucra cómputo del modelo.

El baseline de embeddings SHALL calcularse con **agregación sobre todos los tokens del
prompt**, no sobre la última posición. Bajo una plantilla de conversación el último token
es un marcador de turno idéntico en todos los ejemplos, lo que haría el baseline
constante y por tanto vacuo por construcción.

#### Scenario: Baseline de embeddings no degenerado
- **WHEN** se calcula el baseline de la capa de embeddings
- **THEN** su valor varía entre ejemplos
- **AND** si resultara constante, el sistema lo señala como baseline inválido en vez de
  reportarlo como capacidad discriminativa nula

#### Scenario: Comparación obligatoria
- **WHEN** se reporta la capacidad discriminativa de la dirección
- **THEN** se reportan junto a ella las de ambos baselines sobre el mismo holdout

#### Scenario: Baseline inesperadamente fuerte
- **WHEN** un baseline alcanza una capacidad discriminativa alta
- **THEN** el sistema lo señala como posible fuga en el diseño del contraste
- **AND** no presenta la superioridad de la dirección como establecida

### Requirement: Significancia por permutación con reajuste

El sistema SHALL estimar la significancia permutando las etiquetas y **reconstruyendo la
dirección dentro de cada permutación**, y SHALL acompañar todo valor de significancia de
un tamaño de efecto y un intervalo de confianza obtenido remuestreando ítems.

Permutar sin reconstruir la dirección contrasta una hipótesis nula distinta de la que
interesa, y devuelve significancia siempre.

#### Scenario: Nulo correcto
- **WHEN** se ejecuta el test de permutación
- **THEN** la dirección se reajusta en cada repetición a partir de las etiquetas permutadas

#### Scenario: Significancia sin tamaño de efecto
- **WHEN** se reporta un valor de significancia
- **THEN** se acompaña del tamaño de efecto y del intervalo de confianza
- **AND** ninguna conclusión se apoya solo en el valor de significancia

### Requirement: Decisión sobre el residuo por medición

El sistema SHALL construir una dirección con el conjunto emparejado y otra con el
residuo emparejado de forma más laxa, sobre **conjuntos disjuntos**, y SHALL comparar su
alineación contra dos referencias: direcciones aleatorias y dos mitades disjuntas del
conjunto emparejado.

Comparar direcciones que compartan ejemplos daría alineación alta por construcción y no
sería informativo.

#### Scenario: Conjuntos disjuntos
- **WHEN** se comparan las dos direcciones
- **THEN** ningún ejemplo aparece en ambas

#### Scenario: Referencias de escala
- **WHEN** se reporta la alineación
- **THEN** se reporta junto al valor esperado entre direcciones aleatorias y al obtenido
  entre dos mitades del conjunto emparejado
- **AND** la alineación no se interpreta sin esas dos referencias

#### Scenario: El residuo codifica lo mismo
- **WHEN** la alineación alcanza el umbral declarado respecto a la referencia superior
- **THEN** el sistema recomienda fusionar ambos conjuntos

#### Scenario: El residuo codifica algo distinto
- **WHEN** la alineación queda por debajo del umbral
- **THEN** el sistema mantiene el conjunto emparejado como conjunto de trabajo
- **AND** marca el residuo como conjunto de evaluación

### Requirement: Aislamiento del conjunto de transferencia

El sistema SHALL excluir las categorías declaradas como reservadas de toda decisión de
este change: construcción de la dirección, selección de capa y cualquier ajuste de
parámetros.

#### Scenario: Exclusión efectiva
- **WHEN** se construye cualquier dirección o se toma cualquier decisión de ajuste
- **THEN** ningún ejemplo de las categorías reservadas interviene

#### Scenario: Declaración en la salida
- **WHEN** se persiste el resultado
- **THEN** el artefacto enumera las categorías reservadas y confirma que no se usaron

### Requirement: Reproducibilidad

El sistema SHALL persistir la semilla, la configuración del contraste, la capa elegida,
las normas y factores de reescalado, y las versiones de las bibliotecas relevantes.

#### Scenario: Reejecución
- **WHEN** se repite con la misma semilla y configuración
- **THEN** las direcciones y métricas obtenidas son idénticas

#### Scenario: Auditoría posterior
- **WHEN** alguien abre el artefacto sin acceso a la sesión original
- **THEN** puede determinar qué datos, qué capa y qué construcción produjeron cada cifra
