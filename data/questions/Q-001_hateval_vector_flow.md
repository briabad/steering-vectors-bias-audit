---
id: Q-001
type: question
status: open
---

# Q-001 — ¿Puede una dirección latente detectar y moderar razonamientos sesgados sin tocar los pesos del modelo?

## Contexto
El proyecto está explorando si los modelos de lenguaje pueden detectar y controlar sesgos no solo a nivel de salida textual, sino también a nivel del flujo de representaciones internas. La revisión documental sugiere que las direcciones latentes, los vectores de control y la geometría del razonamiento son candidatos plausibles para distinguir textos con contenido de odio de textos neutrales.

## Pregunta
¿Es posible aprender una dirección latente discriminativa para discurso de odio en un modelo Qwen 7B-clase, y verificar que esa dirección captura diferencias reales en el flujo de representaciones, permitiendo detectar y mitigar razonamientos sesgados sin modificar los pesos del modelo?

## Por qué importa
La respuesta determina si la investigación puede avanzar desde un clasificador superficial hacia un mecanismo más interpretable y controlable. Si la dirección latente es estable y útil, entonces podría servir como base para:

- detectar contenido sesgado en tiempo de inferencia,
- explicar la causa del sesgo a partir del flujo interno,
- y aplicar una intervención mínima para evitar razonamientos de odio sin retraining.

## Enlaces
- motiva: [DEC-001, DEC-002, EXP-001]

> Nota: antes esta sección declaraba `motivado_por: [DEC-001]`, lo que creaba un
> ciclo con el `motivated_by: [Q-001]` de DEC-001. La dirección correcta es la de
> arriba: la pregunta es la raíz, y las decisiones se toman **para** responderla.
