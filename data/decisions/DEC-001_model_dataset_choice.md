---
id: DEC-001
type: decision
supersede: null
---

# DEC-001 — Elegimos Qwen 2.5 7B Instruct y Hateval como base del primer experimento

## Trigger
Necesitamos una línea experimental viable con un modelo razonador de tamaño moderado y un dataset con etiquetas claras para un primer estudio de latencia de sesgo. La hipótesis exige un modelo con capacidad de razonamiento y un corpus suficientemente maduro para validar separación latente.

## Alternativas consideradas
- Opción A — Qwen 2.5 7B Instruct + HatEval
- Opción B — Llama 3.2 3B Instruct + HatEval
- Opción C — Qwen 7B + POLAR
- Opción D — modelo muy grande y dataset más complejo

## Decisión
Elegimos Qwen/Qwen2.5-7B-Instruct como modelo base y Hateval como dataset principal para la primera batería experimental. La ejecución se realizará desde WSL y la experimentación se diseñará para operar sin cambiar pesos del modelo.

## Rationale
- Qwen 2.5 7B Instruct ofrece una buena relación entre capacidad de razonamiento y consumo de GPU, y encaja con la línea de trabajo de steering y flujo latente sin retraining.
- El tamaño 7B-class es viable para extracción de activaciones y pruebas de intervención en una máquina de trabajo razonable, mucho más que un modelo grande de 70B+.
- HatEval es un benchmark muy consolidado para discurso de odio; su etiqueta es más clara y menos multicausal que POLAR, lo que ayuda a validar la hipótesis de separación latente desde una base robusta.
- Toda la línea experimental se diseña primero para detectar si hay una firma latente real antes de escalar a un dataset más amplio o más sutil.

## Predicción
Si la hipótesis es correcta, deberíamos observar:

- una dirección latente discriminativa estable entre hate y non-hate,
- diferencias geométricas en velocidad/curvatura/continuidad del flujo,
- y una intervención por vector que cambie la salida de forma consistente sin tocar los pesos del modelo.

## Enlaces
- motivated_by: [Q-001]
- supersede: null
