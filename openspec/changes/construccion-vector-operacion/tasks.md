## 1. Construcción del contraste (sin GPU)

Sale del crudo ya medido. Aquí viven los errores que producirían un vector plausible pero
equivocado, así que todo se testea sin GPU ni red.

- [x] 1.1 Reutilizar la reconstrucción posicional de `scripts/05_recompute_gate.py` extrayéndola a `src/bbq_gate/`, con la verificación del ciclo de permutaciones que aborta si el patrón se rompe; verificar con un test que un ciclo alterado provoca el aborto (design.md D1)
- [x] 1.2 Derivar $P^+$ (rol `stereotyped` o `anti_stereotyped`) y $P^-$ (rol `unknown`) sobre ítems `ambig`; verificar contra las cifras conocidas: 1063 y 22910 en `chat`, 934 y 23047 en `plano`
- [x] 1.3 Implementar el emparejamiento 1:1 por `(categoría, plantilla)`; verificar que reproduce 752 pares en `chat` y 533 en `plano`, y que la distribución por categoría es **idéntica** en ambos lados (spec: "Contraste emparejado 1:1")
- [x] 1.4 Registrar el residuo por plantilla; verificar que suma 311 en `chat` y que las plantillas `Religion 21` y `Religion 24` aparecen con cero negativos disponibles
- [x] 1.5 Implementar la separación del conjunto de transferencia como **estructura de datos aparte** (design.md D6); verificar con un test que las funciones de ajuste no aceptan ítems de `Race_x_gender`, `Gender_identity` ni `Sexual_orientation`

## 2. Álgebra de direcciones (sin GPU)

- [x] 2.1 Implementar las tres construcciones de `SOA-004` sobre activaciones sintéticas; verificar con un test que la contrastiva usa diferencias **por pareja** y no diferencia de medias de conjuntos sueltos (spec: "Tres construcciones...")
- [x] 2.2 Implementar el reescalado por norma real de las activaciones para la variante PCA; verificar que las tres direcciones quedan en el mismo orden de magnitud y que el factor aplicado se registra
- [x] 2.3 Implementar proyección, normalización y coseno; verificar con vectores de prueba de resultado conocido (ortogonales, paralelos, opuestos)
- [x] 2.4 Implementar el test de permutación **reajustando la dirección dentro de cada permutación**; verificar con un test que permutar sin reajustar da un $p$ sistemáticamente menor — es el error que el spec prohíbe (spec: "Significancia por permutación con reajuste")
- [x] 2.5 Implementar $d$ de Cohen y bootstrap remuestreando **ítems**; verificar que duplicar artificialmente los datos no estrecha el intervalo
- [x] 2.6 Verificar cobertura de `src/bbq_gate/domain/` > 70 % (AGENTS.md §3.6) — medida: 92 %

## 3. Extracción de activaciones

- [x] 3.1 Declarar el `Protocol` del extractor en `domain/` (texto → activaciones por capa); verificar que `domain/` no importa `transformers` en ningún módulo
- [x] 3.2 Implementar el extractor con `transformers`, bf16, `torch.no_grad()`, capturando las 29 posiciones en el **último token**; verificar que `torch.cuda.is_available()` y que no hay offload a CPU
- [x] 3.3 Persistir en fp16 con modelo, revisión, dtype, política de posición y semilla; verificar que el fichero pesa ~310 MB y que los cinco campos están presentes (spec: "Extracción de activaciones sin modificar el modelo") — real: 419 MB para los 2016 ítems distintos que requirió el contraste efectivamente construido (más que la estimación de ~1500)
- [x] 3.4 Comprobar que los pesos no se han modificado ni se ha escrito checkpoint alguno

## 4. Selección de capa y evaluación de H1

- [x] 4.1 Partir el conjunto de ajuste en sub-ajuste / selección-de-capa / holdout, estratificado por categoría; verificar que las tres partes son disjuntas y que el holdout no se toca en los pasos 4.2-4.3
- [x] 4.2 Producir la curva de AUC por capa sobre el corte de selección, para las tres construcciones; verificar que se emiten las 29 capas, no solo la elegida (spec: "Selección de capa sin contaminar la evaluación")
- [x] 4.3 Congelar la capa elegida y registrarla **antes** de leer el holdout
- [x] 4.4 Evaluar en holdout la dirección junto a los baselines léxico y de capa 0; verificar que los tres números salen del mismo holdout (spec: "Evaluación relativa a baselines declarados")
- [ ] 4.5 Ejecutar permutación, $d$ de Cohen y bootstrap; reportar los tres juntos — **no ejecutado sobre el corpus real**: la tarea 4.6 disparó el aborto antes de llegar a este paso (ver 4.6). El código está implementado y testeado (2.4/2.5) sobre datos sintéticos, pero no hay cifras reales que reportar sin violar 4.6.
- [x] 4.6 Si algún baseline supera AUC 0.65, **parar y reportarlo como posible fuga** en lugar de continuar — **disparado en la ejecución real**: baseline léxico AUC = 0.9079 sobre el holdout real (`chat`, capa elegida 21). El pipeline abortó con `RuntimeError` en vez de continuar; ver informe del implementador para el diagnóstico.

## 5. Protocolo del residuo

- [ ] 5.1 Construir $v_A$ (752 pares) y $v_B$ (311 por categoría); verificar con un test que los conjuntos son **disjuntos** (design.md D3) — la lógica está implementada y testeada (`match_residue_by_category`, disjunción verificada con datos sintéticos y con la partición real 697/311 en el dry-run de construcción del contraste), pero $v_A$/$v_B$ nunca se construyeron sobre activaciones reales porque el pipeline abortó en la tarea 4.6 antes de llegar a `residue_protocol()`.
- [ ] 5.2 Calcular el suelo: coseno entre pares de direcciones aleatorias en la dimensión del modelo; verificar que la media es ≈ 0 y la desviación ≈ $1/\sqrt{d}$ — implementado y testeado (`random_direction_cosine_floor`), no ejecutado sobre el vector real por el mismo motivo.
- [ ] 5.3 Calcular el techo: coseno entre direcciones de dos mitades disjuntas de los 752, promediado sobre varias particiones — implementado y testeado (`split_half_cosine_ceiling`), no ejecutado sobre el vector real.
- [ ] 5.4 Reportar $\cos(v_A, v_B)$ **junto a** ambas anclas, y aplicar el umbral del 80 % del techo; verificar que la recomendación (fusionar / mantener separados) queda escrita explícitamente — no alcanzado.

## 6. Salida y cierre

- [ ] 6.1 Escribir `data/experiments/EXP-002_bbq_stereotype_direction/output/operation_vector.json` con direcciones, capa elegida, curva por capa, métricas, anclas del residuo y la lista de categorías reservadas confirmando que no se usaron (spec: "Aislamiento del conjunto de transferencia", "Reproducibilidad") — **no escrito**: el script rehúsa escribir el artefacto canónico cuando aborta por fuga (misma lección de la sesión anterior: nunca escribir un veredicto contaminado en la ruta canónica).
- [ ] 6.2 Ejecutar el pipeline completo con el intérprete de DEC-002 y reportar el tiempo real — ejecutado dos veces de extremo a extremo (extracción real de 2016 ítems en 58.4 s ambas veces, reproducible), pero **no "completo"**: se detiene en 4.6 antes del cierre. Tiempo real hasta el aborto: bajo 2 minutos.
- [ ] 6.3 Contrastar contra H1 de `EXP-002/hypothesis.md` y **reportar el veredicto sin interpretarlo como éxito o fracaso del change**: un AUC bajo es un resultado válido — no hay veredicto de H1 que reportar: el propio C2 (baseline vs dirección) es el primer criterio que falla, antes de C1/C3/C4.
- [ ] 6.4 Añadir la nota de limitación sobre selección hacia plantillas con abstenciones disponibles, y sobre posible contaminación de BBQ — no aplica todavía: no se generó el artefacto donde iría esa nota. La limitación real y más grave encontrada (framing léxico del `question_index` en vez de solo vocabulario) se reporta en el informe del implementador, no en un artefacto.
- [x] 6.5 `pytest` en verde, `python tests/validate_graph.py --strict` exit 0, `openspec validate construccion-vector-operacion --strict` exit 0 — **ejecutados, con la salida leída**
- [x] 6.6 Rellenar `trace.implements` de `FEAT-012` y `FEAT-013`; **no** marcar features como `done`, **no** escribir `results.md`, **no** tocar `data/index.md`
