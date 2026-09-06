# Contexto del proyecto

> Nivel 0 de contexto de dominio. Si necesitas el detalle técnico exhaustivo,
> ve a `docs/arquitectura_tecnica.md`, `docs/agente_realtime_pipecat_wip.md`
> y `docs/guia_integracion_cliente.md` — este README es solo la orientación
> mínima para entrar.

```

## Piezas clave

- **Pipecat**: framework de pipelines de audio/voz en tiempo real; gestiona
  transporte WebSocket (protocolo RTVI), VAD, y la integración con el STT.
- **Grafo LangGraph (Tier 1)**: procesa cada chunk de transcripción de forma
  secuencial y hace **merge incremental** del nuevo chunk sobre el estado ya
  extraído de la ficha. Serializado por cola+worker único (resuelve una race
  condition read-modify-write documentada en `docs/arquitectura_tecnica.md §7`).
- **Checkpointer PostgreSQL**: persiste el estado del grafo entre chunks de
  la misma conversación (necesario porque cada chunk depende del estado
  acumulado, no es una ejecución aislada).
- **Grafo de escalas (FEAT-001, v0)**: pipeline de escalas psicométricas
  (empoderamiento, malestares, RVD-BCN) portado de `spagia-ciam-transcripcion-async`,
  disparado por un endpoint manual de prueba — sin scheduler todavía. Usa un
  `thread_id`/checkpointer independiente del de Tier 1 y solo lee (nunca
  escribe) el checkpoint de Tier 1.

## Dónde profundizar

| Pregunta | Dónde |
|---|---|
| ¿Cómo está organizado el código? | `docs/architecture.md`, `docs/conventions.md` |
| ¿Cómo funciona el pipeline de audio y el grafo Tier 1 en detalle? | `docs/arquitectura_tecnica.md` |
| ¿Cómo se integra un cliente (protocolo WS/RTVI)? | `docs/guia_integracion_cliente.md` |
| ¿Qué bugs/limitaciones conocidas existen? | `docs/bug_native_pipecat_stt.md` |
| ¿Cómo se testea? | `docs/testing.md` |
| ¿Qué se está trabajando ahora? | `work/feature_list.json` |
| ¿Qué se sabe/decidió y por qué? | `data/index.md` |
| ¿Qué se decidió sobre el pipeline de escalas v0? | `openspec/changes/scales-rt-skeleton-v0/design.md`, `docs/specs/repo_spec.md §21` |

