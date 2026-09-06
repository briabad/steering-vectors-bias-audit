#!/usr/bin/env python3
"""Sanidad del grafo de conocimiento y de la capa de trabajo.

Comprueba las invariantes que `METHODOLOGY.md §2 Principio B` declara pero que
hasta ahora se sostenían solo por disciplina humana:

1. Todo artefacto en disco aparece como fila en `data/index.md` (no hay huérfanos).
2. Toda fila de `data/index.md` apunta a un artefacto que existe.
3. Todo ID referenciado desde cualquier artefacto existe.
4. `work/feature_list.json` es JSON válido y cumple el schema de `work/README.md`.
5. Todo ID citado en `trace.motivated_by` de una feature existe.

Salida: exit 0 si está sano, 1 si hay errores. `--strict` convierte además los
avisos (nodos aislados, secciones de enlaces ausentes) en errores.

Uso:  python tests/validate_graph.py [--strict]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
INDEX = DATA / "index.md"
FEATURES = REPO / "work" / "feature_list.json"

ID_RE = re.compile(r"\b(?:Q|EXP|DEC|SOA|TOT)-\d{3}\b")
FEAT_RE = re.compile(r"\bFEAT-\d{3}\b")
MD_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

# tipo -> (carpeta, ¿el artefacto es un directorio con manifest.md?)
KINDS: Dict[str, Tuple[str, bool]] = {
    "Q": ("questions", False),
    "EXP": ("experiments", True),
    "DEC": ("decisions", False),
    "SOA": ("references", False),
}

VALID_STATUS = {"pending", "in_progress", "blocked", "deferred", "done", "dropped"}


def discover_artifacts() -> Dict[str, Path]:
    """Localiza los artefactos que existen realmente en disco, por ID."""
    found: Dict[str, Path] = {}
    for prefix, (folder, is_dir) in KINDS.items():
        base = DATA / folder
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir()):
            if entry.name.startswith((".", "_")):
                continue
            if is_dir and entry.is_dir():
                found[entry.name.split("_")[0]] = entry / "manifest.md"
            elif not is_dir and entry.is_file() and entry.suffix == ".md":
                found[entry.name.split("_")[0]] = entry
    return found


def parse_index() -> Dict[str, List[str]]:
    """Extrae de index.md los IDs listados y los destinos de sus enlaces."""
    rows: Dict[str, List[str]] = {}
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        ids = ID_RE.findall(line)
        if not ids:
            continue
        rows[ids[0]] = MD_LINK_RE.findall(line)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strict", action="store_true", help="Trata los avisos como errores")
    args = parser.parse_args()

    errors: List[str] = []
    warnings: List[str] = []

    if not INDEX.is_file():
        print(f"ERROR: no existe {INDEX.relative_to(REPO)}", file=sys.stderr)
        return 1

    on_disk = discover_artifacts()
    in_index = parse_index()

    # 1. huérfanos: en disco pero no en el índice
    for art_id in sorted(set(on_disk) - set(in_index)):
        errors.append(f"[huerfano] {art_id} existe en disco ({on_disk[art_id].relative_to(REPO)}) pero no en index.md")

    # 2. filas fantasma: en el índice pero no en disco
    for art_id in sorted(set(in_index) - set(on_disk)):
        errors.append(f"[fantasma] index.md lista {art_id} pero no hay artefacto en disco")

    # 2b. los enlaces markdown del índice resuelven
    for art_id, targets in sorted(in_index.items()):
        for target in targets:
            if not (INDEX.parent / target).exists():
                errors.append(f"[enlace-roto] fila {art_id} apunta a '{target}', que no existe")

    # 3. referencias cruzadas dentro de cada artefacto
    known: Set[str] = set(on_disk)
    for art_id, path in sorted(on_disk.items()):
        if not path.is_file():
            errors.append(f"[incompleto] {art_id} no tiene {path.name} (esperado en {path.parent.relative_to(REPO)})")
            continue
        body = path.read_text(encoding="utf-8")
        refs = {r for r in ID_RE.findall(body) if r != art_id}
        for ref in sorted(refs - known):
            errors.append(f"[ref-rota] {art_id} referencia {ref}, que no existe")
        if not refs:
            warnings.append(f"[aislado] {art_id} no referencia ningun otro artefacto")

    # 4-5. capa de trabajo
    if not FEATURES.is_file():
        errors.append(f"[work] falta {FEATURES.relative_to(REPO)}")
    else:
        raw = FEATURES.read_text(encoding="utf-8").strip()
        if not raw:
            errors.append("[work] feature_list.json esta vacio (JSON invalido)")
        else:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"[work] feature_list.json no es JSON valido: {exc}")
            else:
                feats = payload.get("features", [])
                if not isinstance(feats, list):
                    errors.append("[work] 'features' debe ser una lista")
                    feats = []
                seen: Set[str] = set()
                for i, feat in enumerate(feats):
                    fid = feat.get("id", f"<sin id en posicion {i}>")
                    if not FEAT_RE.fullmatch(str(fid)):
                        errors.append(f"[work] id de feature invalido: {fid!r}")
                    if fid in seen:
                        errors.append(f"[work] id de feature duplicado: {fid}")
                    seen.add(fid)
                    for field in ("title", "status", "acceptance", "trace"):
                        if field not in feat:
                            errors.append(f"[work] {fid} no tiene '{field}'")
                    status = feat.get("status")
                    if status is not None and status not in VALID_STATUS:
                        errors.append(f"[work] {fid} tiene status {status!r}; validos: {sorted(VALID_STATUS)}")
                    if not feat.get("acceptance"):
                        errors.append(f"[work] {fid} no tiene acceptance criteria")
                    for ref in feat.get("trace", {}).get("motivated_by", []):
                        if ref not in known:
                            errors.append(f"[work] {fid} esta motivada por {ref}, que no existe en data/")
                next_id = payload.get("next_id")
                if next_id in seen:
                    errors.append(f"[work] next_id {next_id} ya esta en uso")

    print(f"artefactos en disco : {len(on_disk)}  ({', '.join(sorted(on_disk)) or 'ninguno'})")
    print(f"filas en index.md   : {len(in_index)}")
    for w in warnings:
        print(f"AVISO {w}")
    for e in errors:
        print(f"ERROR {e}", file=sys.stderr)

    if errors or (args.strict and warnings):
        print(f"\nFALLO: {len(errors)} error(es), {len(warnings)} aviso(s)", file=sys.stderr)
        return 1
    print(f"\nOK: grafo sano ({len(warnings)} aviso(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
