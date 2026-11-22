"""Soporte para ejecutar localmente (sin Colab) el flujo del notebook del Grupo 1.

Que cambia respecto al notebook principal:
- `ensure_archive` descarga UNA vez el mismo `hls_burn_scars.tar.gz` (revision fijada del
  dataset en Hugging Face) y verifica su tamano y SHA-256 publicados.
- `subset_archive_local` es una copia linea a linea de `subset_archive` del modulo del
  notebook; la unica diferencia es la fuente de bytes (archivo local en vez de HTTP).
  Por eso la seleccion de escenas es la misma que obtiene el notebook en Colab.

Que NO cambia: lectura, recorte, normalizacion, modelo, cabeza, linea base y metricas se
importan sin cambios de `prithvi_taller.py`.
"""
from pathlib import Path
import re
import shutil
import tarfile

import requests

from prithvi_taller import SOURCES, LimitedReader, pair_tifs, sample_key, save_json, sha256

DATASET_REPO = "ibm-nasa-geospatial/hls_burn_scars"
DATASET_REVISION = "1864285e25010d346a842e4f068b1a1d4248ed6d"  # sha del repo (lastModified 2023-09-26)
ARCHIVE = "hls_burn_scars.tar.gz"
ARCHIVE_SIZE = 2645552531
ARCHIVE_SHA256 = "4e6f99a75cb2c500547b20662a15cbd531dc421376f815e91846ea542798e8e6"


def ensure_archive(cache):
    """Devuelve la ruta del tar local; lo descarga (reanudable) y verifica si hace falta."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / ARCHIVE
    marker = cache / (ARCHIVE + ".sha256_ok")
    if path.exists() and path.stat().st_size == ARCHIVE_SIZE and marker.exists():
        return path
    url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}/{ARCHIVE}"
    done = path.stat().st_size if path.exists() else 0
    if done < ARCHIVE_SIZE:
        headers = {"Range": f"bytes={done}-"} if done else {}
        with requests.get(url, stream=True, timeout=(30, 120), headers=headers) as r:
            r.raise_for_status()
            mode = "ab" if done and r.status_code == 206 else "wb"
            with open(path, mode) as f:
                for block in r.iter_content(4 * 1024 * 1024):
                    f.write(block)
    if path.stat().st_size != ARCHIVE_SIZE:
        raise RuntimeError("Tamano inesperado del tar descargado; no se continua.")
    if sha256(path) != ARCHIVE_SHA256:
        raise RuntimeError("El SHA-256 del tar no coincide con el publicado; no se continua.")
    marker.write_text(ARCHIVE_SHA256)
    return path


def subset_archive_local(archive_path, root, requested, combined=False, max_stream_mb=3000):
    """Copia de `subset_archive` (modulo del notebook) leyendo un archivo local."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    targets = {"train": requested[0], "test": requested[1]} if combined else requested
    kept = {s: {"image": set(), "label": set()} for s in targets}
    with open(archive_path, "rb") as raw:  # <- unica diferencia: archivo local, no requests
        counter = LimitedReader(raw, int(max_stream_mb * 1024**2))
        with tarfile.open(fileobj=counter, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.lower().endswith((".tif", ".tiff")):
                    continue
                low = member.name.lower()
                if combined:
                    if re.search(r"(^|[/_])(?:training|train)([/_]|$)", low):
                        split = "train"
                    elif re.search(r"(^|[/_])(?:validation|valid|val|test)([/_]|$)", low):
                        split = "test"
                    else:
                        continue
                else:
                    split = next(iter(targets))
                target = targets[split]
                if len(pair_tifs(root / split)) >= target:
                    continue
                kind = "label" if any(t in Path(low).name for t in ["mask", "label"]) else "image"
                key = sample_key(low)
                other = "image" if kind == "label" else "label"
                if key not in kept[split][other] and len(kept[split][kind]) >= 4 * target:
                    continue
                dest = root / split / Path(member.name).name
                dest.parent.mkdir(exist_ok=True)
                stream = tar.extractfile(member)
                with open(dest, "wb") as out:
                    shutil.copyfileobj(stream, out)
                kept[split][kind].add(key)
                if all(len(pair_tifs(root / s)) >= n for s, n in targets.items()):
                    break
        print("Bytes comprimidos leidos:", counter.total)
    result = {s: pair_tifs(root / s)[:n] for s, n in targets.items()}
    for s, n in targets.items():
        if len(result[s]) < n:
            raise RuntimeError(f"Solo se emparejaron {len(result[s])}/{n} en {s}.")
    return result, counter.total


def download_data_local(archive_path, root, ntrain, ntest, max_stream_mb=3000):
    """Equivalente local de `download_data('quemadas', ...)`: mismo manifiesto y revision.

    Si la carpeta ya contiene un manifiesto con la misma configuracion, se reutiliza
    (la seleccion depende solo del orden del tar y de n, asi que es determinista).
    """
    import json

    root = Path(root)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        if m.get("ntrain") == ntrain and m.get("ntest") == ntest:
            result = {"train": [], "test": []}
            for s in m["samples"]:
                result[s["split"]].append((Path(s["image"]), Path(s["label"])))
            return result, m.get("compressed_bytes_read")
    if root.exists() and any(root.rglob("*.tif")):
        raise RuntimeError(f"{root} tiene archivos de otra seleccion; use una carpeta vacia.")
    result, nbytes = subset_archive_local(
        archive_path, root, (ntrain, ntest), combined=True, max_stream_mb=max_stream_mb
    )
    save_json(root / "revision_datos.json", {"repo": DATASET_REPO, "revision": DATASET_REVISION})
    manifest = []
    for split, rows in result.items():
        for im, lab in rows:
            manifest.append({"split": split, "image": str(im), "label": str(lab),
                             "image_sha256": sha256(im), "label_sha256": sha256(lab)})
    save_json(manifest_path, {
        "task": "quemadas", "source": SOURCES["quemadas"], "ntrain": ntrain, "ntest": ntest,
        "compressed_bytes_read": nbytes,
        "note": "Subconjunto por orden del tar, no por desempeno; train/test segun proveedor.",
        "samples": manifest,
    })
    return result, nbytes
