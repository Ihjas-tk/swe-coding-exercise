"""Pre-download the local models so the first ingest does not stall on a silent download.

Everything lands in the Hugging Face hub cache, the directory `huggingface_hub` itself
resolves: `$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`. That is
where sentence-transformers and Docling look at run time, so a prefetched model is never
downloaded twice.

What is fetched:
- the embedding model (`AE_EMBED_MODEL`, default granite-embedding-english-r2, ~290 MB) and
  the bge-small fallback (~130 MB); safetensors weights only when the repo has them (the
  duplicate `pytorch_model.bin` and ONNX exports are skipped);
- Docling's layout model (heron, ~165 MB) and TableFormer weights (~340 MB).

Why not `docling.utils.model_downloader.download_models()`: in docling 2.132 it writes into
a `local_dir` under `~/.cache/docling/models`, which the default pipeline (no
`artifacts_path`, as in `ae/extract/docling_backend.py`) never reads; the pipeline calls
`download_hf_model(..., local_dir=None)`, i.e. the hub cache. We call the same functions
the pipeline calls, so the files are exactly the ones it will look for.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from ae import config  # loads .env first, so HF_HOME / HF_HUB_CACHE / HF_ENDPOINT set there apply

FALLBACK_EMBED = "BAAI/bge-small-en-v1.5"
# Docling repos used by the standard PDF pipeline with our options (layout + TableFormer;
# picture classification, code/formula enrichment and picture description are off).
DOCLING_REPOS = ("docling-project/docling-layout-heron", "docling-project/docling-models")
# Weight formats sentence-transformers never loads on CPU with safetensors present.
_SKIP = ["*.onnx", "onnx/*", "openvino/*", "*.h5", "*.msgpack", "*.ot", "flax_model*", "tf_model*", "rust_model*"]


def hub_cache() -> Path:
    """Return the Hugging Face hub cache directory (honours HF_HUB_CACHE and HF_HOME)."""
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def repo_dir(repo_id: str) -> Path:
    """Cache folder of one model repo (`models--org--name`)."""
    return hub_cache() / ("models--" + repo_id.replace("/", "--"))


def is_cached(repo_id: str) -> bool:
    """Return True when at least one snapshot of the repo has files in the cache."""
    snaps = repo_dir(repo_id) / "snapshots"
    return snaps.is_dir() and any(any(s.iterdir()) for s in snaps.iterdir() if s.is_dir())


def _size(path: Path) -> int:
    # snapshot entries are symlinks into blobs/; stat() follows them
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _fmt(n: int) -> str:
    return f"{n / 2**20:,.0f} MB"


def _say(msg: str) -> None:
    print(msg, flush=True)


def download_embedding(repo_id: str) -> Path:
    """Fetch a sentence-transformers model into the hub cache; returns the snapshot path."""
    from huggingface_hub import HfApi, snapshot_download

    ignore = list(_SKIP)
    try:
        files = HfApi().list_repo_files(repo_id)
        if any(f.endswith(".safetensors") for f in files):
            ignore += ["*.bin", "*.pt"]
    except Exception:  # offline or blocked: snapshot_download falls back to the cache or raises
        pass
    return Path(snapshot_download(repo_id, ignore_patterns=ignore))


def download_docling() -> list[Path]:
    """Fetch the Docling layout and TableFormer models exactly as the pipeline would."""
    try:
        from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions
        from docling.models.stages.table_structure.table_structure_model import TableStructureModel
        from docling.models.utils.hf_model_download import download_hf_model
    except ImportError:  # docling moved these; fall back to the repos it used in 2.132
        from huggingface_hub import snapshot_download

        return [Path(snapshot_download(r)) for r in DOCLING_REPOS]
    spec = LayoutObjectDetectionOptions().model_spec
    progress = sys.stderr.isatty()
    return [
        download_hf_model(repo_id=spec.repo_id, revision=spec.revision, progress=progress),
        TableStructureModel.download_models(progress=progress),
    ]


def prefetch(embed_model: str | None = None, fallback: bool = True, docling: bool = True) -> list[str]:
    """Download the embedding model (+ bge-small fallback) and the Docling models; return local paths.

    Prints one line per model with its size and location. Every model is attempted; if any
    failed, raises RuntimeError at the end listing them (the others are still cached).
    """
    embed_model = embed_model or config.EMBED_MODEL
    model = embed_model
    jobs: list[tuple[str, bool, Callable[[], list[Path]]]] = [
        (model, is_cached(model), lambda: [download_embedding(model)])
    ]
    if fallback and model != FALLBACK_EMBED:
        jobs.append((FALLBACK_EMBED, is_cached(FALLBACK_EMBED), lambda: [download_embedding(FALLBACK_EMBED)]))
    if docling:
        jobs.append(("docling layout + TableFormer", all(is_cached(r) for r in DOCLING_REPOS), download_docling))

    _say(f"Hugging Face cache: {hub_cache()}")
    paths: list[str] = []
    failed: list[str] = []
    for label, cached, fn in jobs:
        _say(f"  - {label}{' (cached, checking for updates)' if cached else ' (downloading)'} ...")
        try:
            got = fn()
        except Exception as e:  # network/HF errors vary by version; report and continue
            failed.append(label)
            _say(f"  ! {label}: failed: {type(e).__name__}: {e}")
            continue
        for p in got:
            paths.append(str(p))
            _say(f"  ✓ {_fmt(_size(p)):>8}  {p}")
    if failed:
        raise RuntimeError(
            "model download failed for: "
            + ", ".join(failed)
            + " (check network access to huggingface.co, or set HF_ENDPOINT to a mirror)"
        )
    total = sum(_size(Path(p)) for p in paths)
    _say(f"  {len(paths)} model folder(s), {_fmt(total)} on disk")
    return paths


if __name__ == "__main__":
    try:
        prefetch()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
