"""Pre-download the local models into models/<name> for offline bundling.

Fetches the local embedding model (voyage-4-nano), the local reranker model
(mxbai-rerank-large-v2), and the STT Whisper CT2 model (large-v3-turbo). Run inside
the build venv BEFORE PyInstaller (the CI freeze build does this) so the frozen bundle
ships the models and never hits the network at runtime:

    python fetch_local_model.py

companion-server.spec bundles the resulting models/ directory as PyInstaller datas;
companion_server.py's _resolve_local_model_ref() / _resolve_local_rerank_ref() /
_resolve_local_stt_ref() then load them offline. NOTE: the reranker model is ~3GB and
the Whisper CT2 model is ~1.6GB (quality-first choices), so the local-edition bundle
grows accordingly.
"""

import os
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_IDS = [
    os.getenv("LOCAL_EMBEDDING_MODEL", "voyageai/voyage-4-nano").strip() or "voyageai/voyage-4-nano",
    os.getenv("LOCAL_RERANK_MODEL", "mixedbread-ai/mxbai-rerank-large-v2").strip() or "mixedbread-ai/mxbai-rerank-large-v2",
    # STT: the CTranslate2 Whisper model behind the faster-whisper alias large-v3-turbo.
    os.getenv("STT_HF_REPO", "mobiuslabsgmbh/faster-whisper-large-v3-turbo").strip() or "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
]


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - build-time helper
        print(f"[fetch-model] huggingface_hub is missing: {exc}", file=sys.stderr)
        print("[fetch-model] install requirements-local-embed.txt first.", file=sys.stderr)
        return 1

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for model_id in MODEL_IDS:
        out_dir = MODELS_DIR / Path(model_id).name
        print(f"[fetch-model] downloading {model_id} -> {out_dir}")
        # Full repo snapshot: weights + config + any trust_remote_code modeling files,
        # all required to load the model from a local path with no network.
        snapshot_download(repo_id=model_id, local_dir=str(out_dir))
    print("[fetch-model] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
