"""BGH Red Thread AI companion server.

Sanitized Flask server for local RAG, NPC memory, GraphRAG, optional STT,
and optional Vertex AI OpenAI-compatible proxying.

This file intentionally contains no campaign data, API keys, ChromaDB files,
or user-specific paths. Configure it with environment variables or a local
.env file copied from .env.example.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import urllib.error
import urllib.request
import uuid
from functools import wraps
from pathlib import Path
from typing import Any, Callable

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional convenience dependency
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

from flask import Flask, Response, jsonify, request
from flask_cors import CORS


APP_NAME = "BGH Red Thread AI Companion"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000

# Runtime locations.
# BASE_DIR is where the code/exe lives. When frozen with PyInstaller, sys.executable
# points at the real exe (not the _internal bundle), so anything beside it survives an
# in-place update. But the exe should be disposable: a user may move it, re-download it,
# or drop it in a new folder. So persistent runtime state (the Chroma index, world graph,
# and hub-set keys) lives in a fixed per-user HOME_DIR independent of the exe's location,
# and a one-time migration relocates any legacy beside-exe state into it. In a dev
# checkout HOME_DIR == BASE_DIR, so behavior there is unchanged.
def _default_companion_home() -> Path:
    """Fixed per-user home for the frozen build's persistent state. COMPANION_HOME wins."""
    override = os.getenv("COMPANION_HOME")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        win_base = os.getenv("APPDATA") or os.getenv("LOCALAPPDATA")
        if win_base:
            return Path(win_base) / "BGHRedThreadCompanion"
    xdg = os.getenv("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "BGHRedThreadCompanion"
    return Path.home() / ".bgh-redthread-companion"


if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
    HOME_DIR = _default_companion_home().resolve()
else:
    BASE_DIR = Path(__file__).resolve().parent
    HOME_DIR = BASE_DIR

load_dotenv(BASE_DIR / ".env")
if HOME_DIR != BASE_DIR:
    load_dotenv(HOME_DIR / ".env")  # also honor a .env kept in the stable home


def _stable_state_path(name: str) -> Path:
    """Resolve a persistent state item (file or dir) to the stable HOME_DIR, migrating a
    legacy beside-exe copy into HOME_DIR once. Falls back to the legacy path if migration
    fails, so an existing install never loses access to its data or keys."""
    home_path = HOME_DIR / name
    legacy_path = BASE_DIR / name
    if HOME_DIR == BASE_DIR or home_path.exists() or not legacy_path.exists():
        return home_path
    try:
        HOME_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_path), str(home_path))
        print(f"[Companion] Migrated runtime state '{name}' -> {home_path}")
        return home_path
    except Exception as exc:  # pragma: no cover - environment specific
        print(f"[Companion] Migration of '{name}' failed ({exc}); using legacy {legacy_path}")
        return legacy_path


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


HOST = os.getenv("COMPANION_HOST", DEFAULT_HOST)
PORT = int(os.getenv("COMPANION_PORT", str(DEFAULT_PORT)))
API_KEY = os.getenv("COMPANION_API_KEY", "").strip()
CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]

# The companion auth key (== COMPANION_API_KEY) can also be set from the Foundry Settings
# Hub, so a prebuilt-bundle user never edits .env. The hub value is persisted next to the
# executable; the COMPANION_API_KEY env var, when set, always wins and cannot be changed
# from the hub. require_auth() resolves the effective key dynamically so a hub-set key
# takes effect without a server restart.
COMPANION_AUTH_KEYSTORE_PATH = _stable_state_path("companion_auth.json")


def load_persisted_companion_auth_key() -> str:
    try:
        with open(COMPANION_AUTH_KEYSTORE_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return ""
    key = data.get("companion_api_key") if isinstance(data, dict) else ""
    return str(key or "").strip()


PERSISTED_COMPANION_AUTH_KEY = load_persisted_companion_auth_key()


def companion_auth_env_key() -> str:
    return (os.getenv("COMPANION_API_KEY", "") or "").strip()


def resolve_companion_auth_key() -> str:
    """Effective companion auth key: the env var wins, else the hub-persisted value."""
    return companion_auth_env_key() or PERSISTED_COMPANION_AUTH_KEY


def set_persisted_companion_auth_key(api_key: str) -> bool:
    """Persist the hub-set companion auth key (empty clears it). Returns True when set."""
    global PERSISTED_COMPANION_AUTH_KEY
    PERSISTED_COMPANION_AUTH_KEY = (api_key or "").strip()
    try:
        with open(COMPANION_AUTH_KEYSTORE_PATH, "w", encoding="utf-8") as handle:
            json.dump({"companion_api_key": PERSISTED_COMPANION_AUTH_KEY}, handle)
    except OSError as exc:
        print(f"[Companion auth] Failed to persist key: {exc}")
    return bool(PERSISTED_COMPANION_AUTH_KEY)

COMPANION_LEGACY_RAG_ENABLED = env_flag("COMPANION_LEGACY_RAG_ENABLED")
COMPANION_LEGACY_MEMORY_ENABLED = env_flag("COMPANION_LEGACY_MEMORY_ENABLED")
COMPANION_LEGACY_GRAPH_ENABLED = env_flag("COMPANION_LEGACY_GRAPH_ENABLED")

_data_dir_env = os.getenv("COMPANION_DATA_DIR")
DATA_DIR = Path(_data_dir_env).resolve() if _data_dir_env else _stable_state_path("data").resolve()
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", DATA_DIR / "chroma_db")).resolve()
GRAPH_FILE = Path(os.getenv("GRAPH_FILE", DATA_DIR / "world_graph.json")).resolve()

RAG_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "bgh_world_rag")
MEMORY_COLLECTION_NAME = os.getenv("MEMORY_COLLECTION_NAME", "bgh_npc_memories")
LEGACY_RAG_COLLECTION_NAME = os.getenv("LEGACY_RAG_COLLECTION_NAME", "legacy_rag")
LEGACY_MEMORY_COLLECTION_NAME = os.getenv("LEGACY_MEMORY_COLLECTION_NAME", "legacy_npc_memories")

APPROVED_RETRIEVAL_COLLECTION_NAME = os.getenv(
    "APPROVED_RETRIEVAL_COLLECTION_NAME",
    "bgh_approved_memory_v1",
)
RED_WORLD_KNOWLEDGE_COLLECTION_NAME = os.getenv(
    "RED_WORLD_KNOWLEDGE_COLLECTION_NAME",
    "bgh_RED_world_knowledge",
)
GM_BOT_KNOWLEDGE_COLLECTION_NAME = os.getenv(
    "GM_BOT_KNOWLEDGE_COLLECTION_NAME",
    "bgh_RED_gm_bot_knowledge",
)
APPROVED_RETRIEVAL_DUMMY_MARKER = "__025E4_DUMMY_APPROVED_RETRIEVAL_RECORD__"
APPROVED_RETRIEVAL_PURGE_CONFIRM_TEXT = "검색 인덱스 삭제"
APPROVED_RETRIEVAL_LEGACY_PURGE_CONFIRM_TEXT = "LEGACY 동기화 데이터 삭제"
WORLD_ISOLATION_MIGRATION_CONFIRM_TEXT = "월드 격리 이관"
APPROVED_RETRIEVAL_AUDIT_PREVIEW_MAX_CHARS = 3200
APPROVED_RETRIEVAL_AUDIT_PREVIEW_SOURCE_INDEX_RECORD = "검색 인덱스 본문"
APPROVED_RETRIEVAL_AUDIT_PREVIEW_SOURCE_METADATA = "metadata preview"
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "").strip()
VOYAGE_EMBEDDING_MODEL = os.getenv("VOYAGE_EMBEDDING_MODEL", "voyage-4-large").strip() or "voyage-4-large"
VOYAGE_EMBEDDING_OUTPUT_DIMENSION = int(os.getenv("VOYAGE_EMBEDDING_OUTPUT_DIMENSION", "1024"))
VOYAGE_EMBEDDING_URL = os.getenv("VOYAGE_EMBEDDING_URL", "https://api.voyageai.com/v1/embeddings").strip()

# Local embedding fallback (open-weight voyage-4-nano). Lets a buyer run the
# companion server WITHOUT a Voyage API key. Voyage's "shared embedding space"
# makes voyage-4-nano vectors interchangeable with the voyage-4-large index at
# the same dimension, so enabling this needs no re-indexing. Off by default; the
# bundled local edition turns it on via .env. Requires sentence-transformers +
# torch (requirements-local-embed.txt), which the lean Voyage-BYOK build omits.
# Default ON in a frozen build (the bundle ships the model + torch); OFF in a plain
# dev checkout where the heavy deps are not installed.
LOCAL_EMBEDDING_ENABLED = os.getenv(
    "LOCAL_EMBEDDING_ENABLED",
    "true" if getattr(sys, "frozen", False) else "false",
).lower() in {"1", "true", "yes", "on"}
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "voyageai/voyage-4-nano").strip() or "voyageai/voyage-4-nano"
LOCAL_EMBEDDING_DIMENSION = int(os.getenv("LOCAL_EMBEDDING_DIMENSION", str(VOYAGE_EMBEDDING_OUTPUT_DIMENSION)))
LOCAL_EMBEDDING_DEVICE = os.getenv("LOCAL_EMBEDDING_DEVICE", "cpu").strip() or "cpu"

# Reranker (second-stage precision). Cloud = Voyage rerank (BYOK, shares the
# Voyage key). Local fallback = an open-weight MULTILINGUAL cross-encoder. The
# multilingual constraint is a do-no-harm requirement: an English-only reranker
# over-weights English token overlap and would demote the language-supporting
# (e.g. Korean) chunk. Default ON in a frozen build, OFF in a dev checkout.
VOYAGE_RERANK_MODEL = os.getenv("VOYAGE_RERANK_MODEL", "rerank-2.5").strip() or "rerank-2.5"
VOYAGE_RERANK_URL = os.getenv("VOYAGE_RERANK_URL", "https://api.voyageai.com/v1/rerank").strip()
LOCAL_RERANK_ENABLED = os.getenv(
    "LOCAL_RERANK_ENABLED",
    "true" if getattr(sys, "frozen", False) else "false",
).lower() in {"1", "true", "yes", "on"}
LOCAL_RERANK_MODEL = os.getenv("LOCAL_RERANK_MODEL", "mixedbread-ai/mxbai-rerank-large-v2").strip() or "mixedbread-ai/mxbai-rerank-large-v2"
LOCAL_RERANK_DEVICE = os.getenv("LOCAL_RERANK_DEVICE", "cpu").strip() or "cpu"

# Default ON in a frozen build (the bundle ships the Whisper CT2 model under models/);
# OFF in a plain dev checkout where faster-whisper + the model are not installed.
STT_ENABLED = os.getenv(
    "STT_ENABLED",
    "true" if getattr(sys, "frozen", False) else "false",
).lower() in {"1", "true", "yes", "on"}
STT_MODEL = os.getenv("STT_MODEL", "small")
# Hugging Face repo of the CTranslate2 Whisper model pre-downloaded for offline
# bundling (fetch_local_model.py) and resolved at runtime in the frozen build. This
# is the concrete repo behind the faster-whisper alias STT_MODEL=large-v3-turbo.
STT_HF_REPO = os.getenv("STT_HF_REPO", "mobiuslabsgmbh/faster-whisper-large-v3-turbo").strip() or "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
STT_DEVICE = os.getenv("STT_DEVICE", "cpu")
STT_COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "int8")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "ko")
try:
    # Decoder beam width. Higher = slightly better accuracy, a bit slower. 8 is a
    # good quality/speed point on GPU (was a hardcoded 5). Tunable via .env.
    STT_BEAM_SIZE = int(os.getenv("STT_BEAM_SIZE", "8"))
except (TypeError, ValueError):
    STT_BEAM_SIZE = 8

VERTEX_PROJECT_ID = (
    os.getenv("VERTEX_PROJECT_ID")
    or os.getenv("GOOGLE_CLOUD_PROJECT")
    or os.getenv("GCLOUD_PROJECT")
    or os.getenv("PROJECT_ID")
)
VERTEX_REGION = os.getenv("VERTEX_REGION") or os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("REGION", "global")
VERTEX_QUOTA_PROJECT_ID = os.getenv("VERTEX_QUOTA_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_QUOTA_PROJECT")


def resolve_vertex_project_id(default_project: str | None = None) -> str | None:
    """Return the Vertex project ID from env aliases or Google ADC metadata."""

    configured = (
        os.getenv("VERTEX_PROJECT_ID")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
        or os.getenv("PROJECT_ID")
        or default_project
    )
    return configured.strip() if isinstance(configured, str) and configured.strip() else None


def resolve_vertex_quota_project_id(default_project: str | None = None) -> str | None:
    """Return a quota project for end-user ADC credentials.

    Google Cloud SDK user credentials can warn when no quota project is attached.
    For local use, the Vertex project is usually also the quota project.
    """

    configured = VERTEX_QUOTA_PROJECT_ID or default_project or resolve_vertex_project_id()
    return configured.strip() if isinstance(configured, str) and configured.strip() else None


VERTEX_PROXY_RESPONSE_HEADERS = {
    "retry-after",
    "x-request-id",
    "x-goog-request-id",
    "x-cloud-trace-context",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
}
VERTEX_PROXY_SAFE_ERROR_KEYS = {
    "@type",
    "code",
    "details",
    "domain",
    "location",
    "message",
    "metadata",
    "method",
    "model",
    "quotaId",
    "quotaLimit",
    "quotaLimitValue",
    "quotaLocation",
    "quotaMetric",
    "quota_id",
    "quota_limit",
    "quota_limit_value",
    "quota_location",
    "quota_metric",
    "reason",
    "requestId",
    "request_id",
    "retryAfter",
    "retry_after",
    "service",
    "status",
    "type",
    "violations",
}


def clamp_vertex_proxy_string(value: Any, limit: int = 500) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def is_vertex_proxy_safe_error_key(key: str) -> bool:
    if key in VERTEX_PROXY_SAFE_ERROR_KEYS:
        return True
    normalized = key.replace("-", "").replace("_", "").lower()
    return "quota" in normalized or normalized in {"requestid", "retryafter"}


def sanitize_vertex_proxy_error_metadata(value: Any, depth: int = 0) -> Any:
    if value is None:
        return None
    if depth > 5:
        return "[truncated]"
    if isinstance(value, (str, int, float, bool)):
        return clamp_vertex_proxy_string(value) if isinstance(value, str) else value
    if isinstance(value, list):
        sanitized_list = [
            sanitize_vertex_proxy_error_metadata(entry, depth + 1)
            for entry in value[:6]
        ]
        return [entry for entry in sanitized_list if entry not in (None, "", [], {})]
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, entry in value.items():
            key_text = str(key)
            if not is_vertex_proxy_safe_error_key(key_text):
                continue
            safe_value = sanitize_vertex_proxy_error_metadata(entry, depth + 1)
            if safe_value not in (None, "", [], {}):
                sanitized[key_text] = safe_value
        return sanitized or None
    return clamp_vertex_proxy_string(value)


def vertex_proxy_response_headers(headers: Any) -> dict[str, str]:
    output: dict[str, str] = {}
    for key in VERTEX_PROXY_RESPONSE_HEADERS:
        value = headers.get(key) if hasattr(headers, "get") else None
        if value:
            output[key] = clamp_vertex_proxy_string(value, 240)
    return output


def normalize_vertex_proxy_error_response(response: Any) -> dict[str, Any]:
    body_status = "empty"
    body_json: Any = None
    if response.content:
        try:
            body_json = response.json()
            body_status = "metadata_present"
        except ValueError:
            body_status = "unreadable"

    source_error = body_json.get("error") if isinstance(body_json, dict) else None
    if not isinstance(source_error, dict):
        source_error = body_json if isinstance(body_json, dict) else {}
    metadata = sanitize_vertex_proxy_error_metadata(source_error)
    if not metadata and body_status == "metadata_present":
        body_status = "metadata_not_allowed"

    message = ""
    if isinstance(metadata, dict):
        message = clamp_vertex_proxy_string(metadata.get("message", ""))
    if not message:
        message = f"Vertex upstream returned HTTP {response.status_code} with {body_status} body."

    error_payload: dict[str, Any] = {
        "message": message,
        "type": clamp_vertex_proxy_string(metadata.get("type", "")) if isinstance(metadata, dict) else "",
        "code": metadata.get("code", response.status_code) if isinstance(metadata, dict) else response.status_code,
        "status": metadata.get("status", response.status_code) if isinstance(metadata, dict) else response.status_code,
        "body_status": body_status,
        "upstream": "vertex_openai_proxy",
    }
    if metadata:
        error_payload["metadata"] = metadata
    return {"error": {key: value for key, value in error_payload.items() if value not in (None, "", [], {})}}


def count_vertex_text_chars(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(count_vertex_text_chars(entry) for entry in value)
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return len(value["text"])
        return sum(count_vertex_text_chars(entry) for entry in value.values())
    return 0


def sanitize_vertex_safety_rating(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    rating = {
        "category": clamp_vertex_proxy_string(value.get("category", ""), 120),
        "blocked": bool(value.get("blocked", False)),
        "severity": clamp_vertex_proxy_string(value.get("severity") or value.get("severityLevel", ""), 80),
        "probability": clamp_vertex_proxy_string(value.get("probability") or value.get("probabilityScore", ""), 80),
    }
    return {key: entry for key, entry in rating.items() if entry not in (None, "", [], {})}


def infer_vertex_empty_reason(diagnostic: dict[str, Any]) -> str:
    if diagnostic.get("output_text_chars", 0) > 0:
        return ""
    if diagnostic.get("prompt_feedback_block_reason"):
        return "safety_blocked"
    if diagnostic.get("candidates_count", 0) <= 0:
        return "no_candidates"
    if diagnostic.get("content_parts_count", 0) <= 0:
        return "no_content_parts"
    finish_reason = str(diagnostic.get("candidate_finish_reason") or "").lower()
    if "safety" in finish_reason or "blocked" in finish_reason or "prohibited" in finish_reason:
        return "safety_blocked"
    if "max_tokens" in finish_reason or "max_token" in finish_reason or finish_reason == "length":
        return "max_tokens"
    if finish_reason:
        return "finish_reason_other"
    return "unknown_empty"


def build_vertex_native_success_diagnostic(body: Any, payload: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(body, dict):
        return None
    candidates = body.get("candidates") if isinstance(body.get("candidates"), list) else None
    prompt_feedback = body.get("promptFeedback") if isinstance(body.get("promptFeedback"), dict) else {}
    usage = body.get("usageMetadata") if isinstance(body.get("usageMetadata"), dict) else {}
    if candidates is None and not prompt_feedback and not usage:
        return None

    first_candidate = candidates[0] if candidates else {}
    if not isinstance(first_candidate, dict):
        first_candidate = {}
    content = first_candidate.get("content") if isinstance(first_candidate.get("content"), dict) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    raw_safety_ratings = first_candidate.get("safetyRatings", [])
    if not isinstance(raw_safety_ratings, list):
        raw_safety_ratings = []
    safety_ratings = [
        rating
        for rating in (sanitize_vertex_safety_rating(entry) for entry in raw_safety_ratings)
        if rating
    ]
    generation_config = payload.get("generationConfig") if isinstance(payload.get("generationConfig"), dict) else {}
    max_output_tokens = payload.get("max_tokens") or generation_config.get("maxOutputTokens")
    diagnostic = {
        "candidates_count": len(candidates or []),
        "candidate_finish_reason": clamp_vertex_proxy_string(first_candidate.get("finishReason", ""), 80),
        "content_parts_count": len(parts),
        "output_text_chars": count_vertex_text_chars(parts),
        "prompt_feedback_block_reason": clamp_vertex_proxy_string(prompt_feedback.get("blockReason", ""), 120),
        "safety_ratings": safety_ratings,
        "prompt_tokens": usage.get("promptTokenCount"),
        "completion_tokens": usage.get("candidatesTokenCount"),
        "output_tokens": usage.get("candidatesTokenCount"),
        "total_tokens": usage.get("totalTokenCount"),
        "maxOutputTokens": max_output_tokens,
    }
    sanitized = {key: value for key, value in diagnostic.items() if value not in (None, "", [], {})}
    empty_reason = infer_vertex_empty_reason(sanitized)
    if empty_reason:
        sanitized["empty_reason"] = empty_reason
    return sanitized or None


def normalize_vertex_proxy_success_response(response: Any, payload: dict[str, Any]) -> tuple[bytes, str | None]:
    content_type = response.headers.get("Content-Type")
    if payload.get("stream"):
        return response.content, content_type
    try:
        body = response.json()
    except ValueError:
        return response.content, content_type
    diagnostic = build_vertex_native_success_diagnostic(body, payload)
    if not diagnostic or not isinstance(body, dict):
        return response.content, content_type
    body["bgh_vertex_diagnostic"] = diagnostic
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), "application/json"


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB webm upload cap.
CORS(app, origins=CORS_ORIGINS)

chroma_client = None
world_collection = None
npc_memory_collection = None
approved_retrieval_collections: dict[str, Any] = {}
world_knowledge_collections: dict[str, Any] = {}
gm_bot_knowledge_collections: dict[str, Any] = {}
chroma_lock = threading.Lock()

whisper_model = None
whisper_lock = threading.Lock()

local_embedder = None
local_embedder_lock = threading.Lock()
local_reranker = None
local_reranker_lock = threading.Lock()


class SimpleGraph:
    """Small undirected graph store used to avoid bundling extra dependencies."""

    def __init__(self) -> None:
        self.adj: dict[str, dict[str, dict[str, str | None]]] = {}

    def add_edge(self, source: str, target: str, relation: str | None = None) -> None:
        self.adj.setdefault(source, {})
        self.adj.setdefault(target, {})
        self.adj[source][target] = {"relation": relation}
        self.adj[target][source] = {"relation": relation}

    def nodes(self) -> list[str]:
        return list(self.adj.keys())

    def neighbors(self, node: str) -> list[str]:
        return list(self.adj.get(node, {}).keys())

    def get_edge_data(self, source: str, target: str) -> dict[str, str | None]:
        return self.adj.get(source, {}).get(target, {})

    def number_of_nodes(self) -> int:
        return len(self.adj)

    def number_of_edges(self) -> int:
        return sum(len(neighbors) for neighbors in self.adj.values()) // 2

    def to_dict(self) -> dict[str, dict[str, dict[str, str | None]]]:
        return self.adj

    def from_dict(self, data: dict[str, dict[str, dict[str, str | None]]]) -> None:
        self.adj = data if isinstance(data, dict) else {}


graph = SimpleGraph()
graph_lock = threading.Lock()


def load_graph() -> None:
    if not GRAPH_FILE.exists():
        return

    try:
        with GRAPH_FILE.open("r", encoding="utf-8") as handle:
            graph.from_dict(json.load(handle))
    except Exception as exc:
        app.logger.warning("Graph load failed; starting with an empty graph: %s", exc)


def save_graph() -> None:
    GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with GRAPH_FILE.open("w", encoding="utf-8") as handle:
        json.dump(graph.to_dict(), handle, ensure_ascii=False, indent=2)


def require_auth(handler: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(handler)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if request.method == "OPTIONS":
            return "", 204

        expected_key = resolve_companion_auth_key()
        if not expected_key:
            return handler(*args, **kwargs)

        if request.headers.get("Authorization", "") != f"Bearer {expected_key}":
            return jsonify({"error": "Unauthorized."}), 401

        return handler(*args, **kwargs)

    return wrapper


def legacy_endpoint_status() -> dict[str, bool]:
    return {
        "rag_enabled": COMPANION_LEGACY_RAG_ENABLED,
        "memory_enabled": COMPANION_LEGACY_MEMORY_ENABLED,
        "graph_enabled": COMPANION_LEGACY_GRAPH_ENABLED,
    }


def legacy_disabled_response(feature: str, env_name: str) -> tuple[Response, int]:
    return jsonify(
        {
            "error": "Legacy companion endpoint disabled.",
            "feature": feature,
            "enable_env": env_name,
        }
    ), 403


def json_payload() -> dict[str, Any]:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def get_voyage_config_status() -> dict[str, Any]:
    return {
        "voyage_configured": bool(resolve_voyage_api_key()),
        "model": VOYAGE_EMBEDDING_MODEL,
        "output_dimension": VOYAGE_EMBEDDING_OUTPUT_DIMENSION,
        "input_types": ["document", "query"],
        "truncation": True,
        "local_embedding_enabled": LOCAL_EMBEDDING_ENABLED,
        "local_embedding_model": LOCAL_EMBEDDING_MODEL if LOCAL_EMBEDDING_ENABLED else None,
        "embedding_backend": resolve_embedding_backend(),
    }


def get_chroma_client() -> Any:
    global chroma_client

    if chroma_client:
        return chroma_client

    try:
        import chromadb
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError("chromadb is not installed. Install requirements-basic.txt.") from exc

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return chroma_client


# --- Per-world collection isolation -----------------------------------------
# Multiple Foundry worlds may share one companion server. To keep their indexes
# from cross-contaminating, world-scoped collections are namespaced by the
# Foundry world id sent on the X-World-Id request header (see backend-client.js
# makeHeaders). The legacy bgh_world_rag / bgh_npc_memories collections
# (default-OFF) are intentionally NOT isolated — /health and setup-doctor reach
# them without a world header.

WORLD_ID_HEADER = "X-World-Id"


def sanitize_world_suffix(value: Any) -> str:
    """Map a Foundry world id to a Chroma-collection-name-safe suffix.

    Chroma names allow ASCII [A-Za-z0-9._-] and must start/end alphanumeric, so
    every other character becomes '-' and leading/trailing separators are
    trimmed. Returns '' when nothing usable remains.
    """

    suffix = re.sub(r"[^A-Za-z0-9_-]", "-", str(value or "").strip())
    return suffix.strip("-_.")


def request_world_id() -> str:
    """Sanitized world id from the X-World-Id request header ('' when absent)."""

    return sanitize_world_suffix(request.headers.get(WORLD_ID_HEADER, ""))


def scoped_collection_name(base: str, world_id: str) -> str:
    """Per-world collection name '<base>__<world_id>'.

    Raises when world_id is blank so a world-scoped route can never silently
    fall back to a shared collection (fail-closed).
    """

    if not world_id:
        raise ValueError("world_id is required for a world-scoped collection name")
    return f"{base}__{world_id}"


def require_world_scope() -> tuple[str, Any]:
    """Fail-closed world-scope gate for route handlers.

    Returns (world_id, None) on success, or ('', error_response) when the
    X-World-Id header is missing/blank.
    """

    world_id = request_world_id()
    if not world_id:
        return "", (
            jsonify(
                {
                    "ok": False,
                    "status": "world_required",
                    "error": "X-World-Id header is required for world-scoped routes.",
                }
            ),
            400,
        )
    return world_id, None


def get_chroma_collections() -> tuple[Any, Any]:
    global chroma_client, world_collection, npc_memory_collection

    with chroma_lock:
        if chroma_client and world_collection and npc_memory_collection:
            return world_collection, npc_memory_collection

        client = get_chroma_client()
        world_collection = client.get_or_create_collection(
            name=RAG_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        npc_memory_collection = client.get_or_create_collection(
            name=MEMORY_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return world_collection, npc_memory_collection


def list_chroma_collection_names_without_create() -> list[str]:
    """List existing Chroma collection names without creating module collections."""

    if not CHROMA_PATH.exists():
        return []

    try:
        client = get_chroma_client()
        names: list[str] = []
        for collection in client.list_collections():
            names.append(getattr(collection, "name", str(collection)))
        return sorted(set(names))
    except Exception:
        return []


def get_chroma_collection_count_without_create(name: str) -> int | None:
    """Return the count for an existing collection without creating it."""

    if name not in list_chroma_collection_names_without_create():
        return None

    try:
        client = get_chroma_client()
        return int(client.get_collection(name).count())
    except Exception:
        return None


def get_approved_retrieval_collection(world_id: str) -> Any:
    """Return/create the per-world approved-only retrieval collection.

    This helper is intentionally separate from get_chroma_collections(), because
    that legacy helper also creates world and NPC memory collections. Cached per
    world id so separate Foundry worlds never share approved-memory records
    (creates bgh_approved_memory_v1__<world_id>).
    """

    with chroma_lock:
        existing = approved_retrieval_collections.get(world_id)
        if existing is not None:
            return existing

        client = get_chroma_client()
        collection = client.get_or_create_collection(
            name=scoped_collection_name(APPROVED_RETRIEVAL_COLLECTION_NAME, world_id),
            metadata={"hnsw:space": "cosine"},
        )
        approved_retrieval_collections[world_id] = collection
        return collection


def get_world_knowledge_collection(world_id: str) -> Any:
    """Return/create the per-world NPC knowledge model collection.

    Intentionally separate from get_chroma_collections(), which also creates
    legacy world and NPC memory collections. Cached per world id so separate
    Foundry worlds never share the world-knowledge index.
    """

    with chroma_lock:
        existing = world_knowledge_collections.get(world_id)
        if existing is not None:
            return existing

        client = get_chroma_client()
        collection = client.get_or_create_collection(
            name=scoped_collection_name(RED_WORLD_KNOWLEDGE_COLLECTION_NAME, world_id),
            metadata={"hnsw:space": "cosine"},
        )
        world_knowledge_collections[world_id] = collection
        return collection


def get_gm_bot_knowledge_collection(world_id: str) -> Any:
    """Return/create the per-world GM bot campaign canon knowledge collection.

    Cached per world id so separate Foundry worlds never share the GM-bot canon
    index (creates bgh_RED_gm_bot_knowledge__<world_id>).
    """

    with chroma_lock:
        existing = gm_bot_knowledge_collections.get(world_id)
        if existing is not None:
            return existing

        client = get_chroma_client()
        collection = client.get_or_create_collection(
            name=scoped_collection_name(GM_BOT_KNOWLEDGE_COLLECTION_NAME, world_id),
            metadata={"hnsw:space": "cosine"},
        )
        gm_bot_knowledge_collections[world_id] = collection
        return collection


def get_approved_retrieval_status(collection_name: str) -> dict[str, Any]:
    names = list_chroma_collection_names_without_create()
    collection_exists = collection_name in names
    collection_count: int | None = None
    collection_error: str | None = None

    if collection_exists:
        try:
            client = get_chroma_client()
            collection_count = int(client.get_collection(collection_name).count())
        except Exception as exc:
            collection_error = str(exc)

    return {
        **get_voyage_config_status(),
        "approved_collection_name": collection_name,
        "approved_collection_exists": collection_exists,
        "approved_collection_count": collection_count,
        "approved_collection_error": collection_error,
        **get_approved_retrieval_dummy_status(collection_name),
        "legacy_rag_enabled": COMPANION_LEGACY_RAG_ENABLED,
        "legacy_memory_enabled": COMPANION_LEGACY_MEMORY_ENABLED,
        "legacy_graph_enabled": COMPANION_LEGACY_GRAPH_ENABLED,
    }


def string_or_empty(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def chroma_metadata_value(value: Any) -> str | int | float | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def get_nested_string(data: dict[str, Any], path: list[str]) -> str:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return ""
        current = current.get(key)
    return string_or_empty(current)


def get_approved_record_text(record: dict[str, Any]) -> str:
    text = string_or_empty(record.get("text"))
    return text or string_or_empty(record.get("body"))


def validate_approved_retrieval_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {"ok": False, "errors": ["record must be an object"]}

    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    provenance = metadata.get("provenance") if isinstance(metadata.get("provenance"), dict) else {}
    text = get_approved_record_text(record)
    required_values = {
        "entry_id": string_or_empty(metadata.get("entry_id")) or string_or_empty(record.get("entry_id")),
        "candidate_id": string_or_empty(provenance.get("candidate_id")) or string_or_empty(record.get("candidate_id")),
        "entry_type": string_or_empty(metadata.get("entry_type")) or string_or_empty(record.get("entry_type")),
        "decided_at": string_or_empty(provenance.get("decided_at")) or string_or_empty(record.get("decided_at")),
        "decided_by_user_id": string_or_empty(provenance.get("decided_by_user_id")) or string_or_empty(record.get("decided_by_user_id")),
        "stored_at": string_or_empty(provenance.get("stored_at")) or string_or_empty(record.get("stored_at")),
        "stored_by_user_id": string_or_empty(provenance.get("stored_by_user_id")) or string_or_empty(record.get("stored_by_user_id")),
        "body_hash": string_or_empty(metadata.get("body_hash")) or string_or_empty(record.get("body_hash")),
        "text": text,
    }
    knowledge_basis_present = (
        "knowledge_basis" in metadata
        or "knowledge_basis" in record
    )
    errors = [f"{field} is required" for field, value in required_values.items() if not value]

    # target_actor_id/uuid is optional: central Vault store entries are indexed
    # globally (no owning actor) and holder scoping is applied client-side at NPC
    # search time. Legacy per-actor records still carry target_actor_* and stay
    # valid; filter_approved_retrieval_result() pass-through handles target-less
    # records for the NPC audience.
    if not knowledge_basis_present:
        errors.append("knowledge_basis is required")

    return {
        "ok": not errors,
        "errors": errors,
        "text": text,
        "metadata": metadata,
        "provenance": provenance,
    }


def build_approved_retrieval_metadata(record: dict[str, Any]) -> dict[str, str | int | float | bool]:
    validation = validate_approved_retrieval_record(record)
    if not validation["ok"]:
        raise ValueError("; ".join(validation["errors"]))

    metadata = validation["metadata"]
    provenance = validation["provenance"]
    actor = record.get("actor") if isinstance(record.get("actor"), dict) else {}
    source_refs_summary = provenance.get("source_refs_summary")

    return {
        "entry_id": chroma_metadata_value(metadata.get("entry_id") or record.get("entry_id")),
        "candidate_id": chroma_metadata_value(provenance.get("candidate_id") or record.get("candidate_id")),
        "entry_type": chroma_metadata_value(metadata.get("entry_type") or record.get("entry_type")),
        "target_actor_id": chroma_metadata_value(metadata.get("target_actor_id") or record.get("target_actor_id")),
        "target_actor_uuid": chroma_metadata_value(metadata.get("target_actor_uuid") or record.get("target_actor_uuid")),
        "actor_id": chroma_metadata_value(actor.get("id") or record.get("actor_id")),
        "actor_uuid": chroma_metadata_value(actor.get("uuid") or record.get("actor_uuid")),
        "actor_name": chroma_metadata_value(actor.get("name") or record.get("actor_name")),
        "knowledge_basis": chroma_metadata_value(metadata.get("knowledge_basis", record.get("knowledge_basis"))),
        "decided_at": chroma_metadata_value(provenance.get("decided_at") or record.get("decided_at")),
        "decided_by_user_id": chroma_metadata_value(provenance.get("decided_by_user_id") or record.get("decided_by_user_id")),
        "stored_at": chroma_metadata_value(provenance.get("stored_at") or record.get("stored_at")),
        "stored_by_user_id": chroma_metadata_value(provenance.get("stored_by_user_id") or record.get("stored_by_user_id")),
        "body_hash": chroma_metadata_value(metadata.get("body_hash") or record.get("body_hash")),
        "source_refs_count": chroma_metadata_value(
            source_refs_summary.get("count", 0) if isinstance(source_refs_summary, dict) else 0
        ),
        "source_refs_summary": chroma_metadata_value(source_refs_summary if isinstance(source_refs_summary, dict) else {}),
        "anchor_quotes_count": chroma_metadata_value(provenance.get("anchor_quotes_count", 0)),
    }


def approved_retrieval_record_id(record: dict[str, Any]) -> str:
    record_id = string_or_empty(record.get("record_id")) or string_or_empty(record.get("id"))
    validation = validate_approved_retrieval_record(record)
    metadata = validation.get("metadata") if isinstance(validation.get("metadata"), dict) else {}
    entry_id = string_or_empty(metadata.get("entry_id")) or string_or_empty(record.get("entry_id"))
    candidate_id = string_or_empty(record.get("candidate_id"))
    return record_id or f"approved-retrieval:{entry_id or candidate_id}"


def clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def normalize_world_knowledge_metadata(metadata: Any) -> dict[str, str | int | float | bool]:
    if not isinstance(metadata, dict):
        return {}

    # Chroma metadata values must be scalar; list/dict values are JSON-flattened.
    normalized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        key_name = key.strip() if isinstance(key, str) else str(key).strip()
        if not key_name:
            continue
        normalized[key_name] = chroma_metadata_value(value)
    return normalized


def normalize_world_knowledge_axes(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        return ["world"]

    axes: list[str] = []
    seen: set[str] = set()
    for item in value:
        axis = string_or_empty(item)
        if not axis or axis in seen:
            continue
        axes.append(axis)
        seen.add(axis)
    return axes or ["world"]


def build_world_knowledge_where(axes: list[str], where_extra: Any) -> dict[str, Any]:
    faction_id_filter = None
    extra_conditions: list[dict[str, Any]] = []

    if isinstance(where_extra, dict):
        for key, value in where_extra.items():
            key_name = key.strip() if isinstance(key, str) else str(key).strip()
            if not key_name:
                continue
            if key_name == "faction_id":
                faction_id_filter = value
            else:
                # Preserve Chroma operators such as {"$in": [...]} in query filters.
                extra_conditions.append({key_name: value})

    world_axes = [axis for axis in axes if axis == "world"]
    non_world_axes = [axis for axis in axes if axis != "world"]

    if faction_id_filter is not None and world_axes and non_world_axes:
        world_branch: dict[str, Any] = {"knowledge_axis": {"$in": world_axes}}
        for condition in extra_conditions:
            world_branch = {"$and": [world_branch, condition]}

        faction_branch: dict[str, Any] = {
            "$and": [
                {"knowledge_axis": {"$in": non_world_axes}},
                {"faction_id": faction_id_filter},
            ]
        }
        for condition in extra_conditions:
            faction_branch = {"$and": [faction_branch, condition]}

        return {"$or": [world_branch, faction_branch]}

    conditions: list[dict[str, Any]] = [{"knowledge_axis": {"$in": axes}}]
    if faction_id_filter is not None:
        conditions.append({"faction_id": faction_id_filter})
    for condition in extra_conditions:
        conditions.append(condition)

    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def approved_preview_text(document: Any, max_length: int = 240) -> str:
    text = str(document or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= max_length:
        return text
    return f"{text[:max_length - 1].rstrip()}..."


def normalize_approved_metadata(metadata: Any) -> dict[str, Any]:
    return metadata if isinstance(metadata, dict) else {}


def approved_metadata_string(metadata: dict[str, Any], key: str) -> str:
    return string_or_empty(metadata.get(key))


def approved_value_contains_dummy_marker(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return APPROVED_RETRIEVAL_DUMMY_MARKER in value
    try:
        return APPROVED_RETRIEVAL_DUMMY_MARKER in json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return False


def approved_result_is_dummy(metadata: dict[str, Any], document: Any) -> bool:
    return approved_value_contains_dummy_marker(document) or approved_value_contains_dummy_marker(metadata)


def get_approved_retrieval_dummy_status(collection_name: str, max_scan: int = 200) -> dict[str, Any]:
    status = {
        "dummy_marker": APPROVED_RETRIEVAL_DUMMY_MARKER,
        "dummy_records_detected": 0,
        "dummy_warning": "",
    }

    try:
        names = list_chroma_collection_names_without_create()
        if collection_name not in names:
            return status

        client = get_chroma_client()
        collection = client.get_collection(collection_name)
        count = int(collection.count())
        if count <= 0:
            return status

        try:
            records = collection.get(
                limit=min(max_scan, count),
                include=["documents", "metadatas"],
            )
        except TypeError:
            records = collection.get(include=["documents", "metadatas"])

        documents = records.get("documents", []) if isinstance(records, dict) else []
        metadatas = records.get("metadatas", []) if isinstance(records, dict) else []
        dummy_count = 0
        for index, document in enumerate(documents):
            metadata = normalize_approved_metadata(metadatas[index] if index < len(metadatas) else {})
            if approved_result_is_dummy(metadata, document):
                dummy_count += 1

        status["dummy_records_detected"] = dummy_count
        if dummy_count > 0:
            status["dummy_warning"] = (
                "Dummy approved retrieval record remains. Remove/replace before 025-E4-D or real-session use."
            )
        return status
    except Exception as exc:
        status["dummy_warning"] = f"Dummy approved retrieval marker scan failed: {exc}"
        return status


def filter_approved_retrieval_result(
    metadata: dict[str, Any],
    audience: str,
    actor_id: str,
    actor_uuid: str,
) -> tuple[bool, str]:
    target_actor_id = approved_metadata_string(metadata, "target_actor_id")
    target_actor_uuid = approved_metadata_string(metadata, "target_actor_uuid")
    target_matches = (
        (actor_id and target_actor_id and actor_id == target_actor_id)
        or (actor_uuid and target_actor_uuid and actor_uuid == target_actor_uuid)
    )
    actor_requested = bool(actor_id or actor_uuid)
    target_present = bool(target_actor_id or target_actor_uuid)

    if audience == "gm":
        return True, ""

    if actor_requested and target_present and not target_matches:
        return False, "target_mismatch_excluded_for_npc"

    return True, ""


def approved_retrieval_result_payload(
    result_id: str,
    distance: Any,
    metadata: dict[str, Any],
    document: Any,
) -> dict[str, Any]:
    return {
        "id": result_id,
        "distance": distance,
        "entry_id": approved_metadata_string(metadata, "entry_id"),
        "candidate_id": approved_metadata_string(metadata, "candidate_id"),
        "entry_type": approved_metadata_string(metadata, "entry_type"),
        "actor_name": approved_metadata_string(metadata, "actor_name"),
        "target_actor_id": approved_metadata_string(metadata, "target_actor_id"),
        "target_actor_uuid": approved_metadata_string(metadata, "target_actor_uuid"),
        "body_hash": approved_metadata_string(metadata, "body_hash"),
        "body_hash_status": "not_revalidated_against_foundry_flags",
        "dummy_record": approved_result_is_dummy(metadata, document),
        "preview": approved_preview_text(document),
    }


def resolve_voyage_api_key() -> str:
    """Per-request Voyage API key supplied by the Foundry module via the
    X-Voyage-Api-Key header (voyageApiKey setting). UI-only: no env fallback."""
    try:
        return (request.headers.get("X-Voyage-Api-Key", "") or "").strip()
    except RuntimeError:
        return ""


def get_voyage_embeddings(texts: list[str], input_type: str = "document") -> list[list[float]]:
    voyage_key = resolve_voyage_api_key()
    if not voyage_key:
        raise RuntimeError("Voyage API key is not provided by the module settings.")

    if input_type not in {"document", "query"}:
        raise ValueError("input_type must be 'document' or 'query'.")

    if not isinstance(texts, list) or not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("texts must be a non-empty list of non-empty strings.")

    payload = json.dumps(
        {
            "input": texts,
            "model": VOYAGE_EMBEDDING_MODEL,
            "input_type": input_type,
            "output_dimension": VOYAGE_EMBEDDING_OUTPUT_DIMENSION,
            "truncation": True,
        }
    ).encode("utf-8")
    request_data = urllib.request.Request(
        VOYAGE_EMBEDDING_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {voyage_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request_data, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Voyage embedding request failed with HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Voyage embedding request failed.") from exc

    embeddings = data.get("data")
    if not isinstance(embeddings, list):
        raise RuntimeError("Voyage embedding response did not include data.")

    return [item["embedding"] for item in embeddings if isinstance(item, dict) and "embedding" in item]


def _resolve_local_model_ref() -> str:
    """Prefer a bundled offline snapshot over the Hugging Face hub id.

    A frozen build ships the model under <bundle>/models/<name>; loading from that
    local path keeps the server fully offline (no multi-GB download on the first
    query). In a plain dev checkout the bundled dir is absent, so we fall back to
    the hub id, which sentence-transformers downloads + caches on first use.
    LOCAL_EMBEDDING_MODEL_PATH, when set to an existing dir, overrides both.
    """
    explicit = os.getenv("LOCAL_EMBEDDING_MODEL_PATH", "").strip()
    if explicit and Path(explicit).is_dir():
        return explicit

    bundle_base = getattr(sys, "_MEIPASS", "") or str(Path(__file__).resolve().parent)
    bundled = Path(bundle_base) / "models" / Path(LOCAL_EMBEDDING_MODEL).name
    if bundled.is_dir():
        # Bundled snapshot present -> force offline so transformers never reaches the hub.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return str(bundled)

    return LOCAL_EMBEDDING_MODEL


def _resolve_local_stt_ref() -> str:
    """Prefer a bundled offline Whisper CT2 snapshot over the faster-whisper alias.

    A frozen build ships the model under <bundle>/models/<repo-name>; loading from
    that local directory keeps STT fully offline (no ~1.6GB download on first use).
    In a plain dev checkout the bundled dir is absent, so we return STT_MODEL (an
    alias like 'large-v3-turbo'), which faster-whisper downloads + caches on first
    use. STT_MODEL_PATH, when set to an existing dir, overrides both.
    """
    explicit = os.getenv("STT_MODEL_PATH", "").strip()
    if explicit and Path(explicit).is_dir():
        return explicit

    bundle_base = getattr(sys, "_MEIPASS", "") or str(Path(__file__).resolve().parent)
    bundled = Path(bundle_base) / "models" / Path(STT_HF_REPO).name
    if bundled.is_dir():
        # Bundled snapshot present -> force offline so faster-whisper never reaches the hub.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        return str(bundled)

    return STT_MODEL


def get_local_embedder() -> Any:
    """Lazily load the local sentence-transformers embedding model.

    Off unless LOCAL_EMBEDDING_ENABLED; the heavy deps (sentence-transformers +
    torch) live in requirements-local-embed.txt and ship only in the local edition.
    """
    global local_embedder

    if not LOCAL_EMBEDDING_ENABLED:
        raise RuntimeError("Local embedding is disabled. Set LOCAL_EMBEDDING_ENABLED=true to enable it.")

    with local_embedder_lock:
        if local_embedder:
            return local_embedder

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "sentence-transformers is not installed. Install requirements-local-embed.txt."
            ) from exc

        local_embedder = SentenceTransformer(
            _resolve_local_model_ref(),
            trust_remote_code=True,
            truncate_dim=LOCAL_EMBEDDING_DIMENSION,
            device=LOCAL_EMBEDDING_DEVICE,
        )
        return local_embedder


def get_local_embeddings(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Embed texts with the bundled open-weight model (voyage-4-nano), no API key.

    Mirrors get_voyage_embeddings' contract (same dimension, document/query input
    types) so callers stay backend-agnostic. voyage-4-nano shares an embedding
    space with voyage-4-large, so these vectors drop into an existing index.
    """
    if input_type not in {"document", "query"}:
        raise ValueError("input_type must be 'document' or 'query'.")

    if not isinstance(texts, list) or not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("texts must be a non-empty list of non-empty strings.")

    model = get_local_embedder()
    # encode_query / encode_document apply the model's task prompts and L2-normalize
    # automatically; truncate_dim was fixed at load time to match the index.
    vectors = model.encode_query(texts) if input_type == "query" else model.encode_document(texts)
    return [[float(value) for value in row] for row in vectors]


def resolve_embedding_backend() -> str | None:
    """Choose the embedding backend for the current request.

    Voyage (BYOK) wins whenever the module supplies a key; otherwise the local
    open-weight model is the fallback when enabled. Returns "voyage", "local",
    or None when no backend is usable.
    """
    if resolve_voyage_api_key():
        return "voyage"
    if LOCAL_EMBEDDING_ENABLED:
        return "local"
    return None


def embedding_backend_available() -> bool:
    return resolve_embedding_backend() is not None


def active_embedding_model() -> str:
    """Model id for the backend that would serve the current request, so response
    payloads report the real model (e.g. the local nano) instead of always the
    Voyage one."""
    return LOCAL_EMBEDDING_MODEL if resolve_embedding_backend() == "local" else VOYAGE_EMBEDDING_MODEL


def embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Backend-agnostic embedding entry point used by every embed/index/search route."""
    backend = resolve_embedding_backend()
    if backend == "voyage":
        return get_voyage_embeddings(texts, input_type=input_type)
    if backend == "local":
        return get_local_embeddings(texts, input_type=input_type)
    raise RuntimeError(
        "No embedding backend available: provide a Voyage API key or enable the local embedding model."
    )


def _normalize_rerank_inputs(query: Any, documents: Any) -> tuple[str, list[str]]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if not isinstance(documents, list) or not documents or not all(
        isinstance(document, str) and document.strip() for document in documents
    ):
        raise ValueError("documents must be a non-empty list of non-empty strings.")
    return query, documents


def get_voyage_rerank(
    query: str, documents: list[str], model: str | None = None, top_k: int | None = None
) -> list[dict[str, Any]]:
    voyage_key = resolve_voyage_api_key()
    if not voyage_key:
        raise RuntimeError("Voyage API key is not provided by the module settings.")
    query, documents = _normalize_rerank_inputs(query, documents)

    body: dict[str, Any] = {
        "query": query,
        "documents": documents,
        "model": model or VOYAGE_RERANK_MODEL,
        "return_documents": False,
    }
    if isinstance(top_k, int) and top_k > 0:
        body["top_k"] = top_k
    payload = json.dumps(body).encode("utf-8")
    request_data = urllib.request.Request(
        VOYAGE_RERANK_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {voyage_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request_data, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Voyage rerank request failed with HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Voyage rerank request failed.") from exc

    results = data.get("data")
    if not isinstance(results, list):
        raise RuntimeError("Voyage rerank response did not include data.")
    ranked: list[dict[str, Any]] = []
    for item in results:
        if isinstance(item, dict) and "index" in item:
            ranked.append(
                {"index": int(item["index"]), "relevance_score": float(item.get("relevance_score", 0.0))}
            )
    return ranked


def _resolve_local_rerank_ref() -> str:
    """Prefer a bundled offline reranker snapshot over the Hugging Face hub id."""
    explicit = os.getenv("LOCAL_RERANK_MODEL_PATH", "").strip()
    if explicit and Path(explicit).is_dir():
        return explicit
    bundle_base = getattr(sys, "_MEIPASS", "") or str(Path(__file__).resolve().parent)
    bundled = Path(bundle_base) / "models" / Path(LOCAL_RERANK_MODEL).name
    if bundled.is_dir():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return str(bundled)
    return LOCAL_RERANK_MODEL


def get_local_reranker() -> Any:
    """Lazily load the local open-weight cross-encoder reranker (mxbai-rerank-v2)."""
    global local_reranker

    if not LOCAL_RERANK_ENABLED:
        raise RuntimeError("Local rerank is disabled. Set LOCAL_RERANK_ENABLED=true to enable it.")

    with local_reranker_lock:
        if local_reranker:
            return local_reranker
        try:
            from mxbai_rerank import MxbaiRerankV2
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "mxbai-rerank is not installed. Install requirements-local-rerank.txt."
            ) from exc
        local_reranker = MxbaiRerankV2(_resolve_local_rerank_ref(), device=LOCAL_RERANK_DEVICE)
        return local_reranker


def get_local_rerank(
    query: str, documents: list[str], top_k: int | None = None
) -> list[dict[str, Any]]:
    query, documents = _normalize_rerank_inputs(query, documents)
    model = get_local_reranker()
    limit = top_k if isinstance(top_k, int) and top_k > 0 else len(documents)
    ranked = model.rank(query, documents, return_documents=False, top_k=limit)
    out: list[dict[str, Any]] = []
    for item in ranked:
        index = getattr(item, "index", None)
        if index is None and isinstance(item, dict):
            index = item.get("index")
        score = getattr(item, "score", None)
        if score is None and isinstance(item, dict):
            score = item.get("score", item.get("relevance_score", 0.0))
        if index is None:
            continue
        out.append({"index": int(index), "relevance_score": float(score or 0.0)})
    return out


def resolve_rerank_backend() -> str | None:
    """Voyage (BYOK) wins when a key is supplied; else the local reranker if enabled."""
    if resolve_voyage_api_key():
        return "voyage"
    if LOCAL_RERANK_ENABLED:
        return "local"
    return None


def rerank_available() -> bool:
    return resolve_rerank_backend() is not None


def rerank_documents(
    query: str,
    documents: list[str],
    top_k: int | None = None,
    model: str | None = None,
    backend: str | None = None,
) -> list[dict[str, Any]]:
    """Backend-agnostic rerank entry. `backend` forces a specific path; None auto-picks."""
    resolved = backend or resolve_rerank_backend()
    if resolved == "voyage":
        return get_voyage_rerank(query, documents, model=model, top_k=top_k)
    if resolved == "local":
        return get_local_rerank(query, documents, top_k=top_k)
    raise RuntimeError(
        "No rerank backend available: provide a Voyage API key or enable the local rerank model."
    )


def list_chroma_collection_names() -> list[str]:
    """Return collection names without creating legacy module collections."""

    return list_chroma_collection_names_without_create()


def bool_from_payload(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return None


def int_from_payload(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_record_preview(document: Any, max_length: int = 180) -> str:
    text = str(document or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= max_length:
        return text
    return f"{text[:max_length - 1].rstrip()}..."


def build_audit_preview_payload(document: Any, metadata: dict[str, Any], max_length: int = APPROVED_RETRIEVAL_AUDIT_PREVIEW_MAX_CHARS) -> dict[str, Any]:
    metadata_preview = ""
    if isinstance(metadata, dict):
        for key in (
            "preview",
            "preview_text",
            "text_preview",
            "previewText",
            "summary",
            "summary_text",
            "text",
            "body",
        ):
            value = string_or_empty(metadata.get(key))
            if value:
                metadata_preview = value
                break

    if not metadata_preview and isinstance(metadata, dict):
        metadata_parts = [
            string_or_empty(metadata.get("entry_id")),
            string_or_empty(metadata.get("body_hash")),
            string_or_empty(metadata.get("entry_type")),
        ]
        metadata_preview = " / ".join(part for part in metadata_parts if part)

    document_text = str(document or "")
    if document_text:
        sanitized_document = document_text.replace("\r", " ").replace("\n", " ")
        body_chars = len(document_text)
        preview_text = safe_record_preview(sanitized_document, max_length=max_length)
        preview_chars = len(preview_text)
        return {
            "preview_source": APPROVED_RETRIEVAL_AUDIT_PREVIEW_SOURCE_INDEX_RECORD,
            "preview_text": preview_text,
            "preview_chars": preview_chars,
            "preview_truncated": body_chars > max_length,
            "body_chars": body_chars,
            "document_chars": body_chars,
            "body_available": True,
        }

    preview_text = safe_record_preview(metadata_preview, max_length=max_length)
    preview_chars = len(preview_text) if preview_text else 0
    return {
        "preview_source": APPROVED_RETRIEVAL_AUDIT_PREVIEW_SOURCE_METADATA if metadata_preview else "unknown",
        "preview_text": preview_text,
        "preview_chars": preview_chars,
        "preview_truncated": None,
        "body_chars": None,
        "document_chars": None,
        "body_available": False,
    }


def is_active_world_scoped_collection(name: str) -> bool:
    """True for an active world-scoped collection of ANY world.

    Matches the three isolated bases (approved-retrieval / world-knowledge /
    gm-bot-knowledge) and their `<base>__<world_id>` per-world variants. Used to
    keep one world's purge/audit from ever touching another world's live data.
    """

    normalized = name.strip().lower()
    for base in (
        APPROVED_RETRIEVAL_COLLECTION_NAME,
        RED_WORLD_KNOWLEDGE_COLLECTION_NAME,
        GM_BOT_KNOWLEDGE_COLLECTION_NAME,
    ):
        base_lower = base.lower()
        if normalized == base_lower or normalized.startswith(base_lower + "__"):
            return True
    return False


def is_legacy_or_old_collection(name: str) -> bool:
    normalized = name.strip().lower()
    if not normalized or is_active_world_scoped_collection(normalized):
        return False
    explicit = {
        RAG_COLLECTION_NAME.lower(),
        MEMORY_COLLECTION_NAME.lower(),
        LEGACY_RAG_COLLECTION_NAME.lower(),
        LEGACY_MEMORY_COLLECTION_NAME.lower(),
    }
    if normalized in explicit:
        return True
    return any(token in normalized for token in ["legacy", "old", "unapproved", "world_rag", "npc_memories"])


def collection_records_without_create(
    collection_name: str,
    *,
    ids: list[str] | None = None,
    limit: int = 300,
) -> dict[str, Any]:
    names = list_chroma_collection_names_without_create()
    if collection_name not in names:
        return {
            "ok": True,
            "exists": False,
            "collection": collection_name,
            "count": 0,
            "records": [],
            "error": "",
        }

    try:
        client = get_chroma_client()
        collection = client.get_collection(collection_name)
        count = int(collection.count())
        if count <= 0:
            return {
                "ok": True,
                "exists": True,
                "collection": collection_name,
                "count": 0,
                "records": [],
                "error": "",
            }

        get_kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
        if ids:
            get_kwargs["ids"] = ids
        else:
            get_kwargs["limit"] = min(max(1, limit), count)

        try:
            payload = collection.get(**get_kwargs)
        except TypeError:
            fallback_kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
            if ids:
                fallback_kwargs["ids"] = ids
            payload = collection.get(**fallback_kwargs)

        raw_ids = payload.get("ids", []) if isinstance(payload, dict) else []
        documents = payload.get("documents", []) if isinstance(payload, dict) else []
        metadatas = payload.get("metadatas", []) if isinstance(payload, dict) else []
        records: list[dict[str, Any]] = []
        for index, record_id in enumerate(raw_ids):
            metadata = normalize_approved_metadata(metadatas[index] if index < len(metadatas) else {})
            document = documents[index] if index < len(documents) else ""
            records.append(
                {
                    "id": str(record_id),
                    "metadata": metadata,
                    "document": document,
                    "preview": safe_record_preview(document),
                }
            )

        return {
            "ok": True,
            "exists": True,
            "collection": collection_name,
            "count": count,
            "records": records,
            "error": "",
        }
    except Exception as exc:
        return {
            "ok": False,
            "exists": True,
            "collection": collection_name,
            "count": None,
            "records": [],
            "error": str(exc),
        }


def normalize_audit_memories(payload: Any) -> list[dict[str, Any]]:
    memories = payload if isinstance(payload, list) else []
    normalized: list[dict[str, Any]] = []
    for memory in memories:
        if not isinstance(memory, dict):
            continue
        entry_id = string_or_empty(memory.get("entry_id")) or string_or_empty(memory.get("entryId"))
        body_hash = string_or_empty(memory.get("body_hash")) or string_or_empty(memory.get("bodyHash"))
        if not entry_id and not body_hash:
            continue
        normalized.append(
            {
                "memory_id": string_or_empty(memory.get("memory_id")) or string_or_empty(memory.get("memoryId")),
                "entry_id": entry_id,
                "title": string_or_empty(memory.get("title")) or entry_id,
                "lifecycle_state": string_or_empty(memory.get("lifecycle_state")) or string_or_empty(memory.get("lifecycleState")),
                "status": string_or_empty(memory.get("status")),
                "retrieval_eligible": bool_from_payload(memory.get("retrieval_eligible", memory.get("retrievalEligible"))),
                "prompt_eligible": bool_from_payload(memory.get("prompt_eligible", memory.get("promptEligible"))),
                "body_hash": body_hash,
                "entity_ref_count": int_from_payload(memory.get("entity_ref_count", memory.get("entityRefCount")), 0),
                "source_ref_count": int_from_payload(memory.get("source_ref_count", memory.get("sourceRefCount")), 0),
                "target_kind": string_or_empty(memory.get("target_kind")) or string_or_empty(memory.get("targetKind")),
                "index_state": string_or_empty(memory.get("index_state")) or string_or_empty(memory.get("indexState")),
                "index_record_id": string_or_empty(memory.get("index_record_id")) or string_or_empty(memory.get("indexRecordId")),
                "index_collection": string_or_empty(memory.get("index_collection")) or string_or_empty(memory.get("indexCollection")),
            }
        )
    return normalized


def memory_lookup_maps(memories: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_entry_id: dict[str, dict[str, Any]] = {}
    by_body_hash: dict[str, dict[str, Any]] = {}
    for memory in memories:
        if memory["entry_id"]:
            by_entry_id[memory["entry_id"]] = memory
        if memory["body_hash"]:
            by_body_hash[memory["body_hash"]] = memory
    return by_entry_id, by_body_hash


def metadata_bool(metadata: dict[str, Any], key: str) -> bool | None:
    return bool_from_payload(metadata.get(key))


def record_entry_id(metadata: dict[str, Any]) -> str:
    return (
        approved_metadata_string(metadata, "entry_id")
        or approved_metadata_string(metadata, "approved_memory_entry_id")
        or approved_metadata_string(metadata, "memory_id")
    )


def record_candidate_id(metadata: dict[str, Any]) -> str:
    return approved_metadata_string(metadata, "candidate_id") or approved_metadata_string(metadata, "source_candidate_id")


def record_body_hash(metadata: dict[str, Any]) -> str:
    return approved_metadata_string(metadata, "body_hash") or approved_metadata_string(metadata, "source_body_hash")


def document_looks_like_raw_dump(document: Any, metadata: dict[str, Any]) -> bool:
    text = str(document or "")
    lowered = text.lower()
    metadata_text = json.dumps(metadata, ensure_ascii=False, sort_keys=True).lower()
    raw_markers = [
        "journalentrypage",
        "journalentry",
        "actor.",
        "item.",
        "\"pages\"",
        "\"items\"",
        "\"system\"",
        "\"prototypeToken\"".lower(),
    ]
    if len(text) > 3500 and any(marker in lowered for marker in raw_markers):
        return True
    if lowered.strip().startswith("{") and any(marker in lowered for marker in raw_markers):
        return True
    return any(marker in metadata_text for marker in ["raw_journal", "raw_actor", "raw_item", "document_dump"])


def record_has_test_signal(record_id: str, metadata: dict[str, Any], document: Any) -> bool:
    text = " ".join(
        [
            record_id,
            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            str(document or "")[:1200],
        ]
    ).lower()
    return (
        approved_result_is_dummy(metadata, document)
        or any(token in text for token in ["dummy", "fixture", "smoke", "__test__", "lorem ipsum"])
    )


def make_audit_candidate(
    *,
    collection: str,
    record: dict[str, Any],
    canonical: dict[str, Any] | None,
    issues: list[str],
) -> dict[str, Any] | None:
    if not issues:
        return None

    metadata = normalize_approved_metadata(record.get("metadata"))
    record_id = string_or_empty(record.get("id"))
    entry_id = record_entry_id(metadata)
    candidate_id = record_candidate_id(metadata)
    issue_set = sorted(set(issues))
    high_markers = {
        "orphaned_index_record",
        "body_hash_mismatch",
        "dummy_or_test_index_record",
        "legacy_collection_record",
        "raw_document_dump_suspected",
        "unapproved_candidate_indexed",
    }
    medium_markers = {
        "non_active_lifecycle_indexed",
        "retrieval_excluded_indexed",
        "prompt_excluded_prompt_risk",
        "missing_entity_ref_prompt_risk",
    }
    risk = "high" if any(issue in high_markers for issue in issue_set) else "medium" if any(issue in medium_markers for issue in issue_set) else "low"
    if "body_hash_mismatch" in issue_set:
        body_hash_status = "mismatch"
    elif record_body_hash(metadata):
        body_hash_status = "present"
    else:
        body_hash_status = "missing"

    legacy_marker = "legacy_collection_record" in issue_set
    if canonical is None and not record_body_hash(metadata) and legacy_marker:
        recommended_action = "원본 공식 기억이 없는 검색 인덱스 기록입니다. 삭제 preview 후 개별 삭제를 검토하세요."
    elif canonical is None:
        recommended_action = "검색 인덱스 삭제 preview 후 개별 삭제를 검토하세요."
    elif any(issue in issue_set for issue in ["retrieval_excluded_indexed", "prompt_excluded_prompt_risk"]):
        recommended_action = "공식 기억은 유지하고 검색/프롬프트 제외 상태를 확인한 뒤 인덱스 삭제를 검토하세요."
    elif "body_hash_mismatch" in issue_set:
        recommended_action = "공식 기억 원본과 인덱스 복사본이 다릅니다. 이번 단계에서는 재색인하지 말고 인덱스 삭제 preview를 확인하세요."
    else:
        recommended_action = "후속 정리 전까지 격리 상태로 두고 개별 record 삭제 preview를 확인하세요."

    label = string_or_empty(canonical.get("title") if canonical else "") or entry_id or candidate_id or record_id
    preview_payload = build_audit_preview_payload(record.get("document"), metadata)

    return {
        "candidate_key": f"{collection}::{record_id or entry_id or candidate_id}::{','.join(issue_set)}",
        "label": label,
        "issue_types": issue_set,
        "risk": risk,
        "approved_memory_exists": canonical is not None,
        "memory_id": string_or_empty(canonical.get("memory_id") if canonical else ""),
        "entry_id": entry_id,
        "candidate_id": candidate_id,
        "lifecycle_state": string_or_empty(canonical.get("lifecycle_state") if canonical else ""),
        "status": string_or_empty(canonical.get("status") if canonical else ""),
        "retrieval_eligible": canonical.get("retrieval_eligible") if canonical else None,
        "prompt_eligible": canonical.get("prompt_eligible") if canonical else None,
        "body_hash_status": body_hash_status,
        "body_hash": record_body_hash(metadata),
        "index_collection": collection,
        "record_id": record_id,
        "preview": preview_payload["preview_text"],
        "preview_text": preview_payload["preview_text"],
        "preview_chars": preview_payload["preview_chars"],
        "preview_truncated": preview_payload["preview_truncated"],
        "body_chars": preview_payload["body_chars"],
        "document_chars": preview_payload["document_chars"],
        "body_available": preview_payload["body_available"],
        "preview_source": preview_payload["preview_source"],
        "recommended_action": recommended_action,
        "safety": "검색용 복사본/index record만 대상으로 합니다. 공식 기억과 Actor/Item/Journal/Page 원본은 삭제하지 않습니다.",
        "usage_toggle_allowed": canonical is not None,
        "purge_allowed": bool(record_id),
    }


def classify_contamination_record(
    *,
    collection: str,
    approved_collection_name: str,
    record: dict[str, Any],
    canonical_by_entry_id: dict[str, dict[str, Any]],
    canonical_by_body_hash: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    metadata = normalize_approved_metadata(record.get("metadata"))
    document = record.get("document")
    record_id = string_or_empty(record.get("id"))
    entry_id = record_entry_id(metadata)
    body_hash = record_body_hash(metadata)
    canonical = canonical_by_entry_id.get(entry_id) if entry_id else None
    if canonical is None and body_hash:
        canonical = canonical_by_body_hash.get(body_hash)

    issues: list[str] = []
    if collection != approved_collection_name or is_legacy_or_old_collection(collection):
        issues.append("legacy_collection_record")

    if canonical is None and collection == approved_collection_name:
        issues.append("orphaned_index_record")

    if canonical is not None:
        lifecycle = string_or_empty(canonical.get("lifecycle_state") or canonical.get("status")) or "active"
        if lifecycle not in {"active"}:
            issues.append("non_active_lifecycle_indexed")

        if canonical.get("retrieval_eligible") is False:
            issues.append("retrieval_excluded_indexed")

        metadata_prompt = metadata_bool(metadata, "prompt_eligible")
        if canonical.get("prompt_eligible") is False and (metadata_prompt is True or collection == approved_collection_name):
            issues.append("prompt_excluded_prompt_risk")

        canonical_hash = string_or_empty(canonical.get("body_hash"))
        if canonical_hash and body_hash and canonical_hash != body_hash:
            issues.append("body_hash_mismatch")

        if int_from_payload(canonical.get("entity_ref_count"), 0) <= 0 and canonical.get("prompt_eligible") is True:
            issues.append("missing_entity_ref_prompt_risk")

    if record_has_test_signal(record_id, metadata, document):
        issues.append("dummy_or_test_index_record")

    if document_looks_like_raw_dump(document, metadata):
        issues.append("raw_document_dump_suspected")

    if not entry_id and record_candidate_id(metadata):
        issues.append("unapproved_candidate_indexed")

    return make_audit_candidate(collection=collection, record=record, canonical=canonical, issues=issues)


def local_memory_contamination_candidates(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for memory in memories:
        if string_or_empty(memory.get("index_state")) not in {"indexed", "needs_index"}:
            continue
        record_id = string_or_empty(memory.get("index_record_id"))
        synthetic_record = {
            "id": record_id,
            "metadata": {
                "entry_id": memory.get("entry_id"),
                "body_hash": memory.get("body_hash"),
            },
            "document": "",
        }
        issues: list[str] = []
        lifecycle = string_or_empty(memory.get("lifecycle_state") or memory.get("status")) or "active"
        if lifecycle not in {"active"}:
            issues.append("non_active_lifecycle_indexed")
        if memory.get("retrieval_eligible") is False:
            issues.append("retrieval_excluded_indexed")
        if memory.get("prompt_eligible") is False:
            issues.append("prompt_excluded_prompt_risk")
        if string_or_empty(memory.get("index_state")) == "needs_index":
            issues.append("body_hash_mismatch")
        candidate = make_audit_candidate(
            collection=string_or_empty(memory.get("index_collection")) or "local_derived_cache",
            record=synthetic_record,
            canonical=memory,
            issues=issues,
        )
        if candidate:
            candidate["purge_allowed"] = False
            candidate["safety"] = "이 항목은 로컬 파생 캐시 상태입니다. D4 backend purge는 Chroma 검색 인덱스 record만 삭제합니다."
            candidates.append(candidate)
    return candidates


def build_contamination_summary(
    *,
    candidates: list[dict[str, Any]],
    memories: list[dict[str, Any]],
    approved_collection: dict[str, Any],
    legacy_collections: list[dict[str, Any]],
    legacy_long_term_memory_count: int,
) -> dict[str, Any]:
    return {
        "contamination_suspected": len(candidates),
        "indexed": int_from_payload(approved_collection.get("count"), 0),
        "index_stale": sum(1 for candidate in candidates if "body_hash_mismatch" in candidate.get("issue_types", [])),
        "orphaned": sum(1 for candidate in candidates if "orphaned_index_record" in candidate.get("issue_types", [])),
        "dummy": sum(1 for candidate in candidates if "dummy_or_test_index_record" in candidate.get("issue_types", [])),
        "legacy": legacy_long_term_memory_count + sum(int_from_payload(item.get("count"), 0) for item in legacy_collections),
        "retrieval_excluded": sum(1 for memory in memories if memory.get("retrieval_eligible") is False),
        "prompt_excluded": sum(1 for memory in memories if memory.get("prompt_eligible") is False),
    }


def build_approved_retrieval_audit(data: dict[str, Any], approved_collection_name: str) -> dict[str, Any]:
    memories = normalize_audit_memories(data.get("memories"))
    canonical_by_entry_id, canonical_by_body_hash = memory_lookup_maps(memories)
    legacy_long_term_memory_count = int_from_payload(data.get("legacy_long_term_memory_count"), 0)
    collection_names = list_chroma_collection_names_without_create()

    approved_collection = collection_records_without_create(approved_collection_name, limit=500)
    candidates = local_memory_contamination_candidates(memories)
    if approved_collection.get("ok") is False:
        return {
            "ok": False,
            "error": approved_collection.get("error") or "approved collection audit failed",
            "status": "error",
            "collection": approved_collection_name,
            "read_only": True,
            "writes_performed": False,
        }

    for record in approved_collection.get("records", []):
        candidate = classify_contamination_record(
            collection=approved_collection_name,
            approved_collection_name=approved_collection_name,
            record=record,
            canonical_by_entry_id=canonical_by_entry_id,
            canonical_by_body_hash=canonical_by_body_hash,
        )
        if candidate:
            candidates.append(candidate)

    legacy_collections: list[dict[str, Any]] = []
    for collection_name in collection_names:
        if not is_legacy_or_old_collection(collection_name):
            continue
        snapshot = collection_records_without_create(collection_name, limit=100)
        legacy_collections.append(
            {
                "name": collection_name,
                "exists": snapshot.get("exists") is True,
                "count": int_from_payload(snapshot.get("count"), 0),
                "error": string_or_empty(snapshot.get("error")),
            }
        )
        for record in snapshot.get("records", []):
            candidate = classify_contamination_record(
                collection=collection_name,
                approved_collection_name=approved_collection_name,
                record=record,
                canonical_by_entry_id=canonical_by_entry_id,
                canonical_by_body_hash=canonical_by_body_hash,
            )
            if candidate:
                candidates.append(candidate)

    seen_keys: set[str] = set()
    unique_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        key = string_or_empty(candidate.get("candidate_key"))
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        unique_candidates.append(candidate)

    summary = build_contamination_summary(
        candidates=unique_candidates,
        memories=memories,
        approved_collection=approved_collection,
        legacy_collections=legacy_collections,
        legacy_long_term_memory_count=legacy_long_term_memory_count,
    )
    no_index = approved_collection.get("exists") is not True or int_from_payload(approved_collection.get("count"), 0) <= 0

    return {
        "ok": True,
        "status": "ok",
        "message": "검색 인덱스가 아직 없습니다." if no_index else "오염 후보 점검이 완료되었습니다.",
        "collection": approved_collection_name,
        "read_only": True,
        "writes_performed": False,
        "voyage_called": False,
        "chroma_written": False,
        "collections_created": False,
        "no_index": no_index,
        "summary": summary,
        "candidates": unique_candidates,
        "legacy": {
            "long_term_memory_count": legacy_long_term_memory_count,
            "collections": legacy_collections,
            "raw_dump_suspected": sum(1 for candidate in unique_candidates if "raw_document_dump_suspected" in candidate.get("issue_types", [])),
            "guidance": "legacy longTermMemory 원본과 legacy collection 전체 삭제는 이번 D4 slice에서 수행하지 않습니다. 필요한 경우 개별 index record 삭제 preview만 사용하세요.",
        },
        "safety": {
            "source_documents_deleted": False,
            "approved_memory_deleted": False,
            "index_records_only": True,
            "reindex_performed": False,
            "legacy_endpoints_reenabled": False,
        },
    }


def normalize_purge_targets(data: dict[str, Any], approved_collection_name: str) -> list[dict[str, str]]:
    raw_records = data.get("records") or data.get("targets") or []
    if not isinstance(raw_records, list):
        return []
    targets: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in raw_records:
        if isinstance(raw, str):
            collection = approved_collection_name
            record_id = raw.strip()
        elif isinstance(raw, dict):
            collection = (
                string_or_empty(raw.get("collection"))
                or string_or_empty(raw.get("index_collection"))
                or approved_collection_name
            )
            record_id = string_or_empty(raw.get("record_id")) or string_or_empty(raw.get("id"))
        else:
            continue
        if not collection or not record_id or record_id in {"*", "all", "__all__"}:
            continue
        key = f"{collection}::{record_id}"
        if key in seen:
            continue
        seen.add(key)
        targets.append({"collection": collection, "record_id": record_id})
    return targets[:100]


def preview_purge_targets(targets: list[dict[str, str]], approved_collection_name: str) -> dict[str, Any]:
    collection_names = list_chroma_collection_names_without_create()
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for target in targets:
        collection = target["collection"]
        record_id = target["record_id"]
        # Never let one world purge records out of another world's active
        # scoped collection (the current world's own approved collection is fine).
        # Checked before existence so another world's collection is never probed.
        if is_active_world_scoped_collection(collection) and collection != approved_collection_name:
            skipped.append({"collection": collection, "record_id": record_id, "reason": "cross_world_protected"})
            continue
        if collection not in collection_names:
            skipped.append({"collection": collection, "record_id": record_id, "reason": "collection_missing"})
            continue
        snapshot = collection_records_without_create(collection, ids=[record_id])
        if snapshot.get("ok") is not True:
            skipped.append({"collection": collection, "record_id": record_id, "reason": snapshot.get("error") or "collection_read_failed"})
            continue
        found = next((record for record in snapshot.get("records", []) if record.get("id") == record_id), None)
        if not found:
            skipped.append({"collection": collection, "record_id": record_id, "reason": "record_missing"})
            continue
        metadata = normalize_approved_metadata(found.get("metadata"))
        records.append(
            {
                "collection": collection,
                "record_id": record_id,
                "entry_id": record_entry_id(metadata),
                "candidate_id": record_candidate_id(metadata),
                "body_hash": record_body_hash(metadata),
                "preview": safe_record_preview(found.get("document")),
            }
        )
    return {
        "ok": True,
        "status": "ok",
        "dry_run": True,
        "records": records,
        "target_count": len(records),
        "skipped": skipped,
        "skipped_count": len(skipped),
        "collection_delete_allowed": False,
        "reindex_performed": False,
        "message": "검색 인덱스 삭제 미리보기입니다. 아직 삭제하지 않았습니다.",
    }


LEGACY_PURGE_CATEGORY_IDS = {
    "legacy_index_records",
    "orphaned_records",
    "dummy_records",
    "body_hash_missing_records",
    "metadata_preview_only_records",
    "raw_document_dump_records",
    "legacy_collections",
}


def normalize_legacy_purge_categories(data: dict[str, Any]) -> set[str]:
    raw = data.get("selected_categories") or data.get("categories") or []
    if not isinstance(raw, list):
        return set()
    return {string_or_empty(item) for item in raw if string_or_empty(item) in LEGACY_PURGE_CATEGORY_IDS}


def is_gm_confirmed_request(data: dict[str, Any]) -> bool:
    return bool_from_payload(data.get("gm_confirmed")) is True


def canonical_for_index_record(
    record: dict[str, Any],
    canonical_by_entry_id: dict[str, dict[str, Any]],
    canonical_by_body_hash: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    metadata = normalize_approved_metadata(record.get("metadata"))
    entry_id = record_entry_id(metadata)
    body_hash = record_body_hash(metadata)
    canonical = canonical_by_entry_id.get(entry_id) if entry_id else None
    if canonical is None and body_hash:
        canonical = canonical_by_body_hash.get(body_hash)
    return canonical


def legacy_purge_record_categories(collection: str, record: dict[str, Any], canonical: dict[str, Any] | None, approved_collection_name: str) -> list[str]:
    if canonical is not None:
        return []

    metadata = normalize_approved_metadata(record.get("metadata"))
    document = record.get("document")
    categories: set[str] = set()

    if collection != approved_collection_name or is_legacy_or_old_collection(collection):
        categories.add("legacy_index_records")

    if collection == approved_collection_name or record_entry_id(metadata):
        categories.add("orphaned_records")

    if record_has_test_signal(string_or_empty(record.get("id")), metadata, document):
        categories.add("dummy_records")

    if not record_body_hash(metadata):
        categories.add("body_hash_missing_records")

    if document_looks_like_raw_dump(document, metadata):
        categories.add("raw_document_dump_records")

    preview_payload = build_audit_preview_payload(document, metadata)
    if preview_payload.get("body_available") is False and preview_payload.get("preview_text"):
        categories.add("metadata_preview_only_records")

    return sorted(categories)


def record_has_review_queue_or_session_review_signal(metadata: dict[str, Any], document: Any) -> bool:
    text = " ".join(
        [
            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            str(document or "")[:1200],
        ]
    ).lower()
    protected_tokens = [
        "review_queue",
        "reviewqueue",
        "source_candidate_page_uuid",
        "candidate_page",
        "sessionreview",
        "session_review",
        "sessionreviewrun",
    ]
    return any(token in text for token in protected_tokens)


def make_legacy_purge_record_candidate(
    *,
    collection: str,
    record: dict[str, Any],
    categories: list[str],
) -> dict[str, Any]:
    metadata = normalize_approved_metadata(record.get("metadata"))
    preview_payload = build_audit_preview_payload(record.get("document"), metadata)
    return {
        "collection": collection,
        "record_id": string_or_empty(record.get("id")),
        "entry_id": record_entry_id(metadata),
        "candidate_id": record_candidate_id(metadata),
        "body_hash": record_body_hash(metadata),
        "categories": categories,
        "preview": preview_payload.get("preview_text") or safe_record_preview(record.get("document")),
        "preview_source": preview_payload.get("preview_source") or "",
        "protected": False,
    }


def make_legacy_purge_protected_record(
    *,
    collection: str,
    record: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    metadata = normalize_approved_metadata(record.get("metadata"))
    return {
        "collection": collection,
        "record_id": string_or_empty(record.get("id")),
        "entry_id": record_entry_id(metadata),
        "candidate_id": record_candidate_id(metadata),
        "body_hash": record_body_hash(metadata),
        "reason": reason,
        "protected": True,
    }


def summarize_legacy_purge_preview(records: list[dict[str, Any]], collections: list[dict[str, Any]], protected: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {category: 0 for category in LEGACY_PURGE_CATEGORY_IDS}
    for record in records:
        for category in record.get("categories", []):
            if category in summary:
                summary[category] += 1

    summary["legacy_collections"] = len(collections)
    summary["legacy_collection_records"] = sum(int_from_payload(item.get("count"), 0) for item in collections)
    summary["protected_records"] = len(protected)
    summary["skipped_records"] = len(skipped)
    summary["record_candidates"] = len(records)
    return summary


def build_legacy_sync_purge_preview(data: dict[str, Any], approved_collection_name: str) -> dict[str, Any]:
    memories = normalize_audit_memories(data.get("memories"))
    canonical_by_entry_id, canonical_by_body_hash = memory_lookup_maps(memories)
    collection_names = list_chroma_collection_names_without_create()
    records: list[dict[str, Any]] = []
    protected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    legacy_collections: list[dict[str, Any]] = []

    # Scan THIS world's approved collection (record-level cleanup) + shared legacy
    # collections only. Other worlds' active scoped collections are excluded by
    # is_legacy_or_old_collection (which protects every active scoped name).
    collections_to_scan = [
        name for name in collection_names
        if name == approved_collection_name or is_legacy_or_old_collection(name)
    ]

    for collection_name in collections_to_scan:
        snapshot = collection_records_without_create(collection_name, limit=2000)
        if snapshot.get("ok") is not True:
            skipped.append(
                {
                    "collection": collection_name,
                    "record_id": "",
                    "reason": snapshot.get("error") or "collection_read_failed",
                }
            )
            continue

        if is_legacy_or_old_collection(collection_name):
            legacy_collections.append(
                {
                    "name": collection_name,
                    "exists": snapshot.get("exists") is True,
                    "count": int_from_payload(snapshot.get("count"), 0),
                    "scanned_count": len(snapshot.get("records", [])),
                    "fully_scanned": len(snapshot.get("records", [])) >= int_from_payload(snapshot.get("count"), 0),
                    "delete_allowed": collection_name != approved_collection_name,
                    "protected_record_count": 0,
                }
            )

        for record in snapshot.get("records", []):
            canonical = canonical_for_index_record(record, canonical_by_entry_id, canonical_by_body_hash)
            if canonical is not None:
                protected_item = make_legacy_purge_protected_record(
                    collection=collection_name,
                    record=record,
                    reason="approved_memory_exists",
                )
                protected.append(protected_item)
                for collection_info in legacy_collections:
                    if collection_info["name"] == collection_name:
                        collection_info["protected_record_count"] += 1
                continue

            metadata = normalize_approved_metadata(record.get("metadata"))
            if (
                record_has_review_queue_or_session_review_signal(metadata, record.get("document"))
                and not record_has_test_signal(string_or_empty(record.get("id")), metadata, record.get("document"))
            ):
                protected.append(
                    make_legacy_purge_protected_record(
                        collection=collection_name,
                        record=record,
                        reason="review_queue_or_session_review_protected",
                    )
                )
                for collection_info in legacy_collections:
                    if collection_info["name"] == collection_name:
                        collection_info["protected_record_count"] += 1
                continue

            categories = legacy_purge_record_categories(collection_name, record, canonical, approved_collection_name)
            if not categories:
                continue
            candidate = make_legacy_purge_record_candidate(
                collection=collection_name,
                record=record,
                categories=categories,
            )
            if candidate["record_id"]:
                records.append(candidate)
            else:
                skipped.append(
                    {
                        "collection": collection_name,
                        "record_id": "",
                        "reason": "record_id_missing",
                    }
                )

    summary = summarize_legacy_purge_preview(records, legacy_collections, protected, skipped)
    return {
        "ok": True,
        "status": "ok",
        "dry_run": True,
        "read_only": True,
        "writes_performed": False,
        "voyage_called": False,
        "chroma_written": False,
        "collections_created": False,
        "collection_delete_allowed": True,
        "approved_collection_full_delete_allowed": False,
        "approved_collection_name": approved_collection_name,
        "records": records,
        "collections": legacy_collections,
        "protected": protected,
        "skipped": skipped,
        "summary": summary,
        "safety": {
            "source_documents_deleted": False,
            "approved_memory_deleted": False,
            "approved_linked_records_skipped": True,
            "review_queue_rewritten": False,
            "session_review_deleted": False,
            "reindex_performed": False,
            "voyage_called": False,
            "legacy_endpoints_reenabled": False,
        },
        "message": "Legacy 동기화 데이터 삭제 dry-run inventory입니다. 아직 삭제하지 않았습니다.",
    }


def filter_legacy_purge_records(records: list[dict[str, Any]], selected_categories: set[str], deleted_collections: set[str]) -> list[dict[str, Any]]:
    selected_records: list[dict[str, Any]] = []
    for record in records:
        collection = string_or_empty(record.get("collection"))
        record_id = string_or_empty(record.get("record_id"))
        categories = {string_or_empty(category) for category in record.get("categories", [])}
        if not collection or not record_id:
            continue
        if collection in deleted_collections:
            continue
        if categories.intersection(selected_categories):
            selected_records.append(record)
    return selected_records


def delete_legacy_collections(collections: list[dict[str, Any]], selected_categories: set[str], approved_collection_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    deleted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    if "legacy_collections" not in selected_categories:
        return deleted, skipped, failed

    names = set(list_chroma_collection_names_without_create())
    for collection_info in collections:
        name = string_or_empty(collection_info.get("name"))
        if not name:
            skipped.append({"collection": name, "reason": "collection_name_missing"})
            continue
        if name == approved_collection_name or is_active_world_scoped_collection(name):
            skipped.append({"collection": name, "reason": "approved_collection_full_delete_forbidden"})
            continue
        if not is_legacy_or_old_collection(name):
            skipped.append({"collection": name, "reason": "not_legacy_collection"})
            continue
        if int_from_payload(collection_info.get("protected_record_count"), 0) > 0:
            skipped.append({"collection": name, "reason": "approved_linked_records_present"})
            continue
        if collection_info.get("fully_scanned") is not True:
            skipped.append({"collection": name, "reason": "collection_not_fully_scanned"})
            continue
        if name not in names:
            skipped.append({"collection": name, "reason": "collection_missing"})
            continue
        try:
            client = get_chroma_client()
            client.delete_collection(name)
            deleted.append(
                {
                    "collection": name,
                    "record_count": int_from_payload(collection_info.get("count"), 0),
                    "reason": "legacy_collection_deleted",
                }
            )
        except Exception as exc:
            failed.append({"collection": name, "reason": str(exc)})

    return deleted, skipped, failed


def delete_legacy_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    deleted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    names = set(list_chroma_collection_names_without_create())
    by_collection: dict[str, list[str]] = {}

    for record in records:
        collection = string_or_empty(record.get("collection"))
        record_id = string_or_empty(record.get("record_id"))
        if not collection or not record_id:
            skipped.append({"collection": collection, "record_id": record_id, "reason": "missing_collection_or_record_id"})
            continue
        if collection not in names:
            skipped.append({"collection": collection, "record_id": record_id, "reason": "collection_missing"})
            continue
        by_collection.setdefault(collection, []).append(record_id)

    for collection_name, record_ids in by_collection.items():
        try:
            client = get_chroma_client()
            collection = client.get_collection(collection_name)
            collection.delete(ids=record_ids)
            for record_id in record_ids:
                deleted.append({"collection": collection_name, "record_id": record_id})
        except Exception as exc:
            for record_id in record_ids:
                failed.append({"collection": collection_name, "record_id": record_id, "reason": str(exc)})

    return deleted, skipped, failed


def purge_legacy_sync_data(data: dict[str, Any], approved_collection_name: str) -> dict[str, Any]:
    selected_categories = normalize_legacy_purge_categories(data)
    preview = build_legacy_sync_purge_preview(data, approved_collection_name)
    collection_deleted, collection_skipped, collection_failed = delete_legacy_collections(
        preview.get("collections", []),
        selected_categories,
        approved_collection_name,
    )
    deleted_collection_names = {item["collection"] for item in collection_deleted}
    record_targets = filter_legacy_purge_records(preview.get("records", []), selected_categories, deleted_collection_names)
    record_deleted, record_skipped, record_failed = delete_legacy_records(record_targets)
    skipped = list(preview.get("protected", [])) + list(preview.get("skipped", [])) + collection_skipped + record_skipped
    failed = collection_failed + record_failed
    collection_record_deleted_count = sum(int_from_payload(item.get("record_count"), 0) for item in collection_deleted)
    deleted_count = collection_record_deleted_count + len(record_deleted)

    return {
        "ok": len(failed) == 0,
        "status": "ok" if len(failed) == 0 else "partial_error",
        "deleted_collections": collection_deleted,
        "deleted_records": record_deleted,
        "deleted_collection_count": len(collection_deleted),
        "deleted_record_count": len(record_deleted),
        "deleted_count": deleted_count,
        "skipped": skipped,
        "skipped_count": len(skipped),
        "failed": failed,
        "failed_count": len(failed),
        "selected_categories": sorted(selected_categories),
        "approved_memory_deleted": False,
        "source_documents_deleted": False,
        "approved_linked_records_skipped": True,
        "collection_delete_allowed": True,
        "approved_collection_full_delete_allowed": False,
        "reindex_performed": False,
        "voyage_called": False,
        "chroma_written": deleted_count > 0,
        "legacy_endpoints_reenabled": False,
        "message": f"Legacy 동기화 데이터 {deleted_count}개를 삭제했습니다. skipped {len(skipped)}개, failed {len(failed)}개입니다.",
    }


def _register_cuda_dll_dirs() -> None:
    """Windows: make pip-installed CUDA libs (nvidia-cublas-cu12 / nvidia-cudnn-cu12 /
    nvidia-cuda-runtime-cu12) loadable by CTranslate2 (faster-whisper's GPU backend).
    Those wheels drop their DLLs under site-packages/nvidia/<lib>/bin, a directory
    Windows does NOT add to the DLL search path automatically (unlike Linux RPATH), so
    on the cuda device the transcribe step fails with "cublas64_12.dll is not found or
    cannot be loaded". CTranslate2's C++ loads these via a plain LoadLibrary that
    searches PATH — NOT the os.add_dll_directory user-dir list — so PATH is the search
    path that actually fixes it (verified: ctypes-only add_dll_directory loaded the DLL
    but CTranslate2 compute still failed; prepending PATH made the GPU encode run).
    add_dll_directory is kept too for any ctypes-based loads. No-op on non-Windows or
    when the packages aren't installed (e.g. STT_DEVICE=cpu) — never hurts CPU."""
    if sys.platform != "win32":
        return
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia")
        locations = list(getattr(spec, "submodule_search_locations", None) or [])
    except Exception:
        locations = []
    if not locations:
        return
    base = locations[0]
    for sub in ("cuda_runtime", "cublas", "cudnn", "cuda_nvrtc"):
        dll_dir = os.path.join(base, sub, "bin")
        if not os.path.isdir(dll_dir):
            continue
        current = os.environ.get("PATH", "")
        if dll_dir not in current.split(os.pathsep):
            os.environ["PATH"] = dll_dir + os.pathsep + current
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(dll_dir)
            except Exception:
                pass


def get_whisper_model() -> Any:
    global whisper_model

    if not STT_ENABLED:
        raise RuntimeError("STT is disabled. Set STT_ENABLED=true to enable it.")

    with whisper_lock:
        if whisper_model:
            return whisper_model

        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("faster-whisper is not installed. Install requirements-stt.txt.") from exc

        _register_cuda_dll_dirs()
        whisper_model = WhisperModel(_resolve_local_stt_ref(), device=STT_DEVICE, compute_type=STT_COMPUTE_TYPE)
        return whisper_model


@app.get("/health")
def health() -> Any:
    vector_db_ready = False
    vector_db_error = None

    try:
        get_chroma_collections()
        vector_db_ready = True
    except Exception as exc:
        vector_db_error = str(exc)

    return jsonify(
        {
            "status": "ok",
            "service": APP_NAME,
            "auth_required": bool(resolve_companion_auth_key()),
            "legacy_endpoints": legacy_endpoint_status(),
            "vector_db_ready": vector_db_ready,
            "vector_db_error": vector_db_error,
            "graph": {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
            },
            "stt_enabled": STT_ENABLED,
            "embedding_backend": resolve_embedding_backend(),
            "local_embedding_enabled": LOCAL_EMBEDDING_ENABLED,
            "vertex_proxy_configured": bool(resolve_vertex_project_id()),
            "vertex_project_source": "env" if resolve_vertex_project_id() else None,
            "vertex_quota_project_configured": bool(resolve_vertex_quota_project_id()),
        }
    )


@app.route("/approved-retrieval/index/status", methods=["GET", "OPTIONS"])
@require_auth
def approved_retrieval_index_status() -> Any:
    world_id, scope_error = require_world_scope()
    if scope_error:
        return scope_error
    status = get_approved_retrieval_status(scoped_collection_name(APPROVED_RETRIEVAL_COLLECTION_NAME, world_id))
    return jsonify({"status": "ok", **status})


@app.route("/approved-retrieval/index/audit", methods=["GET", "POST", "OPTIONS"])
@require_auth
def approved_retrieval_index_audit() -> Any:
    world_id, scope_error = require_world_scope()
    if scope_error:
        return scope_error
    data = json_payload() if request.method == "POST" else {}
    try:
        return jsonify(build_approved_retrieval_audit(data, scoped_collection_name(APPROVED_RETRIEVAL_COLLECTION_NAME, world_id)))
    except Exception as exc:
        return jsonify(
            {
                "ok": False,
                "status": "error",
                "error": str(exc),
                "read_only": True,
                "writes_performed": False,
                "voyage_called": False,
                "chroma_written": False,
                "collections_created": False,
            }
        ), 503


@app.route("/approved-retrieval/index/purge-preview", methods=["POST", "OPTIONS"])
@require_auth
def approved_retrieval_index_purge_preview() -> Any:
    world_id, scope_error = require_world_scope()
    if scope_error:
        return scope_error
    approved_name = scoped_collection_name(APPROVED_RETRIEVAL_COLLECTION_NAME, world_id)
    data = json_payload()
    targets = normalize_purge_targets(data, approved_name)
    if not targets:
        return jsonify(
            {
                "ok": True,
                "status": "ok",
                "dry_run": True,
                "records": [],
                "target_count": 0,
                "skipped": [],
                "skipped_count": 0,
                "collection_delete_allowed": False,
                "reindex_performed": False,
                "message": "삭제 미리보기 대상이 없습니다.",
            }
        )
    return jsonify(preview_purge_targets(targets, approved_name))


@app.route("/approved-retrieval/index/purge-records", methods=["POST", "OPTIONS"])
@require_auth
def approved_retrieval_index_purge_records() -> Any:
    world_id, scope_error = require_world_scope()
    if scope_error:
        return scope_error
    approved_name = scoped_collection_name(APPROVED_RETRIEVAL_COLLECTION_NAME, world_id)
    data = json_payload()
    confirm_text = string_or_empty(data.get("confirm_text")) or string_or_empty(data.get("confirmText"))
    if confirm_text != APPROVED_RETRIEVAL_PURGE_CONFIRM_TEXT:
        return jsonify(
            {
                "ok": False,
                "status": "confirmation_required",
                "error": "strong_confirmation_required",
                "expected_confirmation": APPROVED_RETRIEVAL_PURGE_CONFIRM_TEXT,
                "deleted_count": 0,
                "skipped_count": 0,
                "collection_delete_allowed": False,
                "reindex_performed": False,
            }
        ), 400

    targets = normalize_purge_targets(data, approved_name)
    preview = preview_purge_targets(targets, approved_name)
    deletable = preview.get("records", [])
    skipped = list(preview.get("skipped", []))
    deleted: list[dict[str, str]] = []

    by_collection: dict[str, list[str]] = {}
    for record in deletable:
        by_collection.setdefault(record["collection"], []).append(record["record_id"])

    for collection_name, record_ids in by_collection.items():
        try:
            client = get_chroma_client()
            collection = client.get_collection(collection_name)
            collection.delete(ids=record_ids)
            for record_id in record_ids:
                deleted.append({"collection": collection_name, "record_id": record_id})
        except Exception as exc:
            for record_id in record_ids:
                skipped.append({"collection": collection_name, "record_id": record_id, "reason": str(exc)})

    return jsonify(
        {
            "ok": True,
            "status": "ok",
            "deleted": deleted,
            "deleted_count": len(deleted),
            "skipped": skipped,
            "skipped_count": len(skipped),
            "collection_delete_allowed": False,
            "approved_memory_deleted": False,
            "source_documents_deleted": False,
            "reindex_performed": False,
            "voyage_called": False,
            "chroma_written": bool(deleted),
            "message": f"검색 인덱스 record {len(deleted)}개를 삭제했습니다.",
        }
    )


@app.route("/approved-retrieval/index/legacy-purge-preview", methods=["POST", "OPTIONS"])
@require_auth
def approved_retrieval_index_legacy_purge_preview() -> Any:
    world_id, scope_error = require_world_scope()
    if scope_error:
        return scope_error
    data = json_payload()
    if not is_gm_confirmed_request(data):
        return jsonify(
            {
                "ok": False,
                "status": "gm_required",
                "error": "gm_confirmed_required",
                "read_only": True,
                "writes_performed": False,
                "voyage_called": False,
                "chroma_written": False,
                "collections_created": False,
            }
        ), 403
    try:
        return jsonify(build_legacy_sync_purge_preview(data, scoped_collection_name(APPROVED_RETRIEVAL_COLLECTION_NAME, world_id)))
    except Exception as exc:
        return jsonify(
            {
                "ok": False,
                "status": "error",
                "error": str(exc),
                "read_only": True,
                "writes_performed": False,
                "voyage_called": False,
                "chroma_written": False,
                "collections_created": False,
            }
        ), 503


@app.route("/approved-retrieval/index/legacy-purge", methods=["POST", "OPTIONS"])
@require_auth
def approved_retrieval_index_legacy_purge() -> Any:
    world_id, scope_error = require_world_scope()
    if scope_error:
        return scope_error
    data = json_payload()
    if not is_gm_confirmed_request(data):
        return jsonify(
            {
                "ok": False,
                "status": "gm_required",
                "error": "gm_confirmed_required",
                "deleted_count": 0,
                "skipped_count": 0,
                "failed_count": 0,
                "voyage_called": False,
                "chroma_written": False,
                "collections_created": False,
            }
        ), 403

    confirm_text = string_or_empty(data.get("confirm_text")) or string_or_empty(data.get("confirmText"))
    if confirm_text != APPROVED_RETRIEVAL_LEGACY_PURGE_CONFIRM_TEXT:
        return jsonify(
            {
                "ok": False,
                "status": "confirmation_required",
                "error": "strong_confirmation_required",
                "expected_confirmation": APPROVED_RETRIEVAL_LEGACY_PURGE_CONFIRM_TEXT,
                "deleted_count": 0,
                "skipped_count": 0,
                "failed_count": 0,
                "approved_collection_full_delete_allowed": False,
                "reindex_performed": False,
                "voyage_called": False,
            }
        ), 400

    selected_categories = normalize_legacy_purge_categories(data)
    if not selected_categories:
        return jsonify(
            {
                "ok": False,
                "status": "categories_required",
                "error": "selected_categories_required",
                "deleted_count": 0,
                "skipped_count": 0,
                "failed_count": 0,
                "approved_collection_full_delete_allowed": False,
                "reindex_performed": False,
                "voyage_called": False,
            }
        ), 400

    try:
        return jsonify(purge_legacy_sync_data(data, scoped_collection_name(APPROVED_RETRIEVAL_COLLECTION_NAME, world_id)))
    except Exception as exc:
        return jsonify(
            {
                "ok": False,
                "status": "error",
                "error": str(exc),
                "deleted_count": 0,
                "skipped_count": 0,
                "failed_count": 1,
                "failed": [{"reason": str(exc)}],
                "approved_memory_deleted": False,
                "source_documents_deleted": False,
                "reindex_performed": False,
                "voyage_called": False,
                "chroma_written": False,
                "collections_created": False,
            }
        ), 503


def build_world_isolation_migration_plan(world_id: str) -> list[dict[str, Any]]:
    """Plan to move pre-isolation unsuffixed collections into <base>__<world_id>.

    For each of the three active bases: rename the unsuffixed collection to the
    current world's namespace. If an empty target already exists (auto-created by
    a request that hit the server after the F5 reload but before migration), that
    empty target is replaced; a target with data is left untouched (never clobber
    real data — likely already migrated, or another world's by misconfig).
    """

    existing = set(list_chroma_collection_names_without_create())
    plan: list[dict[str, Any]] = []
    for base in (
        APPROVED_RETRIEVAL_COLLECTION_NAME,
        RED_WORLD_KNOWLEDGE_COLLECTION_NAME,
        GM_BOT_KNOWLEDGE_COLLECTION_NAME,
    ):
        target = scoped_collection_name(base, world_id)
        entry: dict[str, Any] = {
            "base": base,
            "target": target,
            "source_exists": base in existing,
            "target_exists": target in existing,
            "source_count": None,
            "target_count": None,
            "action": "skip",
            "reason": "",
        }
        if not entry["source_exists"]:
            entry["reason"] = "source_missing"
        else:
            entry["source_count"] = get_chroma_collection_count_without_create(base)
            if not entry["target_exists"]:
                entry["action"] = "rename"
            else:
                target_count = get_chroma_collection_count_without_create(target) or 0
                entry["target_count"] = target_count
                if target_count == 0:
                    entry["action"] = "replace_empty_target"
                else:
                    entry["reason"] = "target_has_data"
        plan.append(entry)
    return plan


@app.route("/admin/world-isolation/migrate", methods=["POST", "OPTIONS"])
@require_auth
def world_isolation_migrate() -> Any:
    """One-time GM-run migration: rename the legacy unsuffixed active collections
    into the current world's per-world namespace. Index write -> HARD GATE:
    preview is read-only; run requires the strong confirm text.
    """

    world_id, scope_error = require_world_scope()
    if scope_error:
        return scope_error
    data = json_payload()
    mode = string_or_empty(data.get("mode")) or "preview"
    plan = build_world_isolation_migration_plan(world_id)
    actionable = [item for item in plan if item["action"] in {"rename", "replace_empty_target"}]

    if mode == "preview":
        return jsonify(
            {
                "ok": True,
                "status": "ok",
                "dry_run": True,
                "world_id": world_id,
                "plan": plan,
                "actionable_count": len(actionable),
                "expected_confirmation": WORLD_ISOLATION_MIGRATION_CONFIRM_TEXT,
                "chroma_written": False,
                "message": "월드 격리 이관 dry-run입니다. 아직 아무것도 바꾸지 않았습니다.",
            }
        )

    if mode != "run":
        return jsonify({"ok": False, "error": "mode must be 'preview' or 'run'."}), 400

    confirm_text = string_or_empty(data.get("confirm_text")) or string_or_empty(data.get("confirmText"))
    if confirm_text != WORLD_ISOLATION_MIGRATION_CONFIRM_TEXT:
        return jsonify(
            {
                "ok": False,
                "status": "confirmation_required",
                "error": "strong_confirmation_required",
                "expected_confirmation": WORLD_ISOLATION_MIGRATION_CONFIRM_TEXT,
                "renamed_count": 0,
                "chroma_written": False,
            }
        ), 400

    renamed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    client = get_chroma_client()
    for item in plan:
        if item["action"] not in {"rename", "replace_empty_target"}:
            skipped.append(item)
            continue
        try:
            if item["action"] == "replace_empty_target":
                client.delete_collection(item["target"])
            client.get_collection(item["base"]).modify(name=item["target"])
            renamed.append(item)
        except Exception as exc:
            failed.append({**item, "reason": str(exc)})

    # Drop cached per-world handles so later requests resolve the renamed collections.
    with chroma_lock:
        approved_retrieval_collections.clear()
        world_knowledge_collections.clear()
        gm_bot_knowledge_collections.clear()

    return jsonify(
        {
            "ok": len(failed) == 0,
            "status": "ok" if len(failed) == 0 else "partial_error",
            "world_id": world_id,
            "renamed": renamed,
            "renamed_count": len(renamed),
            "skipped": skipped,
            "skipped_count": len(skipped),
            "failed": failed,
            "failed_count": len(failed),
            "chroma_written": bool(renamed),
            "message": f"월드 격리 이관 완료: collection {len(renamed)}개를 '__{world_id}' 네임스페이스로 이동했습니다.",
        }
    )


@app.route("/approved-retrieval/index/build", methods=["POST", "OPTIONS"])
@require_auth
def approved_retrieval_index_build() -> Any:
    world_id, scope_error = require_world_scope()
    if scope_error:
        return scope_error
    collection_name = scoped_collection_name(APPROVED_RETRIEVAL_COLLECTION_NAME, world_id)
    data = json_payload()
    records = data.get("records", [])
    mode = string_or_empty(data.get("mode")) or "upsert"

    if mode not in {"replace", "upsert"}:
        return jsonify({"ok": False, "error": "mode must be 'replace' or 'upsert'."}), 400

    if not isinstance(records, list):
        return jsonify({"ok": False, "error": "records must be a list."}), 400

    validation_errors: list[dict[str, Any]] = []
    validated: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        validation = validate_approved_retrieval_record(record)
        if not validation["ok"]:
            validation_errors.append(
                {
                    "index": index,
                    "id": record.get("id") if isinstance(record, dict) else "",
                    "record_id": record.get("record_id") if isinstance(record, dict) else "",
                    "errors": validation["errors"],
                }
            )
            continue
        validated.append({"record": record, "validation": validation})

    if validation_errors:
        return jsonify(
            {
                "ok": False,
                "error": "Approved retrieval records failed validation.",
                "invalid_count": len(validation_errors),
                "errors": validation_errors,
                "indexed_count": 0,
                "collection": collection_name,
            }
        ), 400

    if not validated:
        return jsonify(
            {
                "ok": True,
                "status": "ok",
                "mode": mode,
                "indexed_count": 0,
                "submitted_count": 0,
                "collection": collection_name,
                "voyage_called": False,
                "chroma_written": False,
            }
        )

    if not embedding_backend_available():
        return jsonify(
            {
                "ok": False,
                "error": "Voyage API key is not provided by the module settings.",
                "indexed_count": 0,
                "submitted_count": len(validated),
                "collection": collection_name,
            }
        ), 503

    texts = [item["validation"]["text"] for item in validated]
    ids = [approved_retrieval_record_id(item["record"]) for item in validated]
    metadatas = [build_approved_retrieval_metadata(item["record"]) for item in validated]

    if len(set(ids)) != len(ids):
        return jsonify(
            {
                "ok": False,
                "error": "Approved retrieval record IDs must be unique.",
                "indexed_count": 0,
                "submitted_count": len(validated),
                "collection": collection_name,
            }
        ), 400

    try:
        embeddings = embed_texts(texts, input_type="document")
    except Exception as exc:
        return jsonify(
            {
                "ok": False,
                "error": str(exc),
                "indexed_count": 0,
                "submitted_count": len(validated),
                "collection": collection_name,
            }
        ), 503

    if len(embeddings) != len(validated):
        return jsonify(
            {
                "ok": False,
                "error": "Voyage embedding count did not match submitted records.",
                "indexed_count": 0,
                "submitted_count": len(validated),
                "collection": collection_name,
            }
        ), 503

    try:
        collection = get_approved_retrieval_collection(world_id)
        replaced_count = 0
        if mode == "replace":
            existing = collection.get()
            existing_ids = existing.get("ids", []) if isinstance(existing, dict) else []
            if existing_ids:
                collection.delete(ids=existing_ids)
                replaced_count = len(existing_ids)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts,
        )
        collection_count = int(collection.count())
    except Exception as exc:
        return jsonify(
            {
                "ok": False,
                "error": str(exc),
                "indexed_count": 0,
                "submitted_count": len(validated),
                "collection": collection_name,
            }
        ), 503

    return jsonify(
        {
            "ok": True,
            "status": "ok",
            "mode": mode,
            "indexed_count": len(validated),
            "submitted_count": len(validated),
            "collection": collection_name,
            "collection_count": collection_count,
            "replaced_count": replaced_count,
            "voyage_called": True,
            "chroma_written": True,
        }
    )


@app.route("/approved-retrieval/index/search", methods=["POST", "OPTIONS"])
@require_auth
def approved_retrieval_index_search() -> Any:
    world_id, scope_error = require_world_scope()
    if scope_error:
        return scope_error
    collection_name = scoped_collection_name(APPROVED_RETRIEVAL_COLLECTION_NAME, world_id)
    data = json_payload()
    query = string_or_empty(data.get("query"))
    audience = string_or_empty(data.get("audience")) or "npc"
    actor_id = string_or_empty(data.get("actor_id")) or string_or_empty(data.get("actorId"))
    actor_uuid = string_or_empty(data.get("actor_uuid")) or string_or_empty(data.get("actorUuid"))
    limit = clamp_int(data.get("limit", data.get("top_k", 5)), 5, 1, 20)
    include_filtered = data.get("include_filtered") is True or data.get("includeFiltered") is True

    if not query:
        return jsonify({"ok": False, "error": "query is required."}), 400

    if audience not in {"npc", "gm"}:
        return jsonify({"ok": False, "error": "audience must be 'npc' or 'gm'."}), 400

    status = get_approved_retrieval_status(collection_name)
    warnings = [status["dummy_warning"]] if status.get("dummy_warning") else []
    if not status.get("approved_collection_exists"):
        return jsonify(
            {
                "ok": True,
                "status": "ok",
                "collection": collection_name,
                "query": query,
                "audience": audience,
                "candidate_count": 0,
                "filtered_count": 0,
                "warnings": warnings,
                "results": [],
                **({"filtered": []} if include_filtered else {}),
            }
        )

    collection_count = int(status.get("approved_collection_count") or 0)
    if collection_count <= 0:
        return jsonify(
            {
                "ok": True,
                "status": "ok",
                "collection": collection_name,
                "query": query,
                "audience": audience,
                "candidate_count": 0,
                "filtered_count": 0,
                "warnings": warnings,
                "results": [],
                **({"filtered": []} if include_filtered else {}),
            }
        )

    if not embedding_backend_available():
        return jsonify(
            {
                "ok": False,
                "error": "Voyage API key is not provided by the module settings.",
                "collection": collection_name,
            }
        ), 503

    try:
        query_embedding = embed_texts([query], input_type="query")[0]
    except Exception as exc:
        return jsonify(
            {
                "ok": False,
                "error": str(exc),
                "collection": collection_name,
            }
        ), 503

    candidate_limit = min(collection_count, max(limit * 4, min(20, collection_count)))

    try:
        client = get_chroma_client()
        collection = client.get_collection(collection_name)
        raw_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_limit,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        return jsonify(
            {
                "ok": False,
                "error": str(exc),
                "collection": collection_name,
            }
        ), 503

    ids = raw_results.get("ids", [[]])[0] if isinstance(raw_results, dict) else []
    documents = raw_results.get("documents", [[]])[0] if isinstance(raw_results, dict) else []
    metadatas = raw_results.get("metadatas", [[]])[0] if isinstance(raw_results, dict) else []
    distances = raw_results.get("distances", [[]])[0] if isinstance(raw_results, dict) else []

    results: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []

    for index, result_id in enumerate(ids):
        metadata = normalize_approved_metadata(metadatas[index] if index < len(metadatas) else {})
        document = documents[index] if index < len(documents) else ""
        distance = distances[index] if index < len(distances) else None
        is_dummy = approved_result_is_dummy(metadata, document)
        if is_dummy and not warnings:
            warnings.append("Dummy approved retrieval record remains. Remove/replace before 025-E4-D or real-session use.")
        allowed, reason = filter_approved_retrieval_result(metadata, audience, actor_id, actor_uuid)
        if allowed and len(results) < limit:
            results.append(approved_retrieval_result_payload(str(result_id), distance, metadata, document))
            continue

        if include_filtered:
            filtered.append(
                {
                    "id": str(result_id),
                    "reason": reason or "result_limit_exceeded",
                    "entry_id": approved_metadata_string(metadata, "entry_id"),
                    "candidate_id": approved_metadata_string(metadata, "candidate_id"),
                    "target_actor_id": approved_metadata_string(metadata, "target_actor_id"),
                    "target_actor_uuid": approved_metadata_string(metadata, "target_actor_uuid"),
                    "body_hash": approved_metadata_string(metadata, "body_hash"),
                    "body_hash_status": "not_revalidated_against_foundry_flags",
                    "dummy_record": is_dummy,
                }
            )

    payload = {
        "ok": True,
        "status": "ok",
        "collection": collection_name,
        "query": query,
        "audience": audience,
        "candidate_count": len(ids),
        "filtered_count": max(0, len(ids) - len(results)),
        "warnings": warnings,
        "results": results,
    }
    if include_filtered:
        payload["filtered"] = filtered
    return jsonify(payload)


@app.route("/world-knowledge/upsert", methods=["POST", "OPTIONS"])
@require_auth
def world_knowledge_upsert() -> Any:
    world_id, scope_error = require_world_scope()
    if scope_error:
        return scope_error
    collection_name = scoped_collection_name(RED_WORLD_KNOWLEDGE_COLLECTION_NAME, world_id)
    data = json_payload()
    chunks = data.get("chunks")
    if not isinstance(chunks, list):
        chunks = []

    valid_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue

        chunk_id = string_or_empty(chunk.get("chunk_id"))
        text = chunk.get("text")
        embedding = chunk.get("embedding")
        if not chunk_id or not isinstance(text, str) or not text.strip():
            continue

        valid_chunks.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                # Pre-computed embedding is optional. When absent the server embeds
                # the text as a DOCUMENT (input_type="document") so the index uses
                # Voyage's document/query asymmetry. Queries stay on
                # /world-knowledge/embed-query (input_type="query").
                "embedding": embedding if (isinstance(embedding, list) and embedding) else None,
                "metadata": normalize_world_knowledge_metadata(chunk.get("metadata")),
            }
        )

    if not valid_chunks:
        return jsonify({"error": "No valid chunks provided."}), 400

    pending = [chunk for chunk in valid_chunks if chunk["embedding"] is None]
    if pending:
        if not embedding_backend_available():
            return jsonify(
                {
                    "ok": False,
                    "error": "Voyage API key is not provided by the module settings.",
                    "voyage_called": False,
                    "chroma_written": False,
                }
            ), 503
        try:
            document_embeddings = embed_texts([chunk["text"] for chunk in pending], input_type="document")
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc), "voyage_called": True, "chroma_written": False}), 503
        for chunk, embedding in zip(pending, document_embeddings):
            chunk["embedding"] = embedding

    valid_chunks = [chunk for chunk in valid_chunks if isinstance(chunk["embedding"], list) and chunk["embedding"]]
    if not valid_chunks:
        return jsonify({"error": "No valid chunks provided."}), 400

    try:
        collection = get_world_knowledge_collection(world_id)
        collection.upsert(
            ids=[chunk["chunk_id"] for chunk in valid_chunks],
            embeddings=[chunk["embedding"] for chunk in valid_chunks],
            documents=[chunk["text"] for chunk in valid_chunks],
            metadatas=[chunk["metadata"] for chunk in valid_chunks],
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "collection": collection_name}), 503

    return jsonify({"status": "success", "upserted": len(valid_chunks)})


@app.route("/world-knowledge/delete", methods=["POST", "OPTIONS"])
@require_auth
def world_knowledge_delete() -> Any:
    # GM-트리거 정리용. chunk_ids(레거시 월드지식 인덱서) 또는 ids/where(팩션 clean rebuild) 수용.
    # 빈 입력(ids/필터 모두 없음)으로 전체 삭제하는 것은 거부한다.
    world_id, scope_error = require_world_scope()
    if scope_error:
        return scope_error
    collection_name = scoped_collection_name(RED_WORLD_KNOWLEDGE_COLLECTION_NAME, world_id)
    data = json_payload()
    raw_ids = data.get("ids")
    if not isinstance(raw_ids, list):
        raw_ids = data.get("chunk_ids")
    where = data.get("where")
    valid_ids = [string_or_empty(x) for x in raw_ids if string_or_empty(x)] if isinstance(raw_ids, list) else []
    has_ids = bool(valid_ids)
    has_where = isinstance(where, dict) and bool(where)
    if not has_ids and not has_where:
        return jsonify({"error": "Provide ids/chunk_ids or a non-empty where filter."}), 400

    try:
        collection = get_world_knowledge_collection(world_id)
        deleted = 0
        try:
            if has_where:
                existing = collection.get(where=where)
            else:
                existing = collection.get(ids=valid_ids)
            if isinstance(existing, dict):
                deleted = len(existing.get("ids") or [])
        except Exception:
            deleted = 0
        if has_where and has_ids:
            collection.delete(ids=valid_ids, where=where)
        elif has_where:
            collection.delete(where=where)
        else:
            collection.delete(ids=valid_ids)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "collection": collection_name}), 503

    return jsonify({"status": "success", "deleted": deleted, "requested": len(valid_ids)})


@app.route("/rerank", methods=["POST", "OPTIONS"])
@require_auth
def rerank_route() -> Any:
    data = json_payload()
    query = string_or_empty(data.get("query"))
    documents = data.get("documents")
    if not query:
        return jsonify({"ok": False, "error": "query is required."}), 400
    if not isinstance(documents, list) or not documents:
        return jsonify({"ok": False, "error": "documents must be a non-empty list."}), 400
    documents = [string_or_empty(document) for document in documents]
    if not all(documents):
        return jsonify({"ok": False, "error": "documents must all be non-empty strings."}), 400

    requested_backend = string_or_empty(data.get("backend")).lower()
    if requested_backend == "voyage":
        backend = "voyage" if resolve_voyage_api_key() else None
    elif requested_backend == "local":
        backend = "local" if LOCAL_RERANK_ENABLED else None
    else:
        backend = resolve_rerank_backend()
    if backend is None:
        return jsonify({"ok": False, "error": "No rerank backend available.", "rerank_backend": None}), 503

    requested_model = string_or_empty(data.get("model")) or None
    top_k_raw = data.get("top_k")
    top_k = top_k_raw if isinstance(top_k_raw, int) and top_k_raw > 0 else None

    try:
        results = rerank_documents(query, documents, top_k=top_k, model=requested_model, backend=backend)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "rerank_backend": backend}), 503

    return jsonify(
        {
            "ok": True,
            "results": results,
            "rerank_backend": backend,
            "model": (requested_model or VOYAGE_RERANK_MODEL) if backend == "voyage" else LOCAL_RERANK_MODEL,
        }
    )


@app.route("/world-knowledge/embed-query", methods=["POST", "OPTIONS"])
@require_auth
def world_knowledge_embed_query() -> Any:
    data = json_payload()
    query = string_or_empty(data.get("query"))
    if not query:
        return jsonify({"ok": False, "error": "query is required."}), 400

    if not embedding_backend_available():
        return jsonify(
            {
                "ok": False,
                "error": "Voyage API key is not provided by the module settings.",
                "voyage_called": False,
                "chroma_written": False,
            }
        ), 503

    try:
        embeddings = embed_texts([query], input_type="query")
    except Exception as exc:
        return jsonify(
            {
                "ok": False,
                "error": str(exc),
                "voyage_called": True,
                "chroma_written": False,
            }
        ), 503

    if not embeddings or not isinstance(embeddings[0], list):
        return jsonify(
            {
                "ok": False,
                "error": "Voyage embedding response did not include a query embedding.",
                "voyage_called": True,
                "chroma_written": False,
            }
        ), 503

    return jsonify(
        {
            "ok": True,
            "status": "ok",
            "input_type": "query",
            "model": active_embedding_model(),
            "output_dimension": VOYAGE_EMBEDDING_OUTPUT_DIMENSION,
            "embeddings": embeddings,
            "embedding": embeddings[0],
            "voyage_called": True,
            "chroma_written": False,
        }
    )


@app.route("/world-knowledge/search", methods=["POST", "OPTIONS"])
@require_auth
def world_knowledge_search() -> Any:
    world_id, scope_error = require_world_scope()
    if scope_error:
        return scope_error
    collection_name = scoped_collection_name(RED_WORLD_KNOWLEDGE_COLLECTION_NAME, world_id)
    data = json_payload()
    query_embedding = data.get("query_embedding")
    if not query_embedding:
        return jsonify({"error": "Missing query_embedding."}), 400

    top_k = clamp_int(data.get("topK", data.get("top_k", 3)), 3, 1, 50)
    axes = normalize_world_knowledge_axes(data.get("axes"))
    where = build_world_knowledge_where(axes, data.get("where_extra"))

    try:
        collection = get_world_knowledge_collection(world_id)
        collection_count = int(collection.count())
        if collection_count <= 0:
            return jsonify({"ok": True, "results": []})

        raw_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection_count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "collection": collection_name}), 503

    ids_rows = raw_results.get("ids", [[]]) if isinstance(raw_results, dict) else [[]]
    document_rows = raw_results.get("documents", [[]]) if isinstance(raw_results, dict) else [[]]
    metadata_rows = raw_results.get("metadatas", [[]]) if isinstance(raw_results, dict) else [[]]
    distance_rows = raw_results.get("distances", [[]]) if isinstance(raw_results, dict) else [[]]
    ids = ids_rows[0] if ids_rows and isinstance(ids_rows[0], list) else []
    documents = document_rows[0] if document_rows and isinstance(document_rows[0], list) else []
    metadatas = metadata_rows[0] if metadata_rows and isinstance(metadata_rows[0], list) else []
    distances = distance_rows[0] if distance_rows and isinstance(distance_rows[0], list) else []

    results: list[dict[str, Any]] = []
    for index, result_id in enumerate(ids or []):
        metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
        document = documents[index] if index < len(documents) else ""
        distance = distances[index] if index < len(distances) else None
        results.append(
            {
                "chunk_id": str(result_id),
                "text": document or "",
                "metadata": metadata,
                "distance": distance,
            }
        )

    return jsonify({"ok": True, "results": results})


@app.route("/world-knowledge/status", methods=["GET", "OPTIONS"])
@require_auth
def world_knowledge_status() -> Any:
    world_id, scope_error = require_world_scope()
    if scope_error:
        return scope_error
    collection_name = scoped_collection_name(RED_WORLD_KNOWLEDGE_COLLECTION_NAME, world_id)
    try:
        collection_names = list_chroma_collection_names_without_create()
        exists = collection_name in collection_names
        count = get_chroma_collection_count_without_create(collection_name) if exists else None
        return jsonify(
            {
                "ok": True,
                "collection": collection_name,
                "exists": exists,
                "count": count,
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "collection": collection_name}), 503


@app.route("/gm-bot-knowledge/upsert", methods=["POST", "OPTIONS"])
@require_auth
def gm_bot_knowledge_upsert() -> Any:
    world_id, scope_error = require_world_scope()
    if scope_error:
        return scope_error
    collection_name = scoped_collection_name(GM_BOT_KNOWLEDGE_COLLECTION_NAME, world_id)
    data = json_payload()
    chunks = data.get("chunks")
    if not isinstance(chunks, list):
        chunks = []

    valid_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue

        chunk_id = string_or_empty(chunk.get("chunk_id"))
        text = chunk.get("text")
        embedding = chunk.get("embedding")
        if not chunk_id or not isinstance(text, str) or not text.strip():
            continue

        valid_chunks.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                # Pre-computed embedding optional. Absent → server embeds the text as a
                # DOCUMENT (input_type="document"); `/gm-bot-knowledge/embed-query` stays
                # query-typed for reads (mirrors /world-knowledge/upsert, 2026-06-15).
                "embedding": embedding if (isinstance(embedding, list) and embedding) else None,
                "metadata": normalize_world_knowledge_metadata(chunk.get("metadata")),
            }
        )

    if not valid_chunks:
        return jsonify({"error": "No valid chunks provided."}), 400

    pending = [chunk for chunk in valid_chunks if chunk["embedding"] is None]
    if pending:
        if not embedding_backend_available():
            return jsonify(
                {
                    "ok": False,
                    "error": "Voyage API key is not provided by the module settings.",
                    "voyage_called": False,
                    "chroma_written": False,
                }
            ), 503
        try:
            document_embeddings = embed_texts([chunk["text"] for chunk in pending], input_type="document")
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc), "voyage_called": True, "chroma_written": False}), 503
        for chunk, embedding in zip(pending, document_embeddings):
            chunk["embedding"] = embedding

    valid_chunks = [chunk for chunk in valid_chunks if isinstance(chunk["embedding"], list) and chunk["embedding"]]
    if not valid_chunks:
        return jsonify({"error": "No valid chunks provided."}), 400

    try:
        collection = get_gm_bot_knowledge_collection(world_id)
        collection.upsert(
            ids=[chunk["chunk_id"] for chunk in valid_chunks],
            embeddings=[chunk["embedding"] for chunk in valid_chunks],
            documents=[chunk["text"] for chunk in valid_chunks],
            metadatas=[chunk["metadata"] for chunk in valid_chunks],
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "collection": collection_name}), 503

    return jsonify({"status": "success", "upserted": len(valid_chunks)})


@app.route("/gm-bot-knowledge/embed-query", methods=["POST", "OPTIONS"])
@require_auth
def gm_bot_knowledge_embed_query() -> Any:
    data = json_payload()
    query = string_or_empty(data.get("query"))
    if not query:
        return jsonify({"ok": False, "error": "query is required."}), 400

    if not embedding_backend_available():
        return jsonify(
            {
                "ok": False,
                "error": "Voyage API key is not provided by the module settings.",
                "voyage_called": False,
                "chroma_written": False,
            }
        ), 503

    try:
        embeddings = embed_texts([query], input_type="query")
    except Exception as exc:
        return jsonify(
            {
                "ok": False,
                "error": str(exc),
                "voyage_called": True,
                "chroma_written": False,
            }
        ), 503

    if not embeddings or not isinstance(embeddings[0], list):
        return jsonify(
            {
                "ok": False,
                "error": "Voyage embedding response did not include a query embedding.",
                "voyage_called": True,
                "chroma_written": False,
            }
        ), 503

    return jsonify(
        {
            "ok": True,
            "status": "ok",
            "input_type": "query",
            "model": active_embedding_model(),
            "output_dimension": VOYAGE_EMBEDDING_OUTPUT_DIMENSION,
            "embeddings": embeddings,
            "embedding": embeddings[0],
            "voyage_called": True,
            "chroma_written": False,
        }
    )


@app.route("/gm-bot-knowledge/search", methods=["POST", "OPTIONS"])
@require_auth
def gm_bot_knowledge_search() -> Any:
    world_id, scope_error = require_world_scope()
    if scope_error:
        return scope_error
    collection_name = scoped_collection_name(GM_BOT_KNOWLEDGE_COLLECTION_NAME, world_id)
    data = json_payload()
    query_embedding = data.get("query_embedding")
    if not query_embedding:
        return jsonify({"error": "Missing query_embedding."}), 400

    top_k = clamp_int(data.get("topK", data.get("top_k", 5)), 5, 1, 50)

    try:
        collection = get_gm_bot_knowledge_collection(world_id)
        collection_count = int(collection.count())
        if collection_count <= 0:
            return jsonify({"ok": True, "results": []})

        raw_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection_count),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "collection": collection_name}), 503

    ids_rows = raw_results.get("ids", [[]]) if isinstance(raw_results, dict) else [[]]
    document_rows = raw_results.get("documents", [[]]) if isinstance(raw_results, dict) else [[]]
    metadata_rows = raw_results.get("metadatas", [[]]) if isinstance(raw_results, dict) else [[]]
    distance_rows = raw_results.get("distances", [[]]) if isinstance(raw_results, dict) else [[]]
    ids = ids_rows[0] if ids_rows and isinstance(ids_rows[0], list) else []
    documents = document_rows[0] if document_rows and isinstance(document_rows[0], list) else []
    metadatas = metadata_rows[0] if metadata_rows and isinstance(metadata_rows[0], list) else []
    distances = distance_rows[0] if distance_rows and isinstance(distance_rows[0], list) else []

    results: list[dict[str, Any]] = []
    for index, result_id in enumerate(ids or []):
        metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
        document = documents[index] if index < len(documents) else ""
        distance = distances[index] if index < len(distances) else None
        results.append(
            {
                "chunk_id": str(result_id),
                "text": document or "",
                "metadata": metadata,
                "distance": distance,
            }
        )

    return jsonify({"ok": True, "results": results})


@app.route("/gm-bot-knowledge/delete", methods=["POST", "OPTIONS"])
@require_auth
def gm_bot_knowledge_delete() -> Any:
    world_id, scope_error = require_world_scope()
    if scope_error:
        return scope_error
    collection_name = scoped_collection_name(GM_BOT_KNOWLEDGE_COLLECTION_NAME, world_id)
    data = json_payload()
    chunk_ids = data.get("chunk_ids")
    if not isinstance(chunk_ids, list) or not chunk_ids:
        return jsonify({"error": "chunk_ids is required."}), 400

    ids = [string_or_empty(chunk_id) for chunk_id in chunk_ids]
    ids = [chunk_id for chunk_id in ids if chunk_id]
    if not ids:
        return jsonify({"error": "chunk_ids is required."}), 400

    try:
        collection = get_gm_bot_knowledge_collection(world_id)
        collection.delete(ids=ids)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "collection": collection_name}), 503

    return jsonify({"status": "success", "requested": len(ids)})


@app.route("/gm-bot-knowledge/status", methods=["GET", "OPTIONS"])
@require_auth
def gm_bot_knowledge_status() -> Any:
    world_id, scope_error = require_world_scope()
    if scope_error:
        return scope_error
    collection_name = scoped_collection_name(GM_BOT_KNOWLEDGE_COLLECTION_NAME, world_id)
    try:
        collection_names = list_chroma_collection_names_without_create()
        exists = collection_name in collection_names
        count = get_chroma_collection_count_without_create(collection_name) if exists else None
        return jsonify(
            {
                "ok": True,
                "collection": collection_name,
                "exists": exists,
                "count": count,
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "collection": collection_name}), 503


@app.route("/rag/build", methods=["POST", "OPTIONS"])
@require_auth
def rag_build() -> Any:
    if not COMPANION_LEGACY_RAG_ENABLED:
        return legacy_disabled_response("rag", "COMPANION_LEGACY_RAG_ENABLED")

    data = json_payload()
    chunks = data.get("chunks") or []
    if not isinstance(chunks, list) or not chunks:
        return jsonify({"error": "No chunks provided."}), 400

    valid_chunks = [chunk for chunk in chunks if chunk.get("text") and chunk.get("embedding")]
    if not valid_chunks:
        return jsonify({"error": "No valid chunks provided."}), 400

    world, _memory = get_chroma_collections()
    existing = world.get()
    if existing and existing.get("ids"):
        world.delete(ids=existing["ids"])

    world.add(
        ids=[f"chunk_{index}" for index, _chunk in enumerate(valid_chunks)],
        embeddings=[chunk["embedding"] for chunk in valid_chunks],
        metadatas=[{"source": chunk.get("source", "unknown")} for chunk in valid_chunks],
        documents=[chunk["text"] for chunk in valid_chunks],
    )
    return jsonify({"status": "success", "count": len(valid_chunks)})


@app.route("/rag/append", methods=["POST", "OPTIONS"])
@require_auth
def rag_append() -> Any:
    if not COMPANION_LEGACY_RAG_ENABLED:
        return legacy_disabled_response("rag", "COMPANION_LEGACY_RAG_ENABLED")

    data = json_payload()
    chunks = data.get("chunks") or []
    valid_chunks = [chunk for chunk in chunks if chunk.get("text") and chunk.get("embedding")]
    if not valid_chunks:
        return jsonify({"error": "No valid chunks provided."}), 400

    world, _memory = get_chroma_collections()
    world.add(
        ids=[f"event_{uuid.uuid4().hex[:12]}" for _chunk in valid_chunks],
        embeddings=[chunk["embedding"] for chunk in valid_chunks],
        metadatas=[{"source": chunk.get("source", "unknown")} for chunk in valid_chunks],
        documents=[chunk["text"] for chunk in valid_chunks],
    )
    return jsonify({"status": "success", "count": len(valid_chunks)})


@app.route("/rag/search", methods=["POST", "OPTIONS"])
@require_auth
def rag_search() -> Any:
    if not COMPANION_LEGACY_RAG_ENABLED:
        return legacy_disabled_response("rag", "COMPANION_LEGACY_RAG_ENABLED")

    data = json_payload()
    query_embedding = data.get("query_embedding")
    top_k = int(data.get("topK", 10))
    if not query_embedding:
        return jsonify({"error": "Missing query_embedding."}), 400

    world, _memory = get_chroma_collections()
    results = world.query(query_embeddings=[query_embedding], n_results=top_k)
    if not results.get("ids") or not results["ids"][0]:
        return jsonify({"results": []})

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return jsonify({"results": [{"source": metas[i].get("source", "unknown"), "text": docs[i]} for i in range(len(docs))]})


@app.route("/rag/status", methods=["GET", "OPTIONS"])
@require_auth
def rag_status() -> Any:
    try:
        collection_names = list_chroma_collection_names()
        world_exists = RAG_COLLECTION_NAME in collection_names
        memory_exists = MEMORY_COLLECTION_NAME in collection_names
        world_count = get_chroma_collection_count_without_create(RAG_COLLECTION_NAME)
        memory_count = get_chroma_collection_count_without_create(MEMORY_COLLECTION_NAME)
        legacy_detected = [
            name
            for name in [LEGACY_RAG_COLLECTION_NAME, LEGACY_MEMORY_COLLECTION_NAME]
            if name in collection_names
        ]
        return jsonify(
            {
                "status": "ok",
                "world_count": world_count,
                "memory_count": memory_count,
                "chroma_path": str(CHROMA_PATH),
                "data_dir": str(DATA_DIR),
                "rag_collection": RAG_COLLECTION_NAME,
                "rag_collection_exists": world_exists,
                "memory_collection": MEMORY_COLLECTION_NAME,
                "memory_collection_exists": memory_exists,
                "status_read_only": True,
                "collections_created": False,
                "legacy_endpoints": legacy_endpoint_status(),
                "collections": collection_names,
                "legacy_collections_detected": legacy_detected,
                "graph_file": str(GRAPH_FILE),
                "graph": {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()},
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc), "chroma_path": str(CHROMA_PATH)}), 503


@app.route("/memory/save", methods=["POST", "OPTIONS"])
@require_auth
def memory_save() -> Any:
    if not COMPANION_LEGACY_MEMORY_ENABLED:
        return legacy_disabled_response("memory", "COMPANION_LEGACY_MEMORY_ENABLED")

    data = json_payload()
    npc_id = data.get("npc_id")
    text = data.get("text")
    embedding = data.get("embedding")
    if not npc_id or not text or not embedding:
        return jsonify({"error": "Missing npc_id, text, or embedding."}), 400

    _world, memory = get_chroma_collections()
    memory_id = f"{npc_id}_{uuid.uuid4().hex[:12]}"
    memory.add(ids=[memory_id], embeddings=[embedding], metadatas=[{"npc_id": npc_id}], documents=[text])
    return jsonify({"status": "success", "memory_id": memory_id})


@app.route("/memory/search", methods=["POST", "OPTIONS"])
@require_auth
def memory_search() -> Any:
    if not COMPANION_LEGACY_MEMORY_ENABLED:
        return legacy_disabled_response("memory", "COMPANION_LEGACY_MEMORY_ENABLED")

    data = json_payload()
    npc_id = data.get("npc_id")
    query_embedding = data.get("query_embedding")
    top_k = int(data.get("topK", 3))
    if not npc_id or not query_embedding:
        return jsonify({"error": "Missing npc_id or query_embedding."}), 400

    _world, memory = get_chroma_collections()
    results = memory.query(query_embeddings=[query_embedding], n_results=top_k, where={"npc_id": npc_id})
    if not results.get("ids") or not results["ids"][0]:
        return jsonify({"results": []})

    return jsonify({"results": results["documents"][0]})


@app.route("/memory/list", methods=["POST", "OPTIONS"])
@require_auth
def memory_list() -> Any:
    if not COMPANION_LEGACY_MEMORY_ENABLED:
        return legacy_disabled_response("memory", "COMPANION_LEGACY_MEMORY_ENABLED")

    data = json_payload()
    npc_id = data.get("npc_id")
    if not npc_id:
        return jsonify({"error": "Missing npc_id."}), 400

    _world, memory = get_chroma_collections()
    results = memory.get(where={"npc_id": npc_id})
    ids = results.get("ids", [])
    documents = results.get("documents", [])
    return jsonify({"memories": [{"id": ids[i], "text": documents[i]} for i in range(len(ids))]})


@app.route("/memory/delete_single", methods=["POST", "OPTIONS"])
@require_auth
def memory_delete_single() -> Any:
    if not COMPANION_LEGACY_MEMORY_ENABLED:
        return legacy_disabled_response("memory", "COMPANION_LEGACY_MEMORY_ENABLED")

    data = json_payload()
    memory_id = data.get("memory_id")
    if not memory_id:
        return jsonify({"error": "Missing memory_id."}), 400

    _world, memory = get_chroma_collections()
    memory.delete(ids=[memory_id])
    return jsonify({"status": "success"})


@app.route("/memory/edit", methods=["POST", "OPTIONS"])
@require_auth
def memory_edit() -> Any:
    if not COMPANION_LEGACY_MEMORY_ENABLED:
        return legacy_disabled_response("memory", "COMPANION_LEGACY_MEMORY_ENABLED")

    data = json_payload()
    memory_id = data.get("memory_id")
    text = data.get("text")
    embedding = data.get("embedding")
    if not memory_id or not text or not embedding:
        return jsonify({"error": "Missing memory_id, text, or embedding."}), 400

    _world, memory = get_chroma_collections()
    memory.update(ids=[memory_id], documents=[text], embeddings=[embedding])
    return jsonify({"status": "success"})


@app.route("/graph/add", methods=["POST", "OPTIONS"])
@require_auth
def graph_add() -> Any:
    if not COMPANION_LEGACY_GRAPH_ENABLED:
        return legacy_disabled_response("graph", "COMPANION_LEGACY_GRAPH_ENABLED")

    data = json_payload()
    triplets = data.get("triplets") or []
    if not isinstance(triplets, list) or not triplets:
        return jsonify({"error": "No triplets provided."}), 400

    with graph_lock:
        for triplet in triplets:
            source = str(triplet.get("source", "")).strip()
            target = str(triplet.get("target", "")).strip()
            relation = str(triplet.get("relation", "")).strip()
            if source and target:
                graph.add_edge(source, target, relation=relation or None)
        save_graph()

    return jsonify({"status": "success", "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()})


@app.route("/graph/search", methods=["POST", "OPTIONS"])
@require_auth
def graph_search() -> Any:
    if not COMPANION_LEGACY_GRAPH_ENABLED:
        return legacy_disabled_response("graph", "COMPANION_LEGACY_GRAPH_ENABLED")

    data = json_payload()
    keywords = [str(keyword).strip() for keyword in data.get("keywords", []) if str(keyword).strip()]
    if not keywords:
        return jsonify({"results": ""})

    context_lines: set[str] = set()
    with graph_lock:
        found_nodes = [node for node in graph.nodes() if any(keyword in node or node in keyword for keyword in keywords)]
        for node in found_nodes:
            for neighbor in graph.neighbors(node):
                relation = graph.get_edge_data(node, neighbor).get("relation") or "related"
                context_lines.add(f"[{node}] ──({relation})──➔ [{neighbor}]")

    return jsonify({"results": "\n".join(sorted(context_lines))})


@app.route("/stt", methods=["POST", "OPTIONS"])
@require_auth
def stt_proxy() -> Any:
    data = json_payload()
    audio_base64 = data.get("audio")
    phrases = data.get("phrases") or []

    if not audio_base64:
        return jsonify({"error": "Missing audio."}), 400

    try:
        audio_data = base64.b64decode(audio_base64)
    except Exception as exc:
        return jsonify({"error": "audio_decode_failed", "detail": str(exc)}), 400

    try:
        model = get_whisper_model()
    except RuntimeError as exc:
        return jsonify({"error": "stt_model_unavailable", "detail": str(exc)}), 503

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_audio:
            temp_audio.write(audio_data)
            temp_path = temp_audio.name

        context_prompt = ", ".join(str(p) for p in phrases if p)

        try:
            segments, _info = model.transcribe(
                temp_path,
                language=STT_LANGUAGE,
                # Proper-noun biasing: pass the client's expected phrases (NPC names/
                # aliases + campaign glossary) as `hotwords` — faster-whisper biases the
                # decoder toward these more strongly than the old initial_prompt hack.
                hotwords=context_prompt or None,
                beam_size=STT_BEAM_SIZE,
                temperature=[0.0, 0.2, 0.4],
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )
        except Exception as exc:
            return jsonify({"error": "transcribe_failed", "detail": str(exc)}), 500

        seg_list = list(segments)
        valid = [s for s in seg_list if getattr(s, "no_speech_prob", 0.0) < 0.5]
        transcript = "".join(s.text for s in valid).strip()

        if valid:
            avg_lp = sum(s.avg_logprob for s in valid) / len(valid)
            confidence = round(max(0.0, min(1.0, (avg_lp + 3.0) / 3.0)), 3)
        else:
            confidence = 0.0

        min_no_speech = round(
            min((getattr(s, "no_speech_prob", 1.0) for s in seg_list), default=1.0), 3
        )

        return jsonify({
            "text": transcript,
            "confidence": confidence,
            "no_speech_prob": min_no_speech,
        })
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


@app.route("/v1beta/openai/<path:subpath>", methods=["GET", "POST", "OPTIONS"])
@require_auth
def vertex_openai_proxy(subpath: str) -> Any:
    try:
        import google.auth
        import requests
        from google.auth.transport.requests import Request as GoogleAuthRequest
    except Exception as exc:  # pragma: no cover - optional dependency
        return jsonify({"error": f"Vertex proxy dependencies are missing: {exc}"}), 503

    preconfigured_project = resolve_vertex_project_id()
    quota_project_id = resolve_vertex_quota_project_id(preconfigured_project)
    try:
        credentials, default_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
            quota_project_id=quota_project_id,
        )
    except TypeError:  # Older google-auth fallback.
        credentials, default_project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    vertex_project_id = resolve_vertex_project_id(default_project)
    if not vertex_project_id:
        return jsonify({"error": "Vertex proxy is not configured. Set VERTEX_PROJECT_ID or configure a Google ADC default project."}), 503

    if not credentials.valid or not credentials.token:
        credentials.refresh(GoogleAuthRequest())

    if VERTEX_REGION == "global":
        google_api_base = f"https://aiplatform.googleapis.com/v1/projects/{vertex_project_id}/locations/global/endpoints/openapi"
    else:
        google_api_base = f"https://{VERTEX_REGION}-aiplatform.googleapis.com/v1/projects/{vertex_project_id}/locations/{VERTEX_REGION}/endpoints/openapi"

    payload = json_payload()
    if payload.get("model") and not str(payload["model"]).startswith("google/"):
        payload["model"] = f"google/{payload['model']}"

    for field in ["frequency_penalty", "presence_penalty", "user"]:
        payload.pop(field, None)

    try:
        response = requests.post(
            f"{google_api_base}/{subpath}",
            headers={"Authorization": f"Bearer {credentials.token}", "Content-Type": "application/json"},
            json=payload,
            stream=bool(payload.get("stream", False)),
            timeout=120,
        )
    except requests.Timeout:
        return jsonify({
            "error": {
                "message": "Vertex upstream request timed out.",
                "type": "timeout",
                "code": "upstream_timeout",
                "status": "timeout",
                "upstream": "vertex_openai_proxy",
            }
        }), 504
    except requests.RequestException as exc:
        return jsonify({
            "error": {
                "message": "Vertex upstream request failed before a response was received.",
                "type": "network_error",
                "code": exc.__class__.__name__,
                "status": "network_error",
                "upstream": "vertex_openai_proxy",
            }
        }), 502

    headers = vertex_proxy_response_headers(response.headers)
    if response.status_code >= 400:
        return jsonify(normalize_vertex_proxy_error_response(response)), response.status_code, headers
    response_body, response_content_type = normalize_vertex_proxy_success_response(response, payload)
    return Response(
        response_body,
        status=response.status_code,
        content_type=response_content_type,
        headers=headers,
    )


# General OpenAI-compatible passthrough proxy ("proxy mode"). Unlike the Vertex route
# above (ADC / aiplatform.googleapis.com only), this keeps an upstream provider's API
# key in the server .env so the Foundry module never ships it to the browser. The
# module points its model base URL at /llm/<provider> and authenticates with
# COMPANION_API_KEY (if set); the server swaps in the real upstream key per request.
LLM_PROXY_DEFAULT_UPSTREAMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "openai": ("https://api.openai.com/v1", ("OPENAI_API_KEY",)),
    "anthropic": ("https://api.anthropic.com/v1", ("ANTHROPIC_API_KEY",)),
    "openrouter": ("https://openrouter.ai/api/v1", ("OPENROUTER_API_KEY",)),
    "google": ("https://generativelanguage.googleapis.com/v1beta/openai", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
}

# Proxy-mode keystore: upstream API keys entered in the Foundry Settings Hub and POSTed
# to /llm/keys are persisted here (next to the executable / script, like .env) so they
# survive a server restart without the GM re-entering them. Plaintext, same threat model
# as .env — never distribute it. Env vars (OPENAI_API_KEY, etc.) still win over it.
LLM_PROXY_KEYSTORE_PATH = _stable_state_path("llm_proxy_keys.json")


def load_llm_proxy_keystore() -> dict[str, str]:
    """Read the proxy keystore file into a {provider: key} dict (best-effort)."""
    try:
        with open(LLM_PROXY_KEYSTORE_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(provider): str(key).strip()
        for provider, key in data.items()
        if provider in LLM_PROXY_DEFAULT_UPSTREAMS and isinstance(key, str) and key.strip()
    }


LLM_PROXY_KEYSTORE: dict[str, str] = load_llm_proxy_keystore()


def set_llm_proxy_key(provider: str, api_key: str) -> bool:
    """Set (or clear, when empty) a provider's proxy key; persist to disk.

    Returns True when the provider now has a key, False when cleared/unset/unknown.
    """
    if provider not in LLM_PROXY_DEFAULT_UPSTREAMS:
        return False
    key = (api_key or "").strip()
    if key:
        LLM_PROXY_KEYSTORE[provider] = key
    else:
        LLM_PROXY_KEYSTORE.pop(provider, None)
    try:
        with open(LLM_PROXY_KEYSTORE_PATH, "w", encoding="utf-8") as handle:
            json.dump(LLM_PROXY_KEYSTORE, handle, ensure_ascii=False, indent=2)
    except OSError as exc:
        print(f"[LLM proxy] Failed to persist keystore: {exc}")
    return bool(key)


def llm_proxy_key_status() -> dict[str, bool]:
    """Per-provider configured booleans (env OR keystore). Never returns key values."""
    return {provider: bool(resolve_llm_proxy_upstream(provider)[1]) for provider in LLM_PROXY_DEFAULT_UPSTREAMS}


def resolve_llm_proxy_upstream(provider: str) -> tuple[str | None, str | None, tuple[str, ...]]:
    """Resolve (base_url, api_key, key_env_names) for a passthrough provider.

    base_url is overridable via LLM_PROXY_<PROVIDER>_BASE_URL; api_key is the first
    non-empty value among the provider's env aliases, falling back to the keystore
    (Settings Hub entry). Returns base_url=None for an unknown provider and api_key=None
    when no key is configured anywhere.
    """
    entry = LLM_PROXY_DEFAULT_UPSTREAMS.get(provider)
    if not entry:
        return None, None, ()
    default_base, key_envs = entry
    base_url = (os.getenv(f"LLM_PROXY_{provider.upper()}_BASE_URL", "") or "").strip() or default_base
    api_key = ""
    for env_name in key_envs:
        api_key = (os.getenv(env_name, "") or "").strip()
        if api_key:
            break
    if not api_key:
        api_key = (LLM_PROXY_KEYSTORE.get(provider, "") or "").strip()
    return base_url.rstrip("/"), (api_key or None), key_envs


@app.route("/llm/<provider>/<path:subpath>", methods=["POST", "OPTIONS"])
@require_auth
def llm_passthrough_proxy(provider: str, subpath: str) -> Any:
    try:
        import requests
    except Exception as exc:  # pragma: no cover - optional dependency
        return jsonify({"error": f"LLM proxy dependencies are missing: {exc}"}), 503

    base_url, api_key, key_envs = resolve_llm_proxy_upstream(provider)
    if not base_url:
        return jsonify({
            "error": {
                "message": f"Unknown LLM proxy provider '{provider}'.",
                "type": "invalid_request",
                "code": "unknown_provider",
                "upstream": "llm_passthrough_proxy",
            }
        }), 404
    if not api_key:
        return jsonify({
            "error": {
                "message": f"LLM proxy for '{provider}' is not configured. Set one of: {', '.join(key_envs)}.",
                "type": "configuration_error",
                "code": "missing_upstream_key",
                "upstream": "llm_passthrough_proxy",
            }
        }), 503

    payload = json_payload()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if provider == "openrouter":
        headers["X-Title"] = "BGH Red Thread AI"

    try:
        response = requests.post(
            f"{base_url}/{subpath}",
            headers=headers,
            json=payload,
            stream=bool(payload.get("stream", False)),
            timeout=120,
        )
    except requests.Timeout:
        return jsonify({
            "error": {
                "message": "LLM upstream request timed out.",
                "type": "timeout",
                "code": "upstream_timeout",
                "status": "timeout",
                "upstream": "llm_passthrough_proxy",
            }
        }), 504
    except requests.RequestException as exc:
        return jsonify({
            "error": {
                "message": "LLM upstream request failed before a response was received.",
                "type": "network_error",
                "code": exc.__class__.__name__,
                "status": "network_error",
                "upstream": "llm_passthrough_proxy",
            }
        }), 502

    out_headers = vertex_proxy_response_headers(response.headers)
    return Response(
        response.content,
        status=response.status_code,
        content_type=response.headers.get("Content-Type"),
        headers=out_headers,
    )


@app.route("/llm/keys", methods=["POST", "OPTIONS"])
@require_auth
def llm_proxy_set_keys() -> Any:
    """Store proxy-mode upstream keys from the Settings Hub. Accepts {provider, api_key}
    or {keys: {provider: key, ...}}. An empty key clears that provider. Returns configured
    booleans only — never echoes key values back to the browser."""
    payload = json_payload()
    updates = payload.get("keys") if isinstance(payload.get("keys"), dict) else None
    if updates is None and payload.get("provider"):
        updates = {payload.get("provider"): payload.get("api_key", "")}
    if not isinstance(updates, dict) or not updates:
        return jsonify({"error": "Provide {provider, api_key} or {keys: {...}}."}), 400

    applied = []
    for provider, key in updates.items():
        if provider not in LLM_PROXY_DEFAULT_UPSTREAMS:
            continue
        set_llm_proxy_key(provider, str(key or ""))
        applied.append(provider)

    return jsonify({"ok": True, "applied": applied, "configured": llm_proxy_key_status()})


@app.route("/llm/keys/status", methods=["GET", "OPTIONS"])
@require_auth
def llm_proxy_keys_status_route() -> Any:
    """Per-provider configured booleans for the Settings Hub (no key values)."""
    return jsonify({"configured": llm_proxy_key_status()})


@app.route("/companion/auth/status", methods=["GET", "OPTIONS"])
def companion_auth_status_route() -> Any:
    """Public: lets the Settings Hub see whether a companion auth key is configured (and
    whether the env var is forcing it) before it can authenticate. No key value returned."""
    if request.method == "OPTIONS":
        return "", 204
    env_key = companion_auth_env_key()
    return jsonify({
        "configured": bool(resolve_companion_auth_key()),
        "source": "env" if env_key else ("hub" if PERSISTED_COMPANION_AUTH_KEY else "none"),
        "env_locked": bool(env_key),
    })


@app.route("/companion/auth/set", methods=["POST", "OPTIONS"])
def companion_auth_set_route() -> Any:
    """Set the companion auth key (== COMPANION_API_KEY) from the Settings Hub.
    Trust-on-first-use: allowed without auth only when no key is configured yet; changing
    or clearing an existing key requires the current key in the Authorization header.
    Refused (409) when COMPANION_API_KEY is set in the server .env (env wins)."""
    if request.method == "OPTIONS":
        return "", 204
    if companion_auth_env_key():
        return jsonify({
            "error": "COMPANION_API_KEY is set in the server .env and takes precedence; edit .env to change it.",
            "env_locked": True,
        }), 409
    current = PERSISTED_COMPANION_AUTH_KEY
    if current and request.headers.get("Authorization", "") != f"Bearer {current}":
        return jsonify({"error": "Unauthorized. Provide the current companion auth key to change it."}), 401
    payload = json_payload()
    set_persisted_companion_auth_key(str(payload.get("api_key", "") or ""))
    return jsonify({"ok": True, "configured": bool(resolve_companion_auth_key()), "env_locked": False})


load_graph()

if STT_ENABLED:
    try:
        get_whisper_model()
        print(f"[STT] Whisper model '{STT_MODEL}' loaded on {STT_DEVICE}.")
    except Exception as _warmup_exc:
        print(f"[STT] Whisper model warmup skipped: {_warmup_exc}")

if __name__ == "__main__":
    try:
        from waitress import serve
    except Exception:
        app.run(host=HOST, port=PORT)
    else:
        print(f"{APP_NAME} listening on http://{HOST}:{PORT}")
        if not resolve_companion_auth_key():
            print("WARNING: No companion auth key set (COMPANION_API_KEY env or Settings Hub). Sensitive endpoints are unauthenticated.")
        configured_project = resolve_vertex_project_id()
        configured_quota_project = resolve_vertex_quota_project_id(configured_project)
        if configured_project:
            print(f"Vertex proxy project configured: {configured_project} ({VERTEX_REGION})")
            if configured_quota_project:
                print(f"Vertex quota project configured: {configured_quota_project}")
        else:
            print("WARNING: Vertex proxy project is not configured. Set VERTEX_PROJECT_ID or Google ADC default project.")
        serve(app, host=HOST, port=PORT)
