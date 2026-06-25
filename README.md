# BGH Red Thread AI Companion Server

This is the sanitized local companion server for the Foundry VTT module.
It provides:

- `/health`
- `/rag/build`, `/rag/append`, `/rag/search`
- `/memory/save`, `/memory/search`, `/memory/list`, `/memory/delete_single`, `/memory/edit`
- `/graph/add`, `/graph/search`
- optional `/stt`
- optional `/v1beta/openai/*` Vertex AI OpenAI-compatible proxy

## What is intentionally not included

Do not commit or distribute:

- real `.env` files
- API keys
- Google credentials
- `data/`, `chroma_db/`, or `world_graph.json`
- private campaign journals, actors, memories, or graph exports

## Prebuilt bundle (recommended — no Python needed)

If you downloaded the prebuilt bundle (`companion-server-windows-x64.zip`), you do
NOT need Python, pip, or a virtual environment. The bundle is **Windows-only** and
unsigned by design:

1. Unzip it anywhere.
2. Run `companion-server.exe` inside.
   - If SmartScreen warns, click **More info → Run anyway** (the build is unsigned).
3. Open `http://127.0.0.1:5000/health` — you should see `"status":"ok"`.
4. (Optional) By default the prebuilt bundle runs RAG **with no API key** using the
   bundled local embedding model (voyage-4-nano). If you prefer Voyage cloud
   embeddings, set your Voyage API key in Foundry (Control Center → Settings Hub) —
   the key is sent per-request and is NOT stored in any local file.

The bundle does **not** include optional STT (voice transcription). If you need
local STT, use the source install below with `requirements-stt.txt`.

### Where your data lives

The frozen server stores its vector index and graph in a `data/` folder **next
to the executable** (`<bundle>/data/chroma_db`, `<bundle>/data/world_graph.json`).
Back up that `data/` folder to preserve your campaign memories. When updating to a
newer bundle, keep your `data/` folder and replace the rest.

### Reset / re-test a clean install

To wipe the local data and start from an empty database (your Foundry world is
untouched — this only clears the companion server's index):

- Double-click `reset.bat` in the bundle.

Or just delete the `data/` folder manually.

### Uninstall

Stop the server, then delete the whole unzipped bundle folder. Nothing is written
elsewhere on the system (no registry entries, no Python, no virtualenv) — so
removing the folder removes everything except your Foundry world data.

The Windows bundle is produced by `.github/workflows/companion-build.yml` from
`companion-server.spec` (PyInstaller). Distribution is Windows-only and unsigned —
macOS/Linux bundles and code signing were declined (GM, 2026-06-14).

## Source install (developers / STT)

```bash
git clone https://github.com/Supple-s/bgh-redthread-companion.git
cd bgh-redthread-companion
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-basic.txt
cp .env.example .env
python companion_server.py
```

Then open:

```text
http://127.0.0.1:5000/health
```

## Optional authentication (a password for the server)

This is a **password you choose** for the companion server — not a provider API key. It
protects RAG, Memory, Graph, STT, and proxy routes. Under the hood it is the
`COMPANION_API_KEY` value; set the same password on both sides.

**Easiest — set it in Foundry, no `.env` editing.** In **Settings Hub → API / Connections →
Companion server password**, type a password and click **Apply to server**. The first time
(when none is set yet) is accepted without prior auth and persisted next to the executable
(`companion_auth.json`); from then on the server requires that password.

**Or via `.env`** (source installs / scripted setups): set `COMPANION_API_KEY` in `.env` and
the same value in the Foundry field above. An `.env` value always wins and the hub cannot
change it.

`/health` remains public so the module can diagnose connectivity without exposing secrets.

## Optional STT

Install the STT dependencies only if you need local transcription:

```bash
pip install -r requirements-stt.txt
```

**시스템 ffmpeg 불필요:** faster-whisper는 PyAV(`av` 패키지, `requirements-stt.txt`에
포함되며 ffmpeg 라이브러리를 휠에 동봉)로 webm 오디오를 디코딩합니다. 별도 시스템
ffmpeg 설치 없이 `/stt`가 동작합니다.

권장 설정:

```text
STT_ENABLED=true
STT_MODEL=large-v3-turbo   # 한국어 기준 최적 균형
STT_DEVICE=cuda             # GPU 없으면 cpu
STT_COMPUTE_TYPE=float16    # CPU면 int8
STT_LANGUAGE=ko
```

`STT_MODEL=small`은 기본값이지만 한국어 정확도가 낮습니다.
로컬 GPU가 있으면 `large-v3-turbo`를 권장합니다.

## Optional Vertex AI proxy

To use the OpenAI-compatible Vertex AI proxy route, authenticate Google ADC on the machine running this server and set:

```text
VERTEX_PROJECT_ID=your-google-cloud-project-id
VERTEX_REGION=global
# Optional. Usually same as VERTEX_PROJECT_ID.
VERTEX_QUOTA_PROJECT_ID=your-google-cloud-project-id

> Vertex uses your local Google Application Default Credentials. `COMPANION_API_KEY` is not a Vertex API key; it only protects the local companion server. If `VERTEX_PROJECT_ID` is blank, the server will also try `GOOGLE_CLOUD_PROJECT`, `GCLOUD_PROJECT`, `PROJECT_ID`, and the default project returned by Google ADC.
>
> If Python prints a Google ADC quota-project warning, set `VERTEX_QUOTA_PROJECT_ID` in `.env` or run `gcloud auth application-default set-quota-project your-google-cloud-project-id`. The warning is not a module API-key problem.
```

The Foundry model base URL can then point to:

```text
http://127.0.0.1:5000/v1beta/openai
```

Use `COMPANION_API_KEY` as the API key in Foundry when companion authentication is enabled.

## Optional LLM proxy mode (keep provider keys server-side)

By default the module is BYOK: it sends your LLM provider API key directly from the
browser to OpenAI / Anthropic / OpenRouter / Google. That key is therefore visible in
the browser (DevTools network/memory) on the machine running Foundry. **Proxy mode**
keeps the key in this server's `.env` instead, so it never reaches the browser.

This is distinct from the Vertex proxy above: Vertex uses Google ADC and only reaches
`aiplatform.googleapis.com`, while proxy mode passes through to the normal cloud APIs.

**Easiest (prebuilt bundle): enter keys in Foundry.** You do not need to edit any file.
In **Control Center → Settings Hub → API / Connections → Proxy mode keys**, type a
provider's API key and click **Save to server**. The key is sent to this companion
server and stored next to the executable (`llm_proxy_keys.json`) — never in your Foundry
world — and survives restarts. Then enable the **Proxy mode** toggle. (Tip: also set
`COMPANION_API_KEY` so only your authenticated module can write keys.)

The `.env` route below is equivalent and still supported (useful for source installs or
scripted setups); environment variables take precedence over keys entered in the hub.

1. In `.env`, set the key(s) for the providers you use:

   ```text
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   OPENROUTER_API_KEY=sk-or-...
   GEMINI_API_KEY=...          # Gemini API (AI Studio), not Vertex
   ```

   Each upstream base URL is overridable via `LLM_PROXY_<PROVIDER>_BASE_URL` (e.g.
   `LLM_PROXY_OPENAI_BASE_URL`) if you front a gateway.

2. Keep the companion server running (proxy mode needs it for every LLM call).
3. In Foundry, enable **Control Center → Settings Hub → Proxy mode**. The module then
   routes calls to `http://127.0.0.1:5000/llm/<provider>` and authenticates with
   `COMPANION_API_KEY` (if set) — no upstream key is sent from the browser.

`custom` / self-hosted models are never proxied (they already use your own base URL).
Turning proxy mode off restores BYOK (direct browser calls).

## Optional reranker (sharper memory / knowledge retrieval)

A second-stage reranker re-orders the memories, world-knowledge, and canon chunks
retrieved for NPC dialogue and the GM Bot, putting the truly relevant ones first. It is
**off by default** and safe by design: a recall floor + confidence gate + fail-open mean
results are never worse than the embedding order. Choose it in **Control Center → Settings
Hub → Campaign Memory → Reranker**:

- **Off** — no reranking (default).
- **Auto** — Voyage rerank when a Voyage key is set, otherwise the local model.
- **Voyage rerank-2.5 / 2.5-lite** — cloud reranking via your Voyage key (the same key as
  embeddings; first 200M tokens/month are free, so a campaign is effectively free).
- **Local** — the bundled open-weight `mxbai-rerank-large-v2` (multilingual incl. Korean),
  no key required. The reranker calls go through this server's `/rerank` route.

The companion server must be running for any non-Off mode.

**Source install:** the local reranker needs `requirements-local-rerank.txt` (shares the
torch / transformers stack with local embedding) and `LOCAL_RERANK_ENABLED=true`. Install
the CPU torch build first, exactly like local embedding:

```text
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-local-rerank.txt
```

**Prebuilt bundle:** `fetch_local_model.py` downloads the reranker model too, so the
frozen build ships it and `LOCAL_RERANK_ENABLED` defaults on. The reranker model is ~3GB
(the quality-first choice), so the local edition is larger. *(The frozen freeze-build of
the reranker — PyInstaller hiddenimports for the Qwen2.5-based model — still needs an
end-to-end verification build, like the local embedding model received.)*
