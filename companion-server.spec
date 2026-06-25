# -*- mode: python ; coding: utf-8 -*-
# PyInstaller build spec for the BGH Red Thread AI companion server.
#
# Build (per OS, from this directory, inside the venv):
#   pyinstaller --noconfirm companion-server.spec
#
# Produces dist/companion-server/  (onedir). The buyer runs the
# companion-server[.exe] inside that folder; no Python/pip/venv needed.
#
# STT (faster-whisper / CTranslate2 / PyAV) IS frozen into this bundle (GM decision
# 2026-06-24: accept the ~10GB total to kill the beta install friction). The Whisper
# CT2 model is pre-fetched by fetch_local_model.py into models/ and bundled below;
# companion_server.py's _resolve_local_stt_ref() loads it offline. faster-whisper's
# native libs (CTranslate2) + PyAV's bundled ffmpeg are gathered via collect_all.

import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = [('.env.example', '.')]
binaries = []
hiddenimports = []

# chromadb (+ its Rust bindings) and onnxruntime ship data files and native libs
# that PyInstaller's static analysis misses; collect them wholesale. The local
# embedding edition (Phase 2, decision "a") also bundles torch + the
# sentence-transformers / transformers stack so the server can embed offline
# without a Voyage API key. torch is heavy and fragile to freeze — if a platform
# build breaks here, that is the first place to look.
for pkg in (
    'chromadb',
    'chromadb_rust_bindings',
    'onnxruntime',
    'torch',
    'sentence_transformers',
    'transformers',
    'tokenizers',
    'safetensors',
    'huggingface_hub',
    'mxbai_rerank',
    # STT: faster-whisper + its CTranslate2 native runtime + PyAV's bundled ffmpeg.
    # collect_all pulls the native .dll/.pyd and faster-whisper's VAD assets that
    # PyInstaller's static analysis would otherwise miss.
    'faster_whisper',
    'ctranslate2',
    'av',
):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hidden
    except Exception:
        # Optional/renamed packages: skip if absent on this platform.
        pass

# Bundle the pre-downloaded local embedding model so the frozen server loads it
# offline (no multi-GB download on first query). fetch_local_model.py populates
# models/<name> in the build venv before PyInstaller runs; if it is absent (e.g. a
# quick local build that skipped the fetch) we ship without it and the server
# falls back to an online hub download at runtime.
if os.path.isdir('models'):
    datas += [('models', 'models')]

# voyage-4-nano ships its architecture as trust_remote_code: transformers loads
# modeling_qwen3_bidirectional.py dynamically at runtime. That file is bundled as
# DATA (under models/), so PyInstaller's static analysis never sees its imports.
# Pin the transformers submodules it imports so the frozen bundle contains them
# (verified 2026-06-14: those imports are PreTrainedModel/Qwen3Model + cache_utils,
# masking_utils, modeling_outputs, processing_utils, utils).
hiddenimports += collect_submodules('transformers.models.qwen3')
hiddenimports += [
    'transformers.cache_utils',
    'transformers.masking_utils',
    'transformers.modeling_outputs',
    'transformers.processing_utils',
    'transformers.utils',
]

# Local reranker (mxbai-rerank-large-v2) is Qwen2.5-based; transformers loads the
# qwen2 model classes at runtime. Pin those submodules so the frozen bundle has them.
# NOTE (unverified-by-freeze): these hiddenimports are best-effort; the exact set must
# be confirmed by a real freeze build + offline load test (as the qwen3 set above was
# verified 2026-06-14). See HANDOVER reranker track slice 5.
hiddenimports += collect_submodules('transformers.models.qwen2')

# Server stack + lazily-imported submodules the analyzer can miss.
hiddenimports += collect_submodules('waitress')
hiddenimports += [
    'flask_cors',
    'google.auth',
    'google.auth.transport.requests',
    'dotenv',
]

block_cipher = None

a = Analysis(
    ['companion_server.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],  # STT (faster_whisper/av/ctranslate2) now collected above, no longer excluded.
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='companion-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='companion-server',
)
