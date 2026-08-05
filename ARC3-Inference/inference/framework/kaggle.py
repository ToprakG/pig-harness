"""Kaggle helpers for the ARC3 duck harness."""

from __future__ import annotations

import os
from dataclasses import dataclass

from inference.utils.openai_compat import normalize_provider

DEFAULT_VLLM_WHEELHOUSE_DATASET_SOURCE = "driessmit1/arc3-vllm-h100-wheelhouse-v3"
DEFAULT_QWEN_MODEL_DATASET_SOURCE = "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"
DEFAULT_GEMMA_MODEL_DATASET_SOURCE = "driessmit1/vrfai-gemma-4-31b-it-fp8-hf-snapshot"
DEFAULT_SERVED_MODEL_NAME = "vrfai/Qwen3.6-27B-FP8"
DEFAULT_GEMMA_SERVED_MODEL_NAME = "vrfai/gemma-4-31B-it-fp8"
DEFAULT_GEMMA_MODEL_HF_REPO = "vrfai/gemma-4-31B-it-fp8"
DEFAULT_VLLM_PORT = 1234
DEFAULT_VLLM_MAX_MODEL_LEN = 65536
DEFAULT_VLLM_TENSOR_PARALLEL_SIZE = 1
DEFAULT_WHEELHOUSE_STAMP_TEXT = "vllm==0.19.0 torch==2.10.0 flashinfer==0.6.6\n"

# The 25 official ARC-AGI-3 games. The first 16 are the original Kaggle duck
# validation harness order; the remaining 9 complete the official tag set.
DUCK_HARNESS_PUBLIC_GAME_IDS: tuple[str, ...] = (
    "tn36-ef4dde99",
    "lf52-271a04aa",
    "cn04-2fe56bfb",
    "bp35-0a0ad940",
    "wa30-ee6fef47",
    "lp85-305b61c3",
    "r11l-495a7899",
    "tu93-0768757b",
    "sp80-589a99af",
    "m0r0-492f87ba",
    "vc33-5430563c",
    "ar25-0c556536",
    "ka59-38d34dbb",
    "sc25-635fd71a",
    "sk48-d8078629",
    "dc22-fdcac232",
    "cd82-fb555c5d",
    "ft09-0d8bbf25",
    "g50t-5849a774",
    "ls20-9607627b",
    "re86-8af5384d",
    "s5i5-18d95033",
    "sb26-7fbdac44",
    "su15-1944f8ab",
    "tr87-cd924810",
)


@dataclass(frozen=True)
class DuckKaggleVllmConfig:
    """Kaggle-side vLLM/model configuration declared by ``HarnessSolver``."""

    wheelhouse_dataset_source: str = DEFAULT_VLLM_WHEELHOUSE_DATASET_SOURCE
    model_dataset_source: str = DEFAULT_QWEN_MODEL_DATASET_SOURCE
    served_model_name: str = DEFAULT_SERVED_MODEL_NAME
    model_hf_repo: str = ""
    vllm_port: int = DEFAULT_VLLM_PORT
    max_model_len: int = DEFAULT_VLLM_MAX_MODEL_LEN
    tensor_parallel_size: int = DEFAULT_VLLM_TENSOR_PARALLEL_SIZE
    wheelhouse_stamp_text: str = DEFAULT_WHEELHOUSE_STAMP_TEXT
    tool_call_parser: str = "qwen3_coder"
    reasoning_parser: str = "qwen3"
    quantization: str = ""
    trust_remote_code: bool = False
    default_chat_template_kwargs: dict[str, object] | None = None


def _analyzer_provider() -> str:
    return normalize_provider(os.environ.get("LOCAL_ANALYZER_PROVIDER", "vllm"))


def _kaggle_uses_vllm() -> bool:
    return _analyzer_provider() == "vllm"


def duck_kaggle_dataset_sources(
    config: DuckKaggleVllmConfig | None = None,
) -> list[str]:
    if not _kaggle_uses_vllm():
        return []
    cfg = config or DuckKaggleVllmConfig()
    sources = [cfg.wheelhouse_dataset_source]
    if str(cfg.model_dataset_source or "").strip():
        sources.append(cfg.model_dataset_source)
    return sources


def duck_kaggle_setup_command(config: DuckKaggleVllmConfig | None = None) -> str:
    if not _kaggle_uses_vllm():
        return duck_kaggle_remote_setup_command()
    cfg = config or DuckKaggleVllmConfig()
    wheelhouse_owner, wheelhouse_slug = _split_dataset_source(
        cfg.wheelhouse_dataset_source,
        option_name="wheelhouse_dataset_source",
    )
    model_source = str(cfg.model_dataset_source or "").strip()
    if model_source:
        model_owner, model_slug = _split_dataset_source(
            model_source,
            option_name="model_dataset_source",
        )
    else:
        model_owner, model_slug = "", ""
    replacements = {
        "__WHEELHOUSE_OWNER__": repr(wheelhouse_owner),
        "__WHEELHOUSE_SLUG__": repr(wheelhouse_slug),
        "__MODEL_OWNER__": repr(model_owner),
        "__MODEL_SLUG__": repr(model_slug),
        "__SERVED_MODEL_NAME__": repr(cfg.served_model_name),
        "__VLLM_PORT__": repr(int(cfg.vllm_port)),
        "__VLLM_MAX_MODEL_LEN__": repr(int(cfg.max_model_len)),
        # The launcher's Makefile exports LOCAL_ANALYZER_CONTEXT_WINDOW from
        # JSON shared.context_window (or analyzer.context_window). Embed it
        # here so the agent's prompt budget on Kaggle is the JSON value, not
        # vllm's max-model-len. Falls back to max_model_len if unset.
        "__ANALYZER_CONTEXT_WINDOW__": repr(
            int(os.environ.get("LOCAL_ANALYZER_CONTEXT_WINDOW") or cfg.max_model_len)
        ),
        # Remaining JSON-driven analyzer/multimodal config: the launcher's
        # Makefile exports each from inference.json; embed the launcher value
        # so the rendered setup_env on Kaggle reflects JSON edits. Fallback
        # equals the historical hardcoded literal so direct kaggle.py callers
        # outside Make are unaffected.
        "__LOCAL_ANALYZER_PROVIDER__": repr(_analyzer_provider()),
        "__LOCAL_ANALYZER_APP_NAME__": repr(os.environ.get("LOCAL_ANALYZER_APP_NAME", "ARC3 Kaggle Harness")),
        "__LOCAL_ANALYZER_MAX_OUTPUT__": repr(os.environ.get("LOCAL_ANALYZER_MAX_OUTPUT", "0")),
        "__LOCAL_ANALYZER_TOOL_STEPS__": repr(os.environ.get("LOCAL_ANALYZER_TOOL_STEPS", "0")),
        "__LOCAL_ANALYZER_TOOL_TIMEOUT__": repr(os.environ.get("LOCAL_ANALYZER_TOOL_TIMEOUT", "30")),
        "__LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS__": repr(os.environ.get("LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS", "1024")),
        "__LOCAL_ANALYZER_YIELD_SECONDS__": repr(os.environ.get("LOCAL_ANALYZER_YIELD_SECONDS", "60")),
        "__LOCAL_ANALYZER_TEMPERATURE__": repr(os.environ.get("LOCAL_ANALYZER_TEMPERATURE", "0.6")),
        "__LOCAL_ANALYZER_TOP_P__": repr(os.environ.get("LOCAL_ANALYZER_TOP_P", "0.95")),
        "__LOCAL_ANALYZER_TOP_K__": repr(os.environ.get("LOCAL_ANALYZER_TOP_K", "20")),
        "__LOCAL_ANALYZER_ENABLE_THINKING__": repr(os.environ.get("LOCAL_ANALYZER_ENABLE_THINKING", "1")),
        "__MULTIMODAL_CONTEXT__": repr(os.environ.get("MULTIMODAL_CONTEXT", "current_grid")),
        "__MULTIMODAL_UPSCALE__": repr(os.environ.get("MULTIMODAL_UPSCALE", "4")),
        # Harness experiment toggles, carried from the deploy environment so a
        # submission can enable one without a code change. Default off.
        "__DUCK_DEAD_ACTION_HINTS__": repr(os.environ.get("DUCK_DEAD_ACTION_HINTS", "")),
        "__DUCK_LEVEL_CARRYOVER__": repr(os.environ.get("DUCK_LEVEL_CARRYOVER", "")),
        "__DUCK_STATIC_REGION_HINTS__": repr(os.environ.get("DUCK_STATIC_REGION_HINTS", "")),
        "__VLLM_TENSOR_PARALLEL_SIZE__": repr(int(cfg.tensor_parallel_size)),
        "__WHEELHOUSE_STAMP_TEXT__": repr(cfg.wheelhouse_stamp_text),
        "__VLLM_TOOL_CALL_PARSER__": repr(cfg.tool_call_parser),
        "__VLLM_REASONING_PARSER__": repr(cfg.reasoning_parser),
        "__VLLM_QUANTIZATION__": repr(cfg.quantization),
        "__VLLM_TRUST_REMOTE_CODE__": repr(bool(cfg.trust_remote_code)),
        "__VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS__": repr(cfg.default_chat_template_kwargs),
        "__KAGGLE_MODEL_HF_REPO__": repr(cfg.model_hf_repo),
    }
    script = _DUCK_VLLM_SETUP_SCRIPT
    for placeholder, value in replacements.items():
        script = script.replace(placeholder, value)
    return f"\"$PYTHON\" - <<'PYSETUP'\n{script}\nPYSETUP"


def duck_kaggle_teardown_command() -> str:
    if not _kaggle_uses_vllm():
        return ""
    return f"\"$PYTHON\" - <<'PYTEARDOWN'\n{_DUCK_VLLM_TEARDOWN_SCRIPT}\nPYTEARDOWN"


def duck_kaggle_remote_setup_command() -> str:
    provider = _analyzer_provider()
    base_url = (
        os.environ.get("LOCAL_ANALYZER_BASE_URL", "").strip()
        or os.environ.get("OPENAI_BASE_URL", "").strip()
        or os.environ.get("SHARED_BASE_URL", "").strip()
    )
    if not base_url:
        raise ValueError(
            "Remote Kaggle analyzer setup requires LOCAL_ANALYZER_BASE_URL or OPENAI_BASE_URL."
        )
    model_id = (
        os.environ.get("LOCAL_ANALYZER_MODEL_ID", "").strip()
        or os.environ.get("SHARED_MODEL_NAME", "").strip()
        or os.environ.get("MODEL", "").strip()
    )
    if not model_id or model_id.lower() == "local":
        raise ValueError(
            "Remote Kaggle analyzer setup requires LOCAL_ANALYZER_MODEL_ID or shared.model_name."
        )
    # Embed launcher API key into the private source bundle when present; the setup
    # script still falls back to Kaggle notebook secrets at runtime.
    api_key = (
        os.environ.get("CEREBRAS_API_KEY", "").strip()
        or os.environ.get("LOCAL_ANALYZER_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    replacements = {
        "__LOCAL_ANALYZER_PROVIDER__": repr(provider),
        "__LOCAL_ANALYZER_BASE_URL__": repr(base_url),
        "__LOCAL_ANALYZER_MODEL_ID__": repr(model_id),
        "__LOCAL_ANALYZER_API_KEY__": repr(api_key),
        "__LOCAL_ANALYZER_APP_NAME__": repr(os.environ.get("LOCAL_ANALYZER_APP_NAME", "ARC3 Kaggle Harness")),
        "__LOCAL_ANALYZER_CONTEXT_WINDOW__": repr(
            int(os.environ.get("LOCAL_ANALYZER_CONTEXT_WINDOW") or DEFAULT_VLLM_MAX_MODEL_LEN)
        ),
        "__LOCAL_ANALYZER_MAX_OUTPUT__": repr(os.environ.get("LOCAL_ANALYZER_MAX_OUTPUT", "0")),
        "__LOCAL_ANALYZER_TOOL_STEPS__": repr(os.environ.get("LOCAL_ANALYZER_TOOL_STEPS", "0")),
        "__LOCAL_ANALYZER_TOOL_TIMEOUT__": repr(os.environ.get("LOCAL_ANALYZER_TOOL_TIMEOUT", "30")),
        "__LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS__": repr(os.environ.get("LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS", "1024")),
        "__LOCAL_ANALYZER_YIELD_SECONDS__": repr(os.environ.get("LOCAL_ANALYZER_YIELD_SECONDS", "60")),
        "__LOCAL_ANALYZER_TEMPERATURE__": repr(os.environ.get("LOCAL_ANALYZER_TEMPERATURE", "0.6")),
        "__LOCAL_ANALYZER_TOP_P__": repr(os.environ.get("LOCAL_ANALYZER_TOP_P", "0.95")),
        "__LOCAL_ANALYZER_TOP_K__": repr(os.environ.get("LOCAL_ANALYZER_TOP_K", "20")),
        "__LOCAL_ANALYZER_ENABLE_THINKING__": repr(os.environ.get("LOCAL_ANALYZER_ENABLE_THINKING", "0")),
        "__MULTIMODAL_CONTEXT__": repr(os.environ.get("MULTIMODAL_CONTEXT", "current_grid")),
        "__MULTIMODAL_UPSCALE__": repr(os.environ.get("MULTIMODAL_UPSCALE", "16")),
        "__DUCK_DEAD_ACTION_HINTS__": repr(os.environ.get("DUCK_DEAD_ACTION_HINTS", "")),
        "__DUCK_LEVEL_CARRYOVER__": repr(os.environ.get("DUCK_LEVEL_CARRYOVER", "")),
        "__DUCK_STATIC_REGION_HINTS__": repr(os.environ.get("DUCK_STATIC_REGION_HINTS", "")),
    }
    script = _DUCK_REMOTE_SETUP_SCRIPT
    for placeholder, value in replacements.items():
        script = script.replace(placeholder, value)
    return f"\"$PYTHON\" - <<'PYSETUP'\n{script}\nPYSETUP"


def _split_dataset_source(value: str, *, option_name: str) -> tuple[str, str]:
    parts = str(value or "").strip().split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"{option_name} must be a Kaggle dataset ref in owner/slug format."
        )
    return parts[0], parts[1]


_DUCK_VLLM_SETUP_SCRIPT = r"""import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

WHEELHOUSE_OWNER = __WHEELHOUSE_OWNER__
WHEELHOUSE_SLUG = __WHEELHOUSE_SLUG__
MODEL_OWNER = __MODEL_OWNER__
MODEL_SLUG = __MODEL_SLUG__
SERVED_MODEL_NAME = __SERVED_MODEL_NAME__
KAGGLE_MODEL_HF_REPO = __KAGGLE_MODEL_HF_REPO__
VLLM_HOST = '127.0.0.1'
VLLM_PORT = __VLLM_PORT__
VLLM_BASE_URL = f'http://{VLLM_HOST}:{VLLM_PORT}/v1'
VLLM_MAX_MODEL_LEN = __VLLM_MAX_MODEL_LEN__
ANALYZER_CONTEXT_WINDOW = __ANALYZER_CONTEXT_WINDOW__
VLLM_TENSOR_PARALLEL_SIZE = __VLLM_TENSOR_PARALLEL_SIZE__
VLLM_TOOL_CALL_PARSER = __VLLM_TOOL_CALL_PARSER__
VLLM_REASONING_PARSER = __VLLM_REASONING_PARSER__
VLLM_QUANTIZATION = __VLLM_QUANTIZATION__
VLLM_TRUST_REMOTE_CODE = __VLLM_TRUST_REMOTE_CODE__
VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS = __VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS__
WORKING_DIR = Path(os.environ['TAAF_KAGGLE_WORKING_DIR'])
SITE_PACKAGES = WORKING_DIR / 'vllm-site-packages'
VLLM_SERVER_LOG = WORKING_DIR / 'vllm-openai-server.log'
VLLM_SERVER_PID = WORKING_DIR / 'vllm-openai-server.pid'
INSTALL_STAMP = SITE_PACKAGES / f'.{WHEELHOUSE_SLUG}'
STAMP_TEXT = __WHEELHOUSE_STAMP_TEXT__

GPU_NAME_PATTERNS = {'rtx-pro-6000': ('rtx pro 6000',), 'h100': ('h100',), 'l4': ('l4',)}


def taaf_kaggle_input_paths() -> dict[str, Path]:
    raw = os.getenv('TAAF_KAGGLE_INPUT_PATHS', '').strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError('TAAF_KAGGLE_INPUT_PATHS must contain a JSON object.')
    return {str(ref): Path(str(path)) for ref, path in data.items()}


def resolve_kaggle_dataset_path(owner: str, slug: str) -> Path:
    mapped = taaf_kaggle_input_paths().get(f'{owner}/{slug}')
    if mapped is not None:
        return mapped
    for dataset_path in (Path('/kaggle/input') / slug, Path('/kaggle/input/datasets') / owner / slug):
        if dataset_path.exists():
            return dataset_path
    return Path('/kaggle/input') / slug


WHEELHOUSE = resolve_kaggle_dataset_path(WHEELHOUSE_OWNER, WHEELHOUSE_SLUG)


def resolve_model_path() -> Path:
    if not MODEL_SLUG:
        hf_repo = (KAGGLE_MODEL_HF_REPO or SERVED_MODEL_NAME or '').strip()
        if not hf_repo:
            raise FileNotFoundError(
                'No Kaggle model dataset attached and no Hugging Face repo configured.'
            )
        print(f'Pulling model weights from Hugging Face: {hf_repo}', flush=True)
        from huggingface_hub import snapshot_download

        download_dir = WORKING_DIR / 'hf-model-snapshot'
        return Path(snapshot_download(repo_id=hf_repo, local_dir=download_dir))
    mapped = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)
    if mapped.exists():
        return mapped
    hf_repo = (KAGGLE_MODEL_HF_REPO or SERVED_MODEL_NAME or '').strip()
    if not hf_repo:
        return mapped
    print(
        f'Model dataset path missing for {MODEL_OWNER}/{MODEL_SLUG}; '
        f'downloading {hf_repo} from Hugging Face',
        flush=True,
    )
    from huggingface_hub import snapshot_download

    download_dir = WORKING_DIR / 'hf-model-snapshot'
    return Path(snapshot_download(repo_id=hf_repo, local_dir=download_dir))


MODEL_PATH = resolve_model_path()


def assert_expected_cuda_gpu() -> None:
    if not Path('/kaggle/input').exists():
        return
    assert shutil.which('nvidia-smi'), 'CUDA GPU check failed: nvidia-smi is not available.'
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
    assert result.returncode == 0, f'nvidia-smi failed: {result.stderr.strip()}'
    gpu_names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert gpu_names, 'nvidia-smi did not report any CUDA GPUs.'
    expected_gpu_type = os.getenv('KAGGLE_GPU_TYPE', 'rtx-pro-6000').strip().lower()
    expected_count = os.getenv('KAGGLE_GPU_COUNT', '1')
    if expected_count.isdigit():
        assert len(gpu_names) == int(expected_count), f'Expected {expected_count} CUDA GPU(s), found {gpu_names}'
    patterns = GPU_NAME_PATTERNS.get(expected_gpu_type, (expected_gpu_type.replace('-', ' '),))
    mismatched = [name for name in gpu_names if not any(pattern in name.lower() for pattern in patterns)]
    assert not mismatched, f'Expected GPU type {expected_gpu_type!r}, found {gpu_names}'
    print(f'CUDA GPU check passed for {expected_gpu_type} x{expected_count}: {gpu_names}', flush=True)


def vllm_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(SITE_PACKAGES) if not existing else f'{SITE_PACKAGES}{os.pathsep}{existing}'
    env.update(
        {
            'USE_TF': '0',
            'TRANSFORMERS_NO_TF': '1',
            'TRANSFORMERS_NO_TORCHVISION': '1',
            'VLLM_NO_USAGE_STATS': '1',
        }
    )
    return env


def cached_install_is_usable() -> bool:
    if not INSTALL_STAMP.exists() or INSTALL_STAMP.read_text(encoding='utf-8') != STAMP_TEXT:
        return False
    result = subprocess.run(
        [sys.executable, '-c', "import vllm, torch; print(f'Cached vLLM {vllm.__version__}, torch {torch.__version__}')"],
        env=vllm_env(),
        text=True,
    )
    return result.returncode == 0


def install_vllm_wheelhouse() -> None:
    requirements = WHEELHOUSE / 'requirements.lock'
    if not requirements.exists():
        raise FileNotFoundError(f'Missing wheelhouse lock file: {requirements}')
    if cached_install_is_usable():
        print(f'Using cached vLLM target install at {SITE_PACKAGES}', flush=True)
        return
    shutil.rmtree(SITE_PACKAGES, ignore_errors=True)
    SITE_PACKAGES.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        '-m',
        'pip',
        'install',
        '--no-index',
        '--find-links',
        str(WHEELHOUSE),
        '--requirement',
        str(requirements),
        '--target',
        str(SITE_PACKAGES),
        '--upgrade',
        '--ignore-installed',
        '--only-binary',
        ':all:',
        '--no-compile',
        '--disable-pip-version-check',
        '--no-warn-conflicts',
    ]
    print('Installing vLLM wheelhouse into', SITE_PACKAGES, flush=True)
    subprocess.run(cmd, check=True)
    INSTALL_STAMP.write_text(STAMP_TEXT, encoding='utf-8')


def request_json(url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def tail_server_log(lines: int = 80) -> str:
    if not VLLM_SERVER_LOG.exists():
        return ''
    return '\n'.join(VLLM_SERVER_LOG.read_text(encoding='utf-8', errors='replace').splitlines()[-lines:])


def wait_for_vllm_server(timeout_seconds: int = 900) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f'{VLLM_BASE_URL}/models'
    while time.monotonic() < deadline:
        if VLLM_SERVER_PID.exists():
            try:
                os.kill(int(VLLM_SERVER_PID.read_text().strip()), 0)
            except OSError as exc:
                raise RuntimeError(f'vLLM server process is not alive: {exc}\n{tail_server_log()}') from exc
        try:
            models = request_json(url, timeout=5)
            print('vLLM server ready:', models, flush=True)
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f'Timed out waiting for vLLM server at {url}.\nLast server log lines:\n{tail_server_log()}')


def start_vllm_server() -> None:
    install_vllm_wheelhouse()
    VLLM_SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    VLLM_SERVER_PID.unlink(missing_ok=True)
    log_handle = VLLM_SERVER_LOG.open('w', encoding='utf-8')
    cmd = [
        sys.executable,
        '-m',
        'vllm.entrypoints.openai.api_server',
        '--model',
        str(MODEL_PATH),
        '--served-model-name',
        SERVED_MODEL_NAME,
        '--host',
        VLLM_HOST,
        '--port',
        str(VLLM_PORT),
        '--tensor-parallel-size',
        str(VLLM_TENSOR_PARALLEL_SIZE),
        '--enable-auto-tool-choice',
        '--tool-call-parser',
        VLLM_TOOL_CALL_PARSER,
        '--generation-config',
        'vllm',
        '--enable-prefix-caching',
        '--max-model-len',
        str(VLLM_MAX_MODEL_LEN),
    ]
    if VLLM_REASONING_PARSER:
        cmd.extend(['--reasoning-parser', VLLM_REASONING_PARSER])
    if VLLM_QUANTIZATION:
        cmd.extend(['--quantization', VLLM_QUANTIZATION])
    if VLLM_TRUST_REMOTE_CODE:
        cmd.append('--trust-remote-code')
    if VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:
        cmd.extend(
            [
                '--default-chat-template-kwargs',
                json.dumps(VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS),
            ]
        )
    print('Starting vLLM OpenAI server:', ' '.join(cmd), flush=True)
    process = subprocess.Popen(cmd, env=vllm_env(), stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    VLLM_SERVER_PID.write_text(str(process.pid), encoding='utf-8')
    wait_for_vllm_server()


def run_vllm_api_smoke_test() -> None:
    payload = {
        'model': SERVED_MODEL_NAME,
        'messages': [{'role': 'user', 'content': 'Answer in one short sentence: what is 2 + 2?'}],
        'temperature': 0.0,
        'max_tokens': 96,
        'chat_template_kwargs': {'enable_thinking': False},
    }
    response = request_json(f'{VLLM_BASE_URL}/chat/completions', payload=payload, timeout=120)
    generated = response['choices'][0]['message'].get('content', '').strip()
    print('\n' + '=' * 88, flush=True)
    print('VLLM OPENAI SERVER SMOKE TEST REAL MODEL OUTPUT', flush=True)
    print('Generated:', generated, flush=True)
    print('=' * 88 + '\n', flush=True)


print(f'vLLM wheelhouse path: {WHEELHOUSE}', flush=True)
print(f'Model path: {MODEL_PATH}', flush=True)
assert_expected_cuda_gpu()
missing = [str(path) for path in (WHEELHOUSE, MODEL_PATH) if not path.exists()]
if missing:
    raise FileNotFoundError('Missing attached dataset path(s): ' + ', '.join(missing))
start_vllm_server()
run_vllm_api_smoke_test()
setup_env = {
    'USE_TF': '0',
    'TRANSFORMERS_NO_TF': '1',
    'TRANSFORMERS_NO_TORCHVISION': '1',
    'VLLM_NO_USAGE_STATS': '1',
    'PYTHONPATH': str(SITE_PACKAGES) + os.pathsep + os.environ.get('PYTHONPATH', ''),
    'LOCAL_ANALYZER_BASE_URL': VLLM_BASE_URL,
    'OPENAI_BASE_URL': VLLM_BASE_URL,
    'LOCAL_ANALYZER_PROVIDER': __LOCAL_ANALYZER_PROVIDER__,
    'OPENAI_PROVIDER': __LOCAL_ANALYZER_PROVIDER__,
    'LOCAL_ANALYZER_MODEL_ID': SERVED_MODEL_NAME,
    'INFERENCE_ANALYZER_MODEL': SERVED_MODEL_NAME,
    'LOCAL_ANALYZER_APP_NAME': __LOCAL_ANALYZER_APP_NAME__,
    'LOCAL_ANALYZER_CONTEXT_WINDOW': str(ANALYZER_CONTEXT_WINDOW),
    'LOCAL_ANALYZER_MAX_OUTPUT': __LOCAL_ANALYZER_MAX_OUTPUT__,
    'LOCAL_ANALYZER_TOOL_STEPS': __LOCAL_ANALYZER_TOOL_STEPS__,
    'LOCAL_ANALYZER_TOOL_TIMEOUT': __LOCAL_ANALYZER_TOOL_TIMEOUT__,
    'LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS': __LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS__,
    'LOCAL_ANALYZER_YIELD_SECONDS': __LOCAL_ANALYZER_YIELD_SECONDS__,
    'LOCAL_ANALYZER_TEMPERATURE': __LOCAL_ANALYZER_TEMPERATURE__,
    'LOCAL_ANALYZER_TOP_P': __LOCAL_ANALYZER_TOP_P__,
    'LOCAL_ANALYZER_TOP_K': __LOCAL_ANALYZER_TOP_K__,
    'LOCAL_ANALYZER_ENABLE_THINKING': __LOCAL_ANALYZER_ENABLE_THINKING__,
    'MULTIMODAL_CONTEXT': __MULTIMODAL_CONTEXT__,
    'MULTIMODAL_UPSCALE': __MULTIMODAL_UPSCALE__,
    'DUCK_DEAD_ACTION_HINTS': __DUCK_DEAD_ACTION_HINTS__,
    'DUCK_LEVEL_CARRYOVER': __DUCK_LEVEL_CARRYOVER__,
    'DUCK_STATIC_REGION_HINTS': __DUCK_STATIC_REGION_HINTS__,
}
setup_env_path = Path(os.environ['TAAF_KAGGLE_SETUP_ENV'])
existing_setup_env = {}
if setup_env_path.exists():
    existing_setup_env = json.loads(setup_env_path.read_text(encoding='utf-8'))
    if not isinstance(existing_setup_env, dict):
        raise RuntimeError('TAAF_KAGGLE_SETUP_ENV must contain a JSON object.')
existing_setup_env.update(setup_env)
setup_env_path.write_text(json.dumps(existing_setup_env, indent=2), encoding='utf-8')
"""

_DUCK_REMOTE_SETUP_SCRIPT = r"""import json
import os
import urllib.request
from pathlib import Path

PROVIDER = __LOCAL_ANALYZER_PROVIDER__
BASE_URL = __LOCAL_ANALYZER_BASE_URL__
MODEL_ID = __LOCAL_ANALYZER_MODEL_ID__
BUNDLED_API_KEY = __LOCAL_ANALYZER_API_KEY__
API_KEY = (
    BUNDLED_API_KEY
    or os.environ.get('CEREBRAS_API_KEY', '').strip()
    or os.environ.get('LOCAL_ANALYZER_API_KEY', '').strip()
    or os.environ.get('OPENAI_API_KEY', '').strip()
)
if not API_KEY:
    raise RuntimeError(
        'Missing Cerebras API key on Kaggle. Set CEREBRAS_API_KEY as a notebook secret '
        'or export it before launching make kaggle-duck.'
    )


def request_json(url: str, payload: dict, headers: dict[str, str], timeout: int = 60) -> dict:
    data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_KEY}',
}
smoke_payload = {
    'model': MODEL_ID,
    'messages': [{'role': 'user', 'content': 'Answer in one short sentence: what is 2 + 2?'}],
    'temperature': 0.0,
    'max_tokens': 64,
}
smoke_response = request_json(
    f'{BASE_URL.rstrip("/")}/chat/completions',
    smoke_payload,
    headers=headers,
    timeout=120,
)
generated = smoke_response.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
print('\n' + '=' * 88, flush=True)
print(f'REMOTE ANALYZER SMOKE TEST ({PROVIDER})', flush=True)
print('Model:', MODEL_ID, flush=True)
print('Generated:', generated, flush=True)
print('=' * 88 + '\n', flush=True)

setup_env = {
    'LOCAL_ANALYZER_BASE_URL': BASE_URL,
    'OPENAI_BASE_URL': BASE_URL,
    'LOCAL_ANALYZER_PROVIDER': PROVIDER,
    'OPENAI_PROVIDER': PROVIDER,
    'LOCAL_ANALYZER_MODEL_ID': MODEL_ID,
    'INFERENCE_ANALYZER_MODEL': MODEL_ID,
    'LOCAL_ANALYZER_API_KEY': API_KEY,
    'OPENAI_API_KEY': API_KEY,
    'LOCAL_ANALYZER_APP_NAME': __LOCAL_ANALYZER_APP_NAME__,
    'LOCAL_ANALYZER_CONTEXT_WINDOW': str(__LOCAL_ANALYZER_CONTEXT_WINDOW__),
    'LOCAL_ANALYZER_MAX_OUTPUT': __LOCAL_ANALYZER_MAX_OUTPUT__,
    'LOCAL_ANALYZER_TOOL_STEPS': __LOCAL_ANALYZER_TOOL_STEPS__,
    'LOCAL_ANALYZER_TOOL_TIMEOUT': __LOCAL_ANALYZER_TOOL_TIMEOUT__,
    'LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS': __LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS__,
    'LOCAL_ANALYZER_YIELD_SECONDS': __LOCAL_ANALYZER_YIELD_SECONDS__,
    'LOCAL_ANALYZER_TEMPERATURE': __LOCAL_ANALYZER_TEMPERATURE__,
    'LOCAL_ANALYZER_TOP_P': __LOCAL_ANALYZER_TOP_P__,
    'LOCAL_ANALYZER_TOP_K': __LOCAL_ANALYZER_TOP_K__,
    'LOCAL_ANALYZER_ENABLE_THINKING': __LOCAL_ANALYZER_ENABLE_THINKING__,
    'MULTIMODAL_CONTEXT': __MULTIMODAL_CONTEXT__,
    'MULTIMODAL_UPSCALE': __MULTIMODAL_UPSCALE__,
    'DUCK_DEAD_ACTION_HINTS': __DUCK_DEAD_ACTION_HINTS__,
    'DUCK_LEVEL_CARRYOVER': __DUCK_LEVEL_CARRYOVER__,
    'DUCK_STATIC_REGION_HINTS': __DUCK_STATIC_REGION_HINTS__,
}
setup_env_path = Path(os.environ['TAAF_KAGGLE_SETUP_ENV'])
existing_setup_env = {}
if setup_env_path.exists():
    existing_setup_env = json.loads(setup_env_path.read_text(encoding='utf-8'))
    if not isinstance(existing_setup_env, dict):
        raise RuntimeError('TAAF_KAGGLE_SETUP_ENV must contain a JSON object.')
existing_setup_env.update(setup_env)
setup_env_path.write_text(json.dumps(existing_setup_env, indent=2), encoding='utf-8')
"""

_DUCK_VLLM_TEARDOWN_SCRIPT = r"""import os
import shutil
import signal
import time
from pathlib import Path

WORKING_DIR = Path(os.environ['TAAF_KAGGLE_WORKING_DIR'])
pid_path = WORKING_DIR / 'vllm-openai-server.pid'
site_packages = WORKING_DIR / 'vllm-site-packages'
if pid_path.exists():
    try:
        pid = int(pid_path.read_text(encoding='utf-8').strip())
        print('Stopping vLLM server', flush=True)
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(1)
        else:
            os.kill(pid, signal.SIGKILL)
    except Exception as exc:
        print(f'Could not stop vLLM server cleanly: {exc!r}', flush=True)
    pid_path.unlink(missing_ok=True)
shutil.rmtree(site_packages, ignore_errors=True)
print(f'Removed temporary vLLM install at {site_packages}', flush=True)
"""
