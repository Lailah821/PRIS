"""
Phase 6: LLM 호출 추상화 클라이언트

planning.md Phase 6-2:
  Agent LLM: 오픈소스 로컬 모델 (로컬 GPU에서 실행, 무료)
    - Llama 3.3-70B (Meta, Community License)
    - Qwen2.5-72B  (Alibaba, Apache 2.0)
  Judge LLM: Claude API (평가 품질 우선)
  Prompt 템플릿
  재시도 로직

사용처:
  - phase_6_main.py: run_prediction() 에서 직접 사용 가능
  - phase_3_agent_llm.py: _call_llm_backend()가 llm_client.py를 탐색 →
      같은 디렉토리의 llm_client.py (phase_6_llm_client를 래핑)를 경유
  - phase_4_soft_score.py: call_judge()로 Judge LLM 호출

Agent LLM 백엔드 우선순위:
  1. 로컬 Hugging Face Transformers (GPU 필요, AGENT_LLM_BACKEND='transformers')
  2. Ollama REST API            (AGENT_LLM_BACKEND='ollama', OLLAMA_BASE_URL 필요)
  3. vLLM OpenAI 호환 API      (AGENT_LLM_BACKEND='vllm',   VLLM_BASE_URL 필요)
  4. Claude API fallback        (ANTHROPIC_API_KEY 환경변수)

Judge LLM:
  - Claude API (ANTHROPIC_API_KEY 환경변수 필수)
  - fallback: 로컬 Agent LLM 재사용 (품질 저하 가능, 경고 출력)
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Any, Dict, Optional


# ============================================================================
# 환경 변수 키
# ============================================================================

ENV_ANTHROPIC_KEY    = "ANTHROPIC_API_KEY"
ENV_AGENT_BACKEND    = "AGENT_LLM_BACKEND"     # 'transformers' | 'ollama' | 'vllm'
ENV_JUDGE_BACKEND    = "JUDGE_LLM_BACKEND"     # 'claude' | 'gemini' | 'gemma' | 'ollama' | 'vllm' | 'transformers'
ENV_JUDGE_MODEL      = "JUDGE_LLM_MODEL"       # 로컬 백엔드(ollama/vllm/transformers)에서 Judge 모델명 강제
ENV_GEMINI_KEY       = "GEMINI_API_KEY"
ENV_OLLAMA_BASE_URL  = "OLLAMA_BASE_URL"        # 기본: http://localhost:11434
ENV_VLLM_BASE_URL    = "VLLM_BASE_URL"          # 기본: http://localhost:8000

DEFAULT_OLLAMA_URL   = "http://localhost:11434"
DEFAULT_VLLM_URL     = "http://localhost:8000"

# Claude 모델 (Judge)
_JUDGE_MODEL_DEFAULT  = "claude-opus-4-6"
# Gemini 모델 (Judge) — cost-aware default. Pro can still be used for
# final/critical-sample audits via JUDGE_LLM_MODEL=gemini-2.5-pro.
_JUDGE_GEMINI_DEFAULT = "gemini-2.5-flash"
# Gemma 모델 (Judge, Ollama 경유)
_JUDGE_GEMMA_DEFAULT  = "gemma-4-31b-it"

# 재시도
_MAX_RETRIES    = 3
_RETRY_BASE_SEC = 1.5


# ============================================================================
# 공개 API
# ============================================================================

def call_agent(
    prompt: str,
    model_name: Optional[str] = None,
    max_retries: int = _MAX_RETRIES,
    enforce_json: bool = True,
) -> str:
    """
    Agent LLM 호출 (로컬 Llama/Qwen → Claude fallback).

    planning.md Phase 6-2:
      "Agent LLM: 오픈소스 로컬 모델 (로컬 GPU에서 실행, 무료)"

    Args:
        prompt    : phase_3_agent_llm.prepare_agent_prompt() 결과
        model_name: 모델 이름 (None이면 환경변수/기본값 사용)
        max_retries: 최대 재시도 횟수

    Returns:
        모델 응답 텍스트 (raw string)
    """
    # OpenAI gpt-* 모델 직접 dispatch
    if model_name and str(model_name).lower().startswith("gpt-"):
        return _call_openai(prompt, model_name, max_retries=max_retries, enforce_json=enforce_json)

    backend = os.environ.get(ENV_AGENT_BACKEND, '').lower()

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            if backend == 'transformers':
                return _call_transformers(prompt, model_name)
            elif backend == 'ollama':
                return _call_ollama(prompt, model_name, enforce_json=enforce_json)
            elif backend == 'vllm':
                return _call_vllm(prompt, model_name)
            else:
                # 자동 탐지: ollama → vllm → transformers → Claude
                return _auto_detect_agent(prompt, model_name, enforce_json=enforce_json)
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(_RETRY_BASE_SEC * (attempt + 1))

    # 모든 로컬 백엔드 실패 → Claude API fallback
    api_key = os.environ.get(ENV_ANTHROPIC_KEY)
    if api_key:
        warnings.warn(
            "로컬 Agent LLM 실패. Claude API로 fallback (품질 이슈 가능).",
            stacklevel=2,
        )
        return _call_claude(prompt, api_key, model_name or _JUDGE_MODEL_DEFAULT, max_retries)

    raise RuntimeError(
        f"Agent LLM 호출 실패: {last_err}\n"
        "  옵션 1: AGENT_LLM_BACKEND 환경변수 설정 (transformers/ollama/vllm)\n"
        "  옵션 2: ANTHROPIC_API_KEY 환경변수 설정 (Claude fallback)"
    )


def call_judge(
    prompt: str,
    model_name: Optional[str] = None,
    max_retries: int = _MAX_RETRIES,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Judge LLM 호출.

    JUDGE_LLM_BACKEND 환경변수로 백엔드 선택:
      'claude'      → Claude API (ANTHROPIC_API_KEY 필요)
      'gemini'      → Gemini API (GEMINI_API_KEY 필요)
      'gemma'       → Gemma 4 31B via Google API (GEMINI_API_KEY 필요, model=gemma-4-31b-it)
      'ollama'      → Ollama 로컬 (품질 저하 가능)
      'vllm'        → vLLM 로컬 (품질 저하 가능)
      'transformers'→ HuggingFace 직접 로딩 (품질 저하 가능)
      미설정        → 자동: Claude → Gemini → 로컬 순서

    Args:
        prompt    : phase_4_soft_score.prepare_soft_score_prompt() 결과
        model_name: 모델명 (None이면 각 백엔드 기본값 사용)
        max_retries: 최대 재시도 횟수
        max_tokens: 최대 토큰 수 (None이면 각 백엔드 기본값 사용)

    Returns:
        모델 응답 텍스트 (raw string)
    """
    judge_backend = os.environ.get(ENV_JUDGE_BACKEND, '').lower()

    if judge_backend == 'claude':
        api_key = os.environ.get(ENV_ANTHROPIC_KEY)
        if not api_key:
            raise RuntimeError("JUDGE_LLM_BACKEND=claude 이나 ANTHROPIC_API_KEY 없음")
        return _call_claude(prompt, api_key, model_name or _JUDGE_MODEL_DEFAULT, max_retries, max_tokens)

    elif judge_backend == 'gemini':
        # honor JUDGE_LLM_MODEL env override so users can pick gemini-2.5-pro vs
        # gemini-3-pro vs flash without code edits.
        chosen = model_name or os.environ.get(ENV_JUDGE_MODEL) or _JUDGE_GEMINI_DEFAULT
        return _call_gemini(prompt, chosen, max_retries)

    elif judge_backend == 'gemma':
        chosen = model_name or os.environ.get(ENV_JUDGE_MODEL) or _JUDGE_GEMMA_DEFAULT
        return _call_gemini(prompt, chosen, max_retries)

    elif judge_backend in ('ollama', 'vllm', 'transformers'):
        # Agent와 분리된 Judge 모델을 주입할 수 있도록 JUDGE_LLM_MODEL 우선 사용.
        # 미설정이면 기존 동작(backend별 기본 모델)을 유지.
        judge_local_model = model_name or os.environ.get(ENV_JUDGE_MODEL) or None
        warnings.warn(
            f"Judge LLM을 로컬 백엔드({judge_backend})로 실행 (품질 저하 가능). "
            f"model={judge_local_model or '<backend default>'}",
            stacklevel=2,
        )
        return call_agent(prompt, model_name=judge_local_model, max_retries=max_retries)

    else:
        # 자동 탐지: Claude → Gemini → 로컬
        return _auto_detect_judge(prompt, model_name, max_retries, max_tokens)


# ============================================================================
# 내부: Judge 자동 탐지
# ============================================================================

def _auto_detect_judge(prompt: str, model_name: Optional[str], max_retries: int, max_tokens: Optional[int] = None) -> str:
    """
    Judge 백엔드 자동 탐지: Claude → Gemini → 로컬 순서 시도.
    """
    # 1. Claude
    api_key = os.environ.get(ENV_ANTHROPIC_KEY)
    if api_key:
        return _call_claude(prompt, api_key, model_name or _JUDGE_MODEL_DEFAULT, max_retries, max_tokens)

    # 2. Gemini
    gemini_key = os.environ.get(ENV_GEMINI_KEY)
    if gemini_key:
        return _call_gemini(prompt, model_name, max_retries)

    # 3. 로컬 fallback
    warnings.warn(
        "ANTHROPIC_API_KEY / GEMINI_API_KEY 없음. 로컬 Agent LLM으로 Judge 역할 대행 (품질 저하).",
        stacklevel=3,
    )
    return call_agent(prompt, model_name=model_name, max_retries=max_retries)


# ============================================================================
# 내부: 로컬 백엔드
# ============================================================================

def _auto_detect_agent(prompt: str, model_name: Optional[str], *, enforce_json: bool = True) -> str:
    """
    백엔드 자동 탐지: ollama → vllm → transformers 순서 시도.
    """
    # Ollama 포트 확인
    try:
        return _call_ollama(prompt, model_name, enforce_json=enforce_json)
    except Exception:
        pass

    # vLLM 포트 확인
    try:
        return _call_vllm(prompt, model_name)
    except Exception:
        pass

    # Transformers 직접 로딩 (느림)
    return _call_transformers(prompt, model_name)


def _call_ollama(prompt: str, model_name: Optional[str], *, enforce_json: bool = True) -> str:
    """
    Ollama REST API 호출.

    Ollama 설치 후 `ollama run llama3.3:70b` 또는 `ollama run qwen2.5:72b` 실행 필요.
    OLLAMA_BASE_URL 환경변수로 호스트 변경 가능.
    """
    import requests  # type: ignore

    base_url = os.environ.get(ENV_OLLAMA_BASE_URL, DEFAULT_OLLAMA_URL)
    _model = model_name or "llama3.3:70b"
    prompt_to_send = _maybe_disable_qwen3_thinking(prompt, _model)

    payload = {
        "model": _model,
        "prompt": prompt_to_send,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 1024, "num_ctx": 8192},
    }
    if _is_qwen3_model(_model):
        # Ollama exposes Qwen3 thinking separately. The /no_think prompt token is
        # not enough on this local build; without the API flag, long prompts can
        # spend the whole generation budget in `thinking` and return an empty
        # `response`.
        payload["think"] = False
    if enforce_json:
        # Qwen3's thinking mode interacts badly with Ollama's JSON formatter:
        # format=json often collapses valid prompts to "{}". The response field
        # is still parseable JSON without the formatter, so let our parser guard it.
        if not _is_qwen3_model(_model):
            payload["format"] = "json"
    resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    text = data.get("response") or ""
    if not text and _is_qwen3_model(_model):
        # Defensive fallback for older/newer Ollama variants that may still place
        # useful output in `thinking` despite think=False.
        thinking = str(data.get("thinking") or "")
        start = thinking.find("{")
        end = thinking.rfind("}")
        if 0 <= start < end:
            return thinking[start:end + 1]
    return text


def _maybe_disable_qwen3_thinking(prompt: str, model_name: str) -> str:
    """Qwen3 thinking mode can collapse Ollama JSON mode to '{}'; disable it."""
    if not _is_qwen3_model(model_name):
        return prompt
    if "/no_think" in prompt:
        return prompt
    return "/no_think\n" + prompt


def _is_qwen3_model(model_name: str) -> bool:
    return str(model_name).lower().startswith("qwen3")


def _call_vllm(prompt: str, model_name: Optional[str]) -> str:
    """
    vLLM OpenAI 호환 API 호출.

    vLLM 서버 예시:
      python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.3-70B-Instruct --port 8000
    VLLM_BASE_URL 환경변수로 호스트 변경 가능.
    """
    import requests  # type: ignore

    base_url = os.environ.get(ENV_VLLM_BASE_URL, DEFAULT_VLLM_URL)
    _model = model_name or "meta-llama/Llama-3.3-70B-Instruct"

    payload = {
        "model": _model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1024,
    }
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=120,
        headers={"Authorization": "Bearer dummy"},  # vLLM은 키 불필요
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_transformers(prompt: str, model_name: Optional[str]) -> str:
    """
    HuggingFace Transformers 직접 로딩 호출 (GPU 필요).

    처음 호출 시 모델 다운로드 및 로딩 시간 발생.
    이후 프로세스 종료 전까지 메모리 유지.

    planning.md:
      "로컬 GPU에서 실행 → 무료, fine-tuning도 직접 가능"
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
    import torch  # type: ignore

    _model_id = model_name or "meta-llama/Llama-3.3-70B-Instruct"

    # 전역 캐시 (프로세스 내 재사용)
    if not hasattr(_call_transformers, '_cache'):
        _call_transformers._cache = {}

    if _model_id not in _call_transformers._cache:
        tokenizer = AutoTokenizer.from_pretrained(_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            _model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        _call_transformers._cache[_model_id] = (tokenizer, model)

    tokenizer, model = _call_transformers._cache[_model_id]
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.2,
            do_sample=True,
        )
    return tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ============================================================================
# 내부: Gemini API
# ============================================================================

def _call_gemini(
    prompt: str,
    model_name: Optional[str],
    max_retries: int = _MAX_RETRIES,
) -> str:
    """
    Google Gemini API 호출 (google-genai 패키지).

    무료 API 키 발급: https://aistudio.google.com
    GEMINI_API_KEY 환경변수 필요.

    pip install google-genai
    """
    from google import genai  # type: ignore

    api_key = os.environ.get(ENV_GEMINI_KEY)
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY 환경변수 없음")

    _model = model_name or _JUDGE_GEMINI_DEFAULT
    client = genai.Client(api_key=api_key)

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=_model,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(_RETRY_BASE_SEC * (attempt + 1))

    raise RuntimeError(f"Gemini API 호출 실패 ({max_retries}회): {last_err}")


# ============================================================================
# 내부: OpenAI API (gpt-4o / gpt-4o-mini 등)
# ============================================================================

def _call_openai(
    prompt: str,
    model_name: str,
    max_retries: int = _MAX_RETRIES,
    enforce_json: bool = True,
) -> str:
    """OpenAI Chat Completions API 호출 (gpt-4o, gpt-4o-mini 등).

    OPENAI_API_KEY 환경변수 필요 (ask-gpt 와 동일 키 공유).
    """
    from openai import OpenAI  # type: ignore

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수 없음 (gpt-* 모델 호출 불가)")

    client = OpenAI(api_key=api_key)
    last_err: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            kwargs: Dict[str, Any] = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
            if enforce_json:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(_RETRY_BASE_SEC * (attempt + 1))

    raise RuntimeError(f"OpenAI API 호출 실패 ({max_retries}회): {last_err}")


# ============================================================================
# 내부: Claude API
# ============================================================================

def _call_claude(
    prompt: str,
    api_key: str,
    model: str,
    max_retries: int,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Anthropic Claude API 호출 (재시도 포함).
    """
    import anthropic  # type: ignore

    client = anthropic.Anthropic(api_key=api_key)
    last_err: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens or 1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(_RETRY_BASE_SEC * (attempt + 1))

    raise RuntimeError(f"Claude API 호출 실패 ({max_retries}회): {last_err}")
