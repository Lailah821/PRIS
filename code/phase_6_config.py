"""
Phase 6: 전역 파라미터 설정

planning.md Phase 6-2:
  α, β, γ (K-Score 가중치)
  λ (유사도 비율)
  임계점 (80%, 90%)
  최대 step, PGTFT 호출 수
  Real-Time vs Quality 설정값

사용처:
  - phase_6_main.py: run_prediction() 기본값
  - phase_2_scoring.py: score_all_sequences(alpha, beta, gamma)
  - phase_2_similarity.py: compute_all_similarities(lam)
  - phase_4_judge_llm.py: budget_max_steps, budget_max_pgtft
  - phase_3_agent_llm.py: model_name 선택
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional


# ============================================================================
# K-Score 가중치
# ============================================================================

ALPHA: float = 0.33   # (1 - U(s)) 가중치 — 불확실도 낮을수록 좋음
BETA:  float = 0.33   # C(s) 가중치 — 연결성 높을수록 좋음
GAMMA: float = 0.34   # P(s) 가중치 — 타겟 진척도 높을수록 좋음


# ============================================================================
# 유사도 파라미터
# ============================================================================

LAMBDA: float = 0.5   # Sim_G, Sim_SVI에서 lag-matrix vs strength-matrix 비율


# ============================================================================
# 임계값 (생성률 판단)
# ============================================================================

THRESHOLD_REALTIME: float = 0.80   # Real-Time mode: 80% 이상 → 종료
THRESHOLD_QUALITY:  float = 0.90   # Quality mode:   90% 이상 → 종료


# ============================================================================
# 반복 예산
# ============================================================================

MAX_ITERATIONS_REALTIME: int = 1    # Real-Time: 최대 1회 반복
MAX_ITERATIONS_QUALITY:  int = 10   # Quality: 최대 10회 반복 (실무상 초과 드뭄)

MAX_PGTFT_CALLS: int = 30           # 전체 세션 최대 PGTFT 호출 수
MAX_STEPS:       int = 20           # 시퀀스 경로 최대 step 수


# ============================================================================
# Top-K 시퀀스 수
# ============================================================================

TOP_K_REALTIME: int = 3    # Real-Time: K=3
TOP_K_QUALITY:  int = 6    # Quality: K=6 (경험에 따라 최대 10까지 조정)


# ============================================================================
# 탐색 반경
# ============================================================================

RADIUS_KM_REALTIME: float = 0.5    # Real-Time: 0.5km 반경
RADIUS_KM_QUALITY:  float = 10.0   # Quality: 도로 수집 최대 반경 (시퀀스 생성은 3km부터 0.5km씩 확장)


# ============================================================================
# 혼잡도 타입 (도로/보행/PM 중 예측 대상 1개 선택)
# ============================================================================

CONGESTION_TYPE: str = 'road_congestion'
# 선택 가능 값:
#   'road_congestion' — 차량 혼잡도  (기본값)
#   'ped_congestion'  — 보행자 혼잡도
#   'pm_congestion'   — PM(미세먼지) 혼잡도


# ============================================================================
# PGTFT 입력/예측 윈도우
# ============================================================================

import os as _os

# 1분 단위 데이터 기준 (input_len = 60*6, forecast_len = 60*1, step = 30)
INPUT_WINDOW: int     = 360   # 과거 6시간 입력
# h=60 / h=120 두 ckpt 공존을 위해 env var로 동적 설정. validation.common.config와 동일 변수.
FORECAST_HORIZON: int = int(_os.environ.get("VALIDATION_FORECAST_HORIZON", "60"))
STEP_SIZE: int        = 30    # 슬라이딩 윈도우 이동 간격 (학습 데이터 생성용)

# 추론 시 사용할 예측 스텝 인덱스
# 모델은 FORECAST_HORIZON(=60)개 스텝을 한꺼번에 출력하므로,
# 어느 시점 값을 최종 예측값으로 사용할지 선택
#   0  → 1분 후 (동시간대 근사)  ← 기본값
#   29 → 30분 후
#   59 → 60분(1시간) 후
# ★ 재학습 없이 이 값만 바꾸면 원하는 예측 시점으로 변경 가능
FORECAST_TARGET_STEP: int = 0   # 기본: 1분 후 (동시간대 예측)


# ============================================================================
# Adaptive shadow expansion (타겟 시퀀스 shadow 슬롯 동적 확장)
# ============================================================================
#
# 설계 근거:
#   타겟 시퀀스 입력 6채널 중 valid 5 + shadow 1 구성은 실제 데이터 5채널이
#   PGTFT 출력을 지배하여, 중간 체인에서 예측한 shadow 값의 변동이 타겟
#   예측값에 반영되지 못하고 iter마다 거의 동일한 값이 나오는 현상이 발생.
#
# 동작:
#   최근 N(window) iter의 타겟 예측값 변동폭이 threshold 미만이면 다음 iter부터
#   타겟 시퀀스의 shadow 슬롯을 reserve개로 확장 (valid 6-reserve + shadow reserve).
#   한번 trigger되면 세션 내 유지.
ADAPTIVE_SHADOW_WINDOW: int      = 3      # 변동폭 측정 iter 수
ADAPTIVE_SHADOW_THRESHOLD: float = 0.1    # 변동폭 임계값 (이 미만이면 trigger)
ADAPTIVE_SHADOW_RESERVE: int     = 3      # trigger 시 타겟 시퀀스 shadow 슬롯 수


# ============================================================================
# Agent LLM 모델 (planning.md Phase 6-2 & 주의사항)
# ============================================================================

# 비교 실험 대상 2종 (로컬 GPU, 무료)
AGENT_MODEL_LLAMA:  str = "meta-llama/Llama-3.3-70B-Instruct"
AGENT_MODEL_QWEN:   str = "Qwen/Qwen2.5-72B-Instruct"
AGENT_MODEL_QWEN_7B: str = "qwen2.5:7b"   # Ollama baseline, fast but weak
AGENT_MODEL_QWEN_14B: str = "qwen2.5:14b" # Ollama mid-size candidate
AGENT_MODEL_QWEN_32B: str = "qwen2.5:32b" # Ollama stronger local candidate

# 2026 recommendation track: Qwen3 has hybrid thinking / non-thinking modes.
# Qwen3 dense sizes are 8B/14B/32B, so keep Qwen2.5:7b only as the cheapest
# legacy baseline when a literal 7B comparison is needed.
AGENT_MODEL_QWEN3_8B:  str = "qwen3:8b"    # fast action ablation
AGENT_MODEL_QWEN3_14B: str = "qwen3:14b"   # recommended A0-A4 action model
AGENT_MODEL_QWEN3_32B: str = "qwen3:32b"   # recommended local reward supervisor
ACTION_LLM_RECOMMENDED: str = AGENT_MODEL_QWEN3_14B
REWARD_LLM_RECOMMENDED_LOCAL: str = AGENT_MODEL_QWEN3_32B
AGENT_MODEL_DEFAULT: str = AGENT_MODEL_LLAMA

# Judge LLM: Claude API (평가 품질 우선)
JUDGE_MODEL: str = "claude-opus-4-6"

# Reward supervisor: rule is the safe default; set to 'hybrid' or 'llm' for
# Qwen/API-based feedback rewriting once the local model is available.
REWARD_SUPERVISOR_MODE: str = "rule"  # 'off' | 'rule' | 'hybrid' | 'llm'
REWARD_SUPERVISOR_MODEL: str = REWARD_LLM_RECOMMENDED_LOCAL


# ============================================================================
# PGTFT 모델 체크포인트 경로
# ============================================================================

from pathlib import Path as _Path

# 디폴트: 2026-04-27 재학습한 forecasting PGTFT (`train_forecasting_pgtft.py`).
#   - 데이터: recon_6th_ped_congestion.csv (2026-04-19 재생성)
#   - 세팅: Stage 2/3 baseline과 동일 (top_n=6, max_hop=5, hop uniform 1.0,
#           input 360 → forecast 60, batch=12, lr=1e-3, peak_weighted_quantile_loss)
#   - 포맷: dict {"model_state_dict", "epoch", "val_loss", "config": {...}}
#   - PGTFT.py `_load_pgtft_model`이 dict/raw 두 포맷 자동 감지하여 로드함.
#
# 레거시(2025-09-17) 옛 학습 ckpt (raw state_dict, num_heads=8) 도 동일 로더로 사용 가능.
# 옛 가중치로 돌리려면 override:
#   import phase_6_config
#   phase_6_config.PGTFT_CHECKPOINT_PATH = (
#       r"C:/Users/uaer/OneDrive/1. 왔다갔다/density_estimation/PGTFT/5m메인가중치들(중요)/"
#       r"best_model_peak_awared_test_best_fold_4_peak_weighted_quantile.pt"
#   )
PGTFT_CHECKPOINT_PATH: str = str(
    _Path(__file__).parent / "model_weights" /
    "forecasting_pgtft_6th_ped_h120_fold5_best.pt"
)


# ============================================================================
# PGTFT 학습 시 사용된 static feature 정규화 통계
# ============================================================================
#
# 배경:
#   PGTFT 학습 노트북에서 static_real(LENGTH/WIDTH)은 min-max 정규화 후 투입.
#   운영에서 length/width=0으로 넣으면 분포가 완전히 달라 출력이 입력 변동에
#   둔감해짐(format mismatch). 아래 stats는 학습 시 쓴 값 기준이며, PGTFT.py
#   predict/_predict_raw 내부에서 meta에 length_norms/width_norms가 없을 때
#   distance-기반 fallback에도 이 범위로 클램프한다.
#
# 출처: 학습 노트북 min/max (Shape_Leng → LENGTH, ROAD_BT → WIDTH)
TRAIN_LENGTH_MIN: float = 26.84
TRAIN_LENGTH_MAX: float = 84.22
TRAIN_WIDTH_MIN:  float = 4.0
TRAIN_WIDTH_MAX:  float = 19.0

# 학습 시 adj_matrix를 hop-기반 similarity(1/(1+|h_i-h_j|))로 구성했을 때의
# 기준 최대 hop. PGTFT 호출 시 hop이 meta에 없으면 [0,1,1,1,2,2,2] 가정으로
# fallback.
MAX_HOP: int = 5


# ============================================================================
# 편의 함수: mode 문자열 → 파라미터 dict
# ============================================================================

def get_mode_config(mode: str) -> Dict:
    """
    mode 문자열로 전체 파라미터 dict 반환.

    사용처:
      - phase_6_main.py: run_prediction() 에서 State 초기화 시
      - phase_6_main.py: run_judge(), score_state_sequences() 에 전달 시

    Returns:
        {
          threshold, max_iterations, top_k, radius_km,
          budget_max_steps, budget_max_pgtft,
          alpha, beta, gamma, lam,
          agent_model, judge_model,
          congestion_type, input_window, forecast_horizon, step_size,
        }
    """
    if mode == 'realtime':
        return {
            'threshold'        : THRESHOLD_REALTIME,
            'max_iterations'   : MAX_ITERATIONS_REALTIME,
            'top_k'            : TOP_K_REALTIME,
            'radius_km'        : RADIUS_KM_REALTIME,
            'budget_max_steps' : MAX_STEPS,
            'budget_max_pgtft' : MAX_PGTFT_CALLS,
            'alpha'            : ALPHA,
            'beta'             : BETA,
            'gamma'            : GAMMA,
            'lam'              : LAMBDA,
            'agent_model'      : AGENT_MODEL_DEFAULT,
            'judge_model'      : JUDGE_MODEL,
            'reward_supervisor_mode': REWARD_SUPERVISOR_MODE,
            'reward_supervisor_model': REWARD_SUPERVISOR_MODEL,
            'congestion_type'  : CONGESTION_TYPE,
            'input_window'     : INPUT_WINDOW,
            'forecast_horizon' : FORECAST_HORIZON,
            'step_size'        : STEP_SIZE,
            'adaptive_shadow_window'   : ADAPTIVE_SHADOW_WINDOW,
            'adaptive_shadow_threshold': ADAPTIVE_SHADOW_THRESHOLD,
            'adaptive_shadow_reserve'  : ADAPTIVE_SHADOW_RESERVE,
        }
    else:  # 'quality'
        return {
            'threshold'        : THRESHOLD_QUALITY,
            'max_iterations'   : MAX_ITERATIONS_QUALITY,
            'top_k'            : TOP_K_QUALITY,
            'radius_km'        : RADIUS_KM_QUALITY,
            'budget_max_steps' : MAX_STEPS,
            'budget_max_pgtft' : MAX_PGTFT_CALLS,
            'alpha'            : ALPHA,
            'beta'             : BETA,
            'gamma'            : GAMMA,
            'lam'              : LAMBDA,
            'agent_model'      : AGENT_MODEL_DEFAULT,
            'judge_model'      : JUDGE_MODEL,
            'reward_supervisor_mode': REWARD_SUPERVISOR_MODE,
            'reward_supervisor_model': REWARD_SUPERVISOR_MODEL,
            'congestion_type'  : CONGESTION_TYPE,
            'input_window'     : INPUT_WINDOW,
            'forecast_horizon' : FORECAST_HORIZON,
            'step_size'        : STEP_SIZE,
            'adaptive_shadow_window'   : ADAPTIVE_SHADOW_WINDOW,
            'adaptive_shadow_threshold': ADAPTIVE_SHADOW_THRESHOLD,
            'adaptive_shadow_reserve'  : ADAPTIVE_SHADOW_RESERVE,
        }
