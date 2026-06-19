"""Explicit-policy entry point (6-Specialists, no-KNN wrapper).

Reproduces the paper's PRIS **Explicit-policy** variant. Thin wrapper over the frozen
Explicit-policy launcher (`eval_pems_pris_shadowed_core6_explicit.py`): keeps the launcher
prompt/code intact and only overrides the tool pool to the 6 Specialists:
pgtft_c0, pgtft_cstd, gru_c0, graph_wavenet_cstd, agcrn_cstd, dcrnn_cstd.
(Companion of the Open-policy entry point `eval_pems_pris_shadowed_core6_open_no_knn.py`.)
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
if str(THIS) not in sys.path:
    sys.path.insert(0, str(THIS))

import eval_pems_pris_shadowed_core6_explicit as archive

LEARNED6_NO_KNN = [
    "pgtft_c0",
    "pgtft_cstd",
    "gru_c0",
    "graph_wavenet_cstd",
    "agcrn_cstd",
    "dcrnn_cstd",
]

archive.TOOLS_CORE7 = list(LEARNED6_NO_KNN)
archive.TOOLS = list(LEARNED6_NO_KNN)

if __name__ == "__main__":
    archive.main()
