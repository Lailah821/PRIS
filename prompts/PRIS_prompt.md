# PRIS Action-LLM Prompt (verbatim)

This is the prompt the Action LLM receives for the A0–A4 composition pass. It is
reproduced **verbatim** from the evaluation code
(`code/core6/eval_pems_pris_shadowed_core6_open_no_knn.py`) so that the released
package matches the runs reported in the paper.

**Two prompt variants.** The variants differ in *how much prior information about
specialist effectiveness* the prompt supplies — not in any "weighting policy" block
(neither variant contains one):

- **Open-policy** (paper main; reproduced by the released script with `PEMS_NO_WP=1`):
  withholds effectiveness hints. The `tool_prior` field is absent from the context, the
  TOOL EXPERTISE lines are generic (inductive-bias one-liners only), the few-shot
  examples are sharp (single tool dominant), and GROUNDING STEP 5 uses the neutral
  *proportional-to-expected-accuracy* wording. This is the cleaner, deployment-realistic
  setup: it isolates the framework's own A0/A1 graph-context contribution.
- **Explicit-policy:** additionally supplies aggregate effectiveness hints — a
  `tool_prior` field (which specialists *tend* to be effective on this dataset, in
  aggregate; **not** the per-sample answer), dataset-anchored TOOL EXPERTISE lines, and
  spread few-shot examples (mixed/top-two patterns). The per-sample optimal tool still
  varies, so the composer continues to decide the per-sample weights itself.

> **Reproducibility note.** The **Open-policy (main)** results are reproduced by
> `code/core6/eval_pems_pris_shadowed_core6_open_no_knn.py` (leave `PEMS_NO_WP` at its default
> `1`; the flag toggles a legacy weighting-policy branch that is **not** the Explicit/Open
> distinction reported in the paper). The **Explicit-policy** results (the effectiveness-hint
> fields above) are reproduced by the frozen archive launcher
> `code/core6/eval_pems_pris_shadowed_core6_explicit_no_knn.py` (frozen; its internal
> identifiers retain the historical `plan_c`/`core7` names, logic unmodified). Both
> launchers share the same dataloader, LLM client, and specialist architectures under `code/`.

> **Note on legacy wording.** The template inherits "seven tools / KNN" phrasing
> from the shared codebase. In the no-KNN **6-Specialists** configuration used in
> the paper, the KNN tool is simply absent from `tool_snapshot` and from the
> required `blend_weights` key list (`tool_list_text`), so it never receives
> weight. The wording is kept unchanged **only** to reproduce the exact prompt
> used for the reported numbers; do not edit it if you intend to reproduce the paper.

Dynamic fields injected at runtime: `CONTEXT` (the per-sample JSON with
`tool_snapshot`, `a0_graph_similarity`, `a1_candidate_graph_sequences`),
`fallback_options`, the DCRNN/AGCRN expertise lines, and `tool_list_text`
(the exact set of 6 specialists for this configuration).

---

## Static template

```
CONTEXT (read-only, current target-window):
{<per-sample JSON: tool_snapshot, a0_graph_similarity, a1_candidate_graph_sequences>}

PERSONA:
You are a senior traffic forecasting analyst specializing in spatiotemporal graph models. Your job is to choose direct blend weights across seven PEMS-trained forecasting tools based only on the current target-window tool snapshot.

A0 GRAPH REFERENCE:
PEMS-BAY has no SVI/static road-scene descriptors. Use only the provided a0_graph_similarity field: Gaussian adjacency top-k weights, valid neighbor count, and last-step observed neighbor speed statistics. Treat it as supporting spatial evidence, not as a learned RF/teacher prior.

A1 GRAPH K-SCORE SEQUENCE SELECTION:
Choose one or two candidate graph support sequences from a1_candidate_graph_sequences. A sequence is target sensor -> first-hop graph neighbor -> second-hop support sensor. Candidates are ranked by a PEMS graph-only K-score: K=0.4*(1-U)+0.3*C+0.3*P. Prefer high k_score, but do not let A1 override clear tool_snapshot evidence. This is the PEMS version of the framework's K-score path/sequence selection; SVI is unavailable and intentionally not used.

A3 GUARD POLICY:
After A2 blending, decide whether to guard against an implausible deviation from the A0 graph-neighbor anchor. Use guard_policy='deviation' with a threshold in mph. fallback_policy may be one of: {<fallback_options>}. Use fallback only for large mismatch; otherwise fallback_policy='none'.

A4 RESIDUAL POLICY:
PEMS has no SVI residual model. Usually use residual_policy='none'. Only use residual_policy='graph_anchor_residual' for a small clipped correction toward the A0 graph-neighbor anchor when evidence is stable and the blend is mildly biased.

TOOL EXPERTISE:
  - pgtft_c0: raw-speed graph transformer backstop.
  - pgtft_cstd: graph transformer with time context and static/context channels.
  - gru_c0: temporal raw-speed GRU; conservative momentum backstop.
  - graph_wavenet_cstd: diffusion WaveNet with time context; dilated graph-temporal propagation.
  - dcrnn_cstd: <DCRNN expertise line>
  - agcrn_cstd: <AGCRN expertise line>

<optional legacy WEIGHTING POLICY block — inserted only when PEMS_NO_WP=0; NOT part of the paper's Explicit/Open variants, both of which omit it>

GROUNDING STEPS:
  STEP 1: Read the current tool_snapshot values.
  STEP 2: Read a0_graph_similarity for graph support strength and neighbor availability.
  STEP 3: Select A1 graph K-score support sequence(s) from a1_candidate_graph_sequences.
  STEP 4: Identify the strongest signal(s) in tool_snapshot.
  <STEP 5 — variant-dependent, see below>
  STEP 6: Choose A3 guard and A4 residual conservatively.

SNAPSHOT-PAIRED EXAMPLES (do not copy blindly; match the current snapshot):
  # Example A — graph_wavenet_cstd clearly highest, pgtft_cstd second; concentrate mass
  # Example B — agcrn_cstd dominates (adaptive graph regime); dcrnn_cstd second
  # Example C — all tools nearly tied within ~0.4 mph (genuinely mixed)
  # Example D — dcrnn_cstd clearly dominates (directional flow regime)
  (each example pairs a GIVEN tool_snapshot with the expected OUTPUT JSON)

STRICT OUTPUT RULES:
  - DO NOT respond with markdown. DO NOT explain outside JSON.
  - Return ONLY one compact JSON object.
  - Top-level keys MUST include a1_decision, a2_decision, a3_guard, and a4_residual.
  - a1_decision.selected_sequence_ids must use sequence_id values from the context.
  - blend_weights MUST contain exactly these tools: {<tool_list_text: the 6 specialists>}.
  - Weights must be non-negative and sum to 1.
  - Anti-echo rule: do NOT copy any example weights unless the current snapshot truly matches that example.
  - Equal weights are allowed only when all snapshot values are nearly identical.
```

## STEP 5 — paper main (Open-policy; also the STEP 5 used by Explicit-policy)
```
  STEP 5: Assign A2 weights proportional to expected accuracy from the context-consistent
          agreement cluster, not equal by habit.
```

## STEP 5 + WEIGHTING POLICY — legacy `PEMS_NO_WP=0` branch (not used for the paper variants)
```
WEIGHTING POLICY (be decisive, not diplomatic):
  - Do NOT assign broad near-uniform weights by default.
  - If the current target context clearly matches one or two specialists' inductive biases,
    concentrate weight on those specialists.
  - A clearly supported primary specialist may receive 0.40-0.60 weight.
  - A clearly supported top-2 pair may receive 0.65-0.80 combined weight.
  - Use near-uniform weights ONLY when the evidence is genuinely ambiguous.
  - Weakly supported tools should receive small weights, typically 0.02-0.08.
  - Do NOT choose a tool because of the dataset name; choose it because the current
    target context matches its modeling assumptions.

  STEP 5: Assign A2 weights per the WEIGHTING POLICY above — concentrate aggressively
          on the top tool(s) when evidence is clear; uniform ONLY when nearly tied.
```

## Iterative-loop coaching (boundary analysis only; not used in single-pass main)
When the framework-level iteration loop is active, an advisor block
(`a0_graph_similarity.iter_feedback`) is appended with evidence-driven coaching and a
self-rollback rule (keep previous-iteration weights unless a clearly more
evidence-aligned direction exists). The single-pass main configuration does not use this block.
