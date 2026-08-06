#!/usr/bin/env bash
#
# Regenerate every table and figure in
#   "Temporal Conditioning for Human Behavioral Cloning in Chess" (IEEE GEM 2026)
#
# Stages are strictly sequential and each one is independently runnable:
#
#   scripts/run_paper_pipeline.sh --list
#   scripts/run_paper_pipeline.sh --only tables
#   scripts/run_paper_pipeline.sh                      # everything, in order
#
# The GPU stages must not overlap -- two concurrent CUDA jobs will exhaust a
# 16 GB card, and on WSL they can take the whole session down. The Stockfish
# stages (quality, counterfactual) run for hours; everything else is minutes.
#
# See REPRODUCE.md for which stage produces which table or figure.

set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-python}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
EVAL_SET="${EVAL_SET:-data/testset_2026-05}"
OUT="${OUT:-results}"
METRICS="$OUT/metrics"
TABLES="$OUT/tables"
FIGURES="$OUT/figures"
STOCKFISH="${STOCKFISH:-stockfish}"
ENGINE_WORKERS="${ENGINE_WORKERS:-12}"
ALLIE_SET="${ALLIE_SET:-data/allie_testset/allie_test_set.parquet}"
ONLY=""

usage() {
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
    echo
    echo "Options:"
    echo "  --only <stage>   run one stage (see --list)"
    echo "  --list           list stages"
    echo
    echo "Environment overrides:"
    echo "  PY, CKPT_DIR, EVAL_SET, OUT, STOCKFISH, ENGINE_WORKERS, ALLIE_SET"
    exit 0
}

STAGES=(tables timehead rows bands comparison ladder maia3repro seedsweep quality counterfactual figures)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) ONLY="$2"; shift 2 ;;
        --list) printf '%s\n' "${STAGES[@]}"; exit 0 ;;
        -h|--help) usage ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

COND="$CKPT_DIR/marvin_23m_cond_1M.pt"
BLIND="$CKPT_DIR/marvin_23m_blind_1M.pt"
C79="$CKPT_DIR/marvin_79m_cond.pt"

run_stage() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }
banner() { echo; echo "=== [$(date +%H:%M:%S)] $1"; }

mkdir -p "$OUT" "$METRICS" "$TABLES" "$FIGURES"

# --- Table I (top block) and Table II -------------------------------------
if run_stage tables; then
    banner "tables: accuracy by bucket, behavioural heads, paired deltas"
    $PY -m evaluation.run --models cond="$COND" agn="$BLIND" \
        --eval-set "$EVAL_SET" --paired-with agn --out "$METRICS"
    $PY -m evaluation.tables \
        --metrics "$METRICS/metrics_$(basename "$EVAL_SET").json" \
        --cond cond --agnostic agn --out "$TABLES"
fi

# --- Table III and the move-conditional think-time comparison -------------
if run_stage timehead; then
    banner "timehead: think time by bucket, move-conditionality"
    $PY -m evaluation.time_head --models cond="$COND" agn="$BLIND" \
        --eval-set "$EVAL_SET" --out "$METRICS"
    $PY -m evaluation.time_head_move_conditional --model cond="$COND" \
        --eval-set "$EVAL_SET" --out "$METRICS/time_head_move_conditional.json"
fi

# --- Per-row model outputs (feed the bands, comparison and ladder stages) --
if run_stage rows; then
    banner "rows: per-position outputs for ours and for released Maia-3"
    $PY -m evaluation.policy_probe --model cond23="$COND" --eval-set "$EVAL_SET" \
        --transform true    --out "$OUT/rows_cond23_true_testset2026.parquet"
    $PY -m evaluation.policy_probe --model agn23="$BLIND" --eval-set "$EVAL_SET" \
        --transform true    --batch-size 192 \
        --out "$OUT/rows_agn23_true_testset2026.parquet"
    $PY -m evaluation.policy_probe --model cond23="$COND" --eval-set "$EVAL_SET" \
        --transform permute --out "$OUT/rows_cond23_permute_testset2026.parquet"
    $PY -m evaluation.policy_probe --model cond79="$C79" --eval-set "$EVAL_SET" \
        --transform true    --out "$OUT/rows_cond79_true_testset2026.parquet"
    $PY -m evaluation.policy_probe --model cond79="$C79" --eval-set "$EVAL_SET" \
        --transform permute --out "$OUT/rows_cond79_permute_testset2026.parquet"
    $PY -m evaluation.eval_maia3_official --model maia3-79m --eval-set "$EVAL_SET" \
        --batch-size 192 --out "$OUT/rows_maia3_79m_testset2026.parquet"
    $PY -m evaluation.eval_maia3_official --model maia3-23m --eval-set "$EVAL_SET" \
        --batch-size 256 --out "$OUT/rows_maia3_23m_testset2026.parquet"
fi

# --- Table I (lower block): paired delta by rating band -------------------
if run_stage bands; then
    banner "bands: paired delta by rating band x pressure bucket"
    $PY -m evaluation.rating_band_deltas \
        --cond  "$OUT/rows_cond23_true_testset2026.parquet" \
        --blind "$OUT/rows_agn23_true_testset2026.parquet" \
        --out "$METRICS"
fi

# --- Figure 2 and the system-comparison table -----------------------------
if run_stage comparison; then
    banner "comparison: ours vs released Maia-3, game-clustered CIs"
    $PY -m evaluation.plot_maia3_comparison \
        --rows "Maia-3 79M"="$OUT/rows_maia3_79m_testset2026.parquet" \
               "Marvin 79M"="$OUT/rows_cond79_true_testset2026.parquet" \
               "Marvin 79M (permuted clocks)"="$OUT/rows_cond79_permute_testset2026.parquet" \
               "Maia-3 23M"="$OUT/rows_maia3_23m_testset2026.parquet" \
               "Marvin 23M"="$OUT/rows_cond23_true_testset2026.parquet" \
               "Marvin 23M time-blind"="$OUT/rows_agn23_true_testset2026.parquet" \
        --pairs "Marvin 79M-Maia-3 79M" \
                "Marvin 79M-Marvin 79M (permuted clocks)" \
                "Marvin 23M-Maia-3 23M" \
                "Marvin 23M-Marvin 23M time-blind" \
        --fig-models "Maia-3 79M" "Marvin 79M" "Marvin 79M (permuted clocks)" \
        --out-dir "$METRICS" --tag maia3_comparison
fi

# --- Table IV: graded-fairness ladder -------------------------------------
if run_stage ladder; then
    banner "ladder: Maia-3 protocol -> blitz unfiltered -> full set"
    $PY -m evaluation.maia3_ladder \
        --baseline "$OUT/rows_maia3_79m_testset2026.parquet" \
        --ours     "$OUT/rows_cond79_true_testset2026.parquet" \
        --eval-set "$EVAL_SET" --tag ladder_79m --out-dir "$METRICS"
    $PY -m evaluation.maia3_ladder \
        --baseline "$OUT/rows_maia3_23m_testset2026.parquet" \
        --ours     "$OUT/rows_cond23_true_testset2026.parquet" \
        --eval-set "$EVAL_SET" --tag ladder_23m --out-dir "$METRICS"
fi

# --- Sanity anchor: reproduce Maia-3's own published 57.1% ----------------
if run_stage maia3repro; then
    banner "maia3repro: released Maia-3 under its own evaluation protocol"
    if [[ -f "$ALLIE_SET" ]]; then
        $PY -m evaluation.eval_maia3_official --model maia3-79m \
            --eval-set "$ALLIE_SET" --allie-filters \
            --out "$OUT/rows_maia3_79m_allie_filtered.parquet"
    else
        echo "  skipped: $ALLIE_SET not found (see REPRODUCE.md)"
    fi
fi

# --- Table I: seed sweep ---------------------------------------------------
if run_stage seedsweep; then
    banner "seedsweep: three matched seed pairs at 250k steps"
    $PY -m evaluation.seed_sweep \
        --pairs s67="$CKPT_DIR/marvin_23m_cond_250k_s67.pt,$CKPT_DIR/marvin_23m_blind_250k_s67.pt" \
                s68="$CKPT_DIR/marvin_23m_cond_250k_s68.pt,$CKPT_DIR/marvin_23m_blind_250k_s68.pt" \
                s69="$CKPT_DIR/marvin_23m_cond_250k_s69.pt,$CKPT_DIR/marvin_23m_blind_250k_s69.pt" \
        --eval-set "$EVAL_SET" --out "$METRICS/seed_sweep"
fi

# --- Figure 1a: sampled move quality vs Stockfish (HOURS) -----------------
if run_stage quality; then
    banner "quality: sampled-move CPL against Stockfish (this takes hours)"
    $PY -m evaluation.move_quality --models cond="$COND" agn="$BLIND" \
        --eval-set "$EVAL_SET" --tc 0,1 --per-cell-cap 4000 \
        --temperature 1.0 --depth 12 --limit 2000000 \
        --engine-path "$STOCKFISH" --engine-workers "$ENGINE_WORKERS" \
        --out "$OUT/cpl_sampled_tcbal_big.parquet"
fi

# --- Figure 1b and the dose-response sweep (HOURS) ------------------------
if run_stage counterfactual; then
    banner "counterfactual: clock/tempo injection on fixed boards (hours)"
    $PY -m evaluation.counterfactual_tempo --models cond="$COND" agn="$BLIND" \
        --eval-set "$EVAL_SET" --depth 12 \
        --engine-path "$STOCKFISH" --engine-workers "$ENGINE_WORKERS" \
        --out "$OUT/counterfactual_tempo.parquet"
    $PY -m evaluation.counterfactual --models cond="$COND" agn="$BLIND" \
        --eval-set "$EVAL_SET" --fracs 1.0,0.5,0.2,0.05,0.02 --limit 20000 \
        --depth 12 --engine-path "$STOCKFISH" --engine-workers "$ENGINE_WORKERS" \
        --out "$OUT/counterfactual_final.parquet"
fi

# --- Camera-ready figures --------------------------------------------------
if run_stage figures; then
    banner "figures: quality.png and comparison.png"
    $PY -m evaluation.paper_figures --results-dir "$OUT" --comparison-json "$METRICS/maia3_comparison.json" \
        --out-dir "$FIGURES"
fi

banner "pipeline complete -> $OUT"
