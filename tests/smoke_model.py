"""Smoke test: the model builds, has the published parameter counts, and runs.

Run with ``python -m tests.smoke_model``. Kept dependency-free (no pytest) so CI
can run it directly after a CPU-only torch install.
"""

from __future__ import annotations

import torch

from model import (Maia3Net, CONFIG_MAIA3_23M, CONFIG_MAIA3_79M, NUM_TIME_BINS,
                   count_parameters)

# Unique parameter counts as reported in the paper. A naive sum over
# ``state_dict()`` is larger because the GAB weight is shared across all 8
# blocks and therefore counted 8 times; ``parameters()`` de-duplicates it.
EXPECTED_PARAMS = {
    "23m_conditioned": 23_315_934,
    "23m_time_blind": 23_249_117,   # no time-forfeit head: 66,817 fewer
    "79m_conditioned": 79_641_566,
}

BATCH = 2


def make_batch(batch: int = BATCH) -> dict[str, torch.Tensor]:
    """A dummy batch. The scalars layout is shared with dataset.py:

    0 elo_norm  1 opp_elo_norm  2 ply_norm  3 my_clock_s  4 opp_clock_s
    5 my_inc_s  6 opp_inc_s  7 hmc_norm  8 base_time_s
    """
    scalars = torch.tensor([
        [(1500 - 1900) / 700, (1700 - 1900) / 700, 0.5,
         60.0, 60.0, 2.0, 2.0, 0.1, 180.0],
        [(2000 - 1900) / 700, (1900 - 1900) / 700, 0.7,
         300.0, 280.0, 0.0, 0.0, 0.2, 600.0],
    ])[:batch]
    return {
        "board_history": torch.randint(0, 13, (batch, 8, 64)),
        "time_history": torch.rand(batch, 8) * 60,
        "rep_flags": torch.randint(0, 2, (batch, 8)).float(),
        "castling": torch.randint(0, 2, (batch, 4)).float(),
        "ep_mask": torch.zeros(batch, 64),
        "scalars": scalars,
        "legal_mask": torch.ones(batch, 4096, dtype=torch.bool),
    }


def check_params(label: str, config: dict) -> Maia3Net:
    model = Maia3Net(config)
    n = count_parameters(model)
    expected = EXPECTED_PARAMS[label]
    assert n == expected, f"{label}: {n:,} parameters, expected {expected:,}"
    print(f"  {label:16s} {n:,} parameters  OK")
    return model


def main() -> None:
    print("Parameter counts")
    model = check_params("23m_conditioned", dict(CONFIG_MAIA3_23M))

    blind = dict(CONFIG_MAIA3_23M)
    blind["no_time_conditioning"] = True
    check_params("23m_time_blind", blind)

    check_params("79m_conditioned", dict(CONFIG_MAIA3_79M))

    print("\nForward pass (23M conditioned)")
    model.eval()
    with torch.no_grad():
        out = model(make_batch())
    (move_logits, wdl_logits, resign_logit, flag_logit,
     time_baseline, time_from_logits, time_to_logits) = out

    assert move_logits.shape == (BATCH, 4096), move_logits.shape
    assert wdl_logits.shape == (BATCH, 3), wdl_logits.shape
    assert resign_logit.shape[0] == BATCH, resign_logit.shape
    assert flag_logit is not None, "conditioned model must carry the time-forfeit head"
    assert time_baseline.shape == (BATCH, NUM_TIME_BINS), time_baseline.shape
    assert time_from_logits.shape == (BATCH, 64, NUM_TIME_BINS), time_from_logits.shape
    assert time_to_logits.shape == (BATCH, 64, NUM_TIME_BINS), time_to_logits.shape
    assert torch.isfinite(move_logits).all(), "move logits contain NaN/Inf"
    print(f"  move={tuple(move_logits.shape)} wdl={tuple(wdl_logits.shape)} "
          f"time_baseline={tuple(time_baseline.shape)}  OK")

    print("\nTime-blind arm ignores its temporal inputs")
    blind_model = Maia3Net(blind).eval()
    # Same board in both batches — only the temporal inputs may differ, or the
    # comparison would confound position with clock.
    slow = make_batch()
    fast = {k: v.clone() for k, v in slow.items()}
    fast["scalars"][:, 3] = 1.0          # my_clock_s: comfortable -> nearly flagged
    fast["scalars"][:, 4] = 1.0          # opp_clock_s
    fast["time_history"] = torch.zeros_like(slow["time_history"])
    with torch.no_grad():
        a = blind_model(slow)[0]
        b = blind_model(fast)[0]
    assert torch.allclose(a, b), "time-blind model responded to a clock change"
    print("  identical policy under two different clocks  OK")

    print("\nsmoke_model: PASSED")


if __name__ == "__main__":
    main()
