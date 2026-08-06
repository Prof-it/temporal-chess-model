"""
Maia3 baseline reimplementation with our auxiliary heads grafted on.

Faithful reproduction of the CSSLab Maia3 backbone (github.com/CSSLab/maia3):
  - Encoder-only transformer with post-LayerNorm order
  - RMSNorm in blocks, LayerNorm at the trunk's final layer (matches upstream)
  - GELU activation, omitted Q/K/V biases
  - Geometric Attention Bias (GAB) with a shared weight across all blocks
  - Elo conditioning via interpolation between two embedding anchors at [0, 5000]
  - Bilinear from/to policy head with per-rank-8 promotion bias projection
  - 3-class WDL head over mean-pooled square tokens

Additions on top (to slot into the existing train.py loop and match model.py's API):
  - Our temporal conditioning (TimeConditioningEncoder / ContinuousTimeEncoder),
    prepended or broadcast-added before the encoder stack.
  - Our BinaryHead-based resign and flag heads.
  - Our move-conditional time head (baseline + per-from + per-to projections).
  - Our extra board-state planes (castling, en passant, repetition flags,
    per-position clocks) concatenated into the input projection, so the same
    dataset/batch dict can feed both models for a clean comparison.

Configs mirror CSSLab's `model_registry.py` BASE_SIZE_CONFIG + per-variant
overrides for the 3M/5M/23M/79M sizes.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

# =============================================================================
# Shared building blocks (inlined from the former Chessformer model.py when
# Maia3 became the canonical model). Constants, time-encoder family, and small
# utility modules used across the architecture.
# =============================================================================

NUM_POLICY_OUTPUTS = 4096


TIME_SECOND_CAP = 120


NUM_TIME_BINS = TIME_SECOND_CAP + 1


NUM_VALUE_CLASSES = 3


NUM_SQUARES = 64


class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x)). Better gradient flow than ReLU/SiLU."""
    def forward(self, x):
        return F.mish(x)


NUM_TIME_LOG_BINS = 16


LOG_BIN_TIME_MAX_SECONDS = 14400.0


_LOG_BIN_TIME_DENOM = math.log1p(LOG_BIN_TIME_MAX_SECONDS)


NUM_INC_CATEGORIES = 5


NUM_TIME_CONDITIONING_TOKENS = 6


def log_bin_time(seconds: torch.Tensor, num_bins: int = NUM_TIME_LOG_BINS) -> torch.Tensor:
    """
    Bin time values logarithmically. Returns bin indices in [0, num_bins-1].

    Covers 0 to LOG_BIN_TIME_MAX_SECONDS (4h) with denser bins at lower times.
    Uses log1p for smooth handling of zero.

    ONNX-compatible: uses torch.clamp with min/max tensors to avoid type mismatch.
    """
    log_time = torch.log1p(torch.clamp(seconds, min=0.0))
    # Normalize to [0, 1] range, then to bin indices.
    normalized = log_time / _LOG_BIN_TIME_DENOM
    bins = (torch.clamp(normalized, min=0.0, max=0.9999) * num_bins).long()
    return bins


def bin_increment(inc: torch.Tensor) -> torch.Tensor:
    """
    Bin increment values into 5 categories: 0, 1, 2, 5, 10+.
    Returns indices 0-4.
    
    ONNX-compatible: uses arithmetic instead of torch.tensor() in traced context.
    """
    # Categories: 0->0, 1->1, 2->2, 3-9->3, 10+->4
    # Use arithmetic masking for ONNX compatibility
    bins = torch.zeros_like(inc)
    bins = bins + (inc == 1).to(inc.dtype)  # 1 -> 1
    bins = bins + 2 * (inc == 2).to(inc.dtype)  # 2 -> 2
    bins = bins + 3 * ((inc >= 3) & (inc < 10)).to(inc.dtype)  # 3-9 -> 3
    bins = bins + 4 * (inc >= 10).to(inc.dtype)  # 10+ -> 4
    return bins.long()


class TimeConditioningEncoder(nn.Module):
    """
    Encode time-related context as 6 tokens prepended to the square sequence.

    Tokens:
    1. [BASE_TIME] - Log-binned initial game time control (per game, fixed)
    2. [MY_CLOCK] - Log-binned remaining time for active player
    3. [OPP_CLOCK] - Log-binned remaining time for opponent
    4. [INCREMENT] - Increment category (0, 1, 2, 5, 10+)
    5. [MY_LAST_TIME] - Log-binned time spent on my last move
    6. [OPP_LAST_TIME] - Log-binned time spent on opponent's last move

    Elo conditioning is handled separately via ConcatEloEncoder (added to square tokens).

    Baseline-for-comparison variant. The broadcast continuous-time variants
    (ContinuousTimeEncoder) consume the same six signals via a different mechanism.
    """
    def __init__(self, d_model: int, shrink_globals: bool = False):
        super().__init__()
        self.d_model = d_model
        self.shrink_globals = shrink_globals
        # When `shrink_globals` is on, the per-square state input is expected to
        # already carry my_clock/opp_clock/my_last/opp_last per history step,
        # so only base_time and increment remain truly global.
        self.num_tokens = 2 if shrink_globals else NUM_TIME_CONDITIONING_TOKENS

        # Preserve the legacy construction + re-init order when the flag is
        # off so the OFF path remains bit-identical to pre-change behavior.
        if shrink_globals:
            self.base_time_embedding = nn.Embedding(NUM_TIME_LOG_BINS, d_model)
            self.inc_embedding = nn.Embedding(NUM_INC_CATEGORIES, d_model)
            self.token_pos_embedding = nn.Embedding(self.num_tokens, d_model)
            embeddings_for_reinit = (
                self.base_time_embedding, self.inc_embedding, self.token_pos_embedding,
            )
        else:
            self.base_time_embedding = nn.Embedding(NUM_TIME_LOG_BINS, d_model)
            self.my_clock_embedding = nn.Embedding(NUM_TIME_LOG_BINS, d_model)
            self.opp_clock_embedding = nn.Embedding(NUM_TIME_LOG_BINS, d_model)
            self.inc_embedding = nn.Embedding(NUM_INC_CATEGORIES, d_model)
            self.my_last_time_embedding = nn.Embedding(NUM_TIME_LOG_BINS, d_model)
            self.opp_last_time_embedding = nn.Embedding(NUM_TIME_LOG_BINS, d_model)
            self.token_pos_embedding = nn.Embedding(self.num_tokens, d_model)
            embeddings_for_reinit = (
                self.base_time_embedding, self.my_clock_embedding,
                self.opp_clock_embedding, self.inc_embedding,
                self.my_last_time_embedding, self.opp_last_time_embedding,
                self.token_pos_embedding,
            )

        # Explicit init + _no_reinit on every embedding so the model-wide apply
        # pass skips them — keeps backbone init bit-identical when swapping
        # encoder variants. See EndpointEloEncoder for the full reasoning.
        for emb in embeddings_for_reinit:
            nn.init.normal_(emb.weight, std=0.02)
            emb._no_reinit = True

        self.register_buffer(
            "token_pos_ids",
            torch.arange(self.num_tokens, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        base_time: torch.Tensor,                          # (B,) initial game base time in seconds
        increment: torch.Tensor,                          # (B,) increment in seconds
        my_clock: Optional[torch.Tensor] = None,          # (B,) seconds remaining for active player
        opp_clock: Optional[torch.Tensor] = None,         # (B,) seconds remaining for opponent
        my_last_time: Optional[torch.Tensor] = None,      # (B,) seconds spent on my last move
        opp_last_time: Optional[torch.Tensor] = None,     # (B,) seconds spent on opponent's last move
    ) -> torch.Tensor:
        """
        Returns:
            (B, num_tokens, d_model) — 6 tokens normally, 2 with shrink_globals.
        """
        base_time_token = self.base_time_embedding(log_bin_time(base_time))
        inc_token = self.inc_embedding(bin_increment(increment))

        if self.shrink_globals:
            tokens = torch.stack([base_time_token, inc_token], dim=1)  # (B, 2, d_model)
        else:
            my_clock_token = self.my_clock_embedding(log_bin_time(my_clock))
            opp_clock_token = self.opp_clock_embedding(log_bin_time(opp_clock))
            my_last_token = self.my_last_time_embedding(log_bin_time(my_last_time))
            opp_last_token = self.opp_last_time_embedding(log_bin_time(opp_last_time))
            tokens = torch.stack([
                base_time_token, my_clock_token, opp_clock_token,
                inc_token, my_last_token, opp_last_token,
            ], dim=1)  # (B, 6, d_model)

        tokens = tokens + self.token_pos_embedding(self.token_pos_ids)
        return tokens


class TimeControlToken(nn.Module):
    """Single token for the (static, per-game) time control: base_time + increment.

    Unlike the dynamic clock/tempo signals — smooth ordinal axes where endpoint
    interpolation fits — the time control defines a *regime* (e.g. 3+0 vs 3+2),
    with a qualitative jump at increment 0 vs >0. A small MLP over
    ``[log1p(base)/D, log1p(inc)/D, 1{inc>0}]`` captures the joint regime and the
    zero-increment step. Being a shared continuous function (not a per-TC table),
    it generalizes to rare/unseen time controls via interpolation + parameter
    sharing.
    """
    _LOG_DENOM = math.log1p(LOG_BIN_TIME_MAX_SECONDS)

    def __init__(self, d_model: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        # Explicit init + _no_reinit so the model-wide apply pass skips these.
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
                m._no_reinit = True

    def forward(self, base_time: torch.Tensor, increment: torch.Tensor) -> torch.Tensor:
        b = torch.log1p(base_time.clamp(min=0.0)) / self._LOG_DENOM
        inc = torch.log1p(increment.clamp(min=0.0)) / self._LOG_DENOM
        z = (increment > 0).to(base_time.dtype)
        feat = torch.stack([b, inc, z], dim=-1)  # (B, 3)
        return self.net(feat)


class TempoHistoryToken(nn.Module):
    """Single token summarizing the recent move-time (tempo) history.

    Consumes the full H-ply ``time_history`` (raw seconds, newest-first,
    alternating opponent/active by parity) through a small MLP → d_model. This
    routes the *whole* tempo trajectory through one attention-accessible token
    instead of the weak per-position broadcast channels, and subsumes the
    my_last/opp_last per-move tokens. ``time_history`` already stores per-ply
    elapsed (= spent) seconds, so ``log1p`` of it is the tempo signal directly.
    """
    _LOG_DENOM = math.log1p(LOG_BIN_TIME_MAX_SECONDS)

    def __init__(self, d_model: int, history_len: int = 8, hidden: int = 32):
        super().__init__()
        self.history_len = history_len
        self.net = nn.Sequential(
            nn.Linear(history_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
                m._no_reinit = True

    def forward(self, time_history: torch.Tensor) -> torch.Tensor:
        # time_history: (B, H) raw seconds, newest-first.
        feat = torch.log1p(time_history.clamp(min=0.0)) / self._LOG_DENOM
        return self.net(feat)


class InterpTimeTokenEncoder(nn.Module):
    """Continuous (non-binned) time-conditioning tokens.

    Splits the signals by their nature:
      - The **dynamic** clock/tempo signals (my_clock, opp_clock, my_last,
        opp_last) are smooth ordinal axes → endpoint interpolation between two
        learned anchors by ``w = clamp(log1p(sec)/log1p(MAX), 0, 1)`` (mirrors
        EndpointEloEncoder). Unlike the 16-bin TimeConditioningEncoder this does
        NOT quantize, so 1s and 2s map to distinct tokens.
      - The **static** time control (base_time + increment) is a joint regime,
        not an ordinal axis → a single MLP token (TimeControlToken).

    Token count (not shrunk):
      - default: 5 — TC, my_clock, opp_clock, my_last, opp_last.
      - tempo_history_token: 4 — TC, my_clock, opp_clock, tempo (the my_last/opp_last
        per-move tokens are subsumed by a single token over the full time_history).
    With shrink_globals: 1 (just the time-control token).
    """
    _W_DENOM = math.log1p(LOG_BIN_TIME_MAX_SECONDS)

    def __init__(self, d_model: int, shrink_globals: bool = False,
                 tempo_history_token: bool = False):
        super().__init__()
        self.d_model = d_model
        self.shrink_globals = shrink_globals
        # Tempo history token only applies when not shrunk (shrink keeps just TC).
        self.tempo_history_token = bool(tempo_history_token) and not shrink_globals
        # Joint time-control token (always present).
        self.tc_token = TimeControlToken(d_model)
        # Dynamic, ordinal signals via endpoint interpolation. With the tempo
        # history token the per-move my_last/opp_last tokens are subsumed, so only
        # the two clocks are interpolated.
        if shrink_globals:
            self._interp_order = ()
        elif self.tempo_history_token:
            self._interp_order = ("my_clock", "opp_clock")
        else:
            self._interp_order = ("my_clock", "opp_clock", "my_last", "opp_last")
        self.tempo_token = TempoHistoryToken(d_model) if self.tempo_history_token else None
        self.num_tokens = 1 + len(self._interp_order) + (1 if self.tempo_history_token else 0)
        # Per-signal (low, high) anchors as bare Parameters — _init_weights only
        # re-inits Linear/Embedding, so these are safe (same as EndpointEloEncoder).
        self.low = nn.ParameterDict(
            {n: nn.Parameter(torch.randn(d_model) * 0.02) for n in self._interp_order}
        )
        self.high = nn.ParameterDict(
            {n: nn.Parameter(torch.randn(d_model) * 0.02) for n in self._interp_order}
        )
        self.token_pos_embedding = nn.Embedding(self.num_tokens, d_model)
        nn.init.normal_(self.token_pos_embedding.weight, std=0.02)
        self.token_pos_embedding._no_reinit = True
        self.register_buffer(
            "token_pos_ids", torch.arange(self.num_tokens, dtype=torch.long), persistent=False
        )

    def _interp(self, name: str, seconds: torch.Tensor) -> torch.Tensor:
        # seconds: (B,) raw seconds -> (B, d_model) interpolated token.
        w = (torch.log1p(seconds.clamp(min=0.0)) / self._W_DENOM).clamp(0.0, 1.0).unsqueeze(-1)
        return (1.0 - w) * self.low[name] + w * self.high[name]

    def forward(
        self,
        base_time: torch.Tensor,
        increment: torch.Tensor,
        my_clock: Optional[torch.Tensor] = None,
        opp_clock: Optional[torch.Tensor] = None,
        my_last_time: Optional[torch.Tensor] = None,
        opp_last_time: Optional[torch.Tensor] = None,
        time_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns (B, num_tokens, d_model). num_tokens: 1 (shrink) / 4 (tempo token) / 5."""
        values = {
            "my_clock": my_clock,
            "opp_clock": opp_clock,
            "my_last": my_last_time,
            "opp_last": opp_last_time,
        }
        toks = [self.tc_token(base_time, increment)]  # (B, d_model)
        toks += [self._interp(n, values[n]) for n in self._interp_order]
        if self.tempo_history_token:
            if time_history is None:
                raise ValueError("tempo_history_token=True requires time_history")
            toks.append(self.tempo_token(time_history))
        tokens = torch.stack(toks, dim=1)  # (B, num_tokens, d_model)
        tokens = tokens + self.token_pos_embedding(self.token_pos_ids)
        return tokens


class InterpTimeChannelEncoder(nn.Module):
    """Elo-style scalar time conditioning routed into per-square input channels.

    This is the channel-routed analogue of InterpTimeTokenEncoder, and the
    closest possible mirror of how Elo is conditioned (Maia3EloEncoder): each
    time signal is a continuous *scalar* that interpolates between two learned
    anchors, and — exactly like ``elo_per_side`` — the result is concatenated
    into the input encoder's per-square channels rather than prepended as
    sequence tokens. No O(L^2) sequence-length cost, no log-bin quantization.

    Signal set matches InterpTimeTokenEncoder so the two are directly
    comparable:
      - static time control (base_time + increment) -> TimeControlToken MLP
      - dynamic clocks (my_clock, opp_clock)         -> endpoint interpolation
      - optional tempo history (full time_history)   -> TempoHistoryToken MLP
    Each signal emits an ``emb_dim``-vector; all are concatenated.

    ``emb_dim`` is the knob a previous (too-small) attempt got wrong: with too
    few channels the time signal is swamped inside the input projection and the
    model collapses to ignoring it. Mirror Elo's dim_emb (128) or larger.
    """
    _W_DENOM = math.log1p(LOG_BIN_TIME_MAX_SECONDS)

    def __init__(
        self,
        emb_dim: int = 64,
        shrink_globals: bool = False,
        tempo_history_token: bool = False,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.shrink_globals = shrink_globals
        self.tempo_history_token = bool(tempo_history_token) and not shrink_globals
        # Joint time-control signal (always present), same MLP as the token variant.
        self.tc_token = TimeControlToken(emb_dim)
        # Dynamic, ordinal signals via endpoint interpolation. With the tempo
        # history token the per-move my_last/opp_last signals are subsumed.
        if shrink_globals:
            self._interp_order = ()
        elif self.tempo_history_token:
            self._interp_order = ("my_clock", "opp_clock")
        else:
            self._interp_order = ("my_clock", "opp_clock", "my_last", "opp_last")
        self.tempo_token = TempoHistoryToken(emb_dim) if self.tempo_history_token else None
        # Per-signal (low, high) anchors as bare Parameters (mirrors Maia3EloEncoder's
        # endpoint embeddings); _init_weights only re-inits Linear/Embedding so these
        # are safe.
        self.low = nn.ParameterDict(
            {n: nn.Parameter(torch.randn(emb_dim) * 0.02) for n in self._interp_order}
        )
        self.high = nn.ParameterDict(
            {n: nn.Parameter(torch.randn(emb_dim) * 0.02) for n in self._interp_order}
        )
        self.num_signals = 1 + len(self._interp_order) + (1 if self.tempo_history_token else 0)

    @property
    def out_dim(self) -> int:
        return self.num_signals * self.emb_dim

    def _interp(self, name: str, seconds: torch.Tensor) -> torch.Tensor:
        # seconds: (B,) raw seconds -> (B, emb_dim) interpolated channels.
        w = (torch.log1p(seconds.clamp(min=0.0)) / self._W_DENOM).clamp(0.0, 1.0).unsqueeze(-1)
        return (1.0 - w) * self.low[name] + w * self.high[name]

    def forward(
        self,
        base_time: torch.Tensor,
        increment: torch.Tensor,
        my_clock: Optional[torch.Tensor] = None,
        opp_clock: Optional[torch.Tensor] = None,
        my_last_time: Optional[torch.Tensor] = None,
        opp_last_time: Optional[torch.Tensor] = None,
        time_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns (B, num_signals * emb_dim) — concatenated time channels."""
        values = {
            "my_clock": my_clock,
            "opp_clock": opp_clock,
            "my_last": my_last_time,
            "opp_last": opp_last_time,
        }
        parts = [self.tc_token(base_time, increment)]  # (B, emb_dim)
        parts += [self._interp(n, values[n]) for n in self._interp_order]
        if self.tempo_history_token:
            if time_history is None:
                raise ValueError("tempo_history_token=True requires time_history")
            parts.append(self.tempo_token(time_history))
        return torch.cat(parts, dim=-1)  # (B, num_signals * emb_dim)


TIME_FEATURIZE_VARIANTS = ("log1p", "sqrt", "linear", "fourier")


FOURIER_NUM_FREQS = 4


FOURIER_MAX_PERIOD_S = float(LOG_BIN_TIME_MAX_SECONDS)


def _fourier_features(
    t: torch.Tensor,
    num_freqs: int = FOURIER_NUM_FREQS,
    max_period: float = FOURIER_MAX_PERIOD_S,
) -> torch.Tensor:
    """Sinusoidal features for continuous scalars (NeRF / diffusion-timestep style).

    For each input scalar, emit sin/cos at `num_freqs` log-spaced periods from
    1s up to `max_period`. Captures multi-scale temporal structure (fine
    tactical seconds AND coarse strategic minutes) without one learned
    compression baked in.

    Input:  (B, N) raw seconds
    Output: (B, N * num_freqs * 2)
    """
    B, N = t.shape
    log_periods = torch.linspace(
        0.0, math.log(max_period), num_freqs, device=t.device, dtype=t.dtype
    )
    omega = 2.0 * math.pi / torch.exp(log_periods)        # (num_freqs,)
    phases = t.unsqueeze(-1) * omega                       # (B, N, num_freqs)
    feat = torch.stack([phases.sin(), phases.cos()], dim=-1)  # (B, N, num_freqs, 2)
    return feat.reshape(B, N * num_freqs * 2)


class ContinuousTimeEncoder(nn.Module):
    """
    Fuse six continuous time signals into a single (B, d_model) vector that is
    broadcast-added to all 64 square tokens (mirrors ConcatEloEncoder).

    The MLP is fed featurized inputs; `featurize` is the ablation axis. Time
    inputs arrive as raw seconds; the encoder picks its own scale.

    featurize options:
      'log1p'   — log1p(s) / log1p(14400); time matters more at the low end
      'sqrt'    — sqrt(s)  / sqrt(14400);  intermediate compression
      'linear'  — s / 14400;                null hypothesis
      'fourier' — sin/cos at log-spaced frequencies; multi-scale primitive

    Per the plan: the final projection MUST be initialized with std>0, not zero.
    With broadcast-add, the gradient signal back into the time MLP is much
    weaker than for a sequence token; a zero-init head frequently collapses to
    "output ~0, model learns to ignore it" and the run looks like no_time.
    """
    NUM_TIME_INPUTS = 6  # base, inc, my_clock, opp_clock, my_last, opp_last (default)
    NUM_TIME_INPUTS_PER_POSITION_CLOCKS = 2  # base, inc (when per_position_clocks)
    FINAL_INIT_STD = 0.02

    def __init__(
        self,
        d_model: int,
        featurize: str = "log1p",
        hidden_dim: int = 128,
        shrink_globals: bool = False,
    ):
        super().__init__()
        if featurize not in TIME_FEATURIZE_VARIANTS:
            raise ValueError(
                f"Unknown time featurize: {featurize!r}. Known: {TIME_FEATURIZE_VARIANTS}"
            )
        self.d_model = d_model
        self.featurize = featurize
        self.shrink_globals = shrink_globals
        self.num_time_inputs = (
            self.NUM_TIME_INPUTS_PER_POSITION_CLOCKS if shrink_globals else self.NUM_TIME_INPUTS
        )
        self.feat_dim = self._compute_feat_dim(featurize, self.num_time_inputs)

        self.mlp = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )
        # Explicit init on BOTH layers + _no_reinit so the model-wide apply pass
        # in Chessformer skips them entirely. Two reasons in one: (1) preserve
        # the careful nonzero init on the final projection (dead-encoder guard),
        # (2) avoid apply consuming a variant-dependent number of RNG draws and
        # shifting the backbone init between encoder variants.
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[-1].weight, std=self.FINAL_INIT_STD)
        nn.init.zeros_(self.mlp[-1].bias)
        self.mlp[0]._no_reinit = True
        self.mlp[-1]._no_reinit = True

    @classmethod
    def _compute_feat_dim(cls, featurize: str, num_time_inputs: int) -> int:
        if featurize == "fourier":
            return num_time_inputs * FOURIER_NUM_FREQS * 2
        return num_time_inputs

    def _apply_featurize(self, raw: torch.Tensor) -> torch.Tensor:
        """raw: (B, num_time_inputs) raw seconds → (B, feat_dim) featurized."""
        t = raw.clamp(min=0)
        if self.featurize == "log1p":
            return torch.log1p(t) / _LOG_BIN_TIME_DENOM
        if self.featurize == "sqrt":
            return torch.sqrt(t) / math.sqrt(LOG_BIN_TIME_MAX_SECONDS)
        if self.featurize == "linear":
            return t / LOG_BIN_TIME_MAX_SECONDS
        if self.featurize == "fourier":
            return _fourier_features(t)
        raise AssertionError(f"unreachable: featurize={self.featurize}")

    def forward(
        self,
        base_time: torch.Tensor,
        increment: torch.Tensor,
        my_clock: Optional[torch.Tensor] = None,
        opp_clock: Optional[torch.Tensor] = None,
        my_last_time: Optional[torch.Tensor] = None,
        opp_last_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.shrink_globals:
            raw = torch.stack([base_time, increment], dim=-1)  # (B, 2)
        else:
            raw = torch.stack([
                base_time, increment, my_clock, opp_clock, my_last_time, opp_last_time,
            ], dim=-1)  # (B, 6)
        return self.mlp(self._apply_featurize(raw))


TIME_ENCODER_VARIANTS = ("tokens", "mlp")


def build_time_encoder(
    variant: str,
    d_model: int,
    featurize: str = "log1p",
    shrink_globals: bool = False,
    token_mode: str = "bins",
    tempo_history_token: bool = False,
) -> nn.Module:
    """Factory for time-conditioning encoders.

    The variant string is the single source of truth for which encoder is in
    play; `featurize` is a sub-axis only meaningful for the continuous (`mlp`)
    variant. Keep ablation runs hermetic by switching only these strings.

    When `shrink_globals` is True, the encoder accepts only the two
    globally-relevant time signals (base_time, increment) — clocks and last
    move times are expected to be carried elsewhere (e.g. per-history-step in
    InputEncoder via `per_position_clocks`).
    """
    if variant == "tokens":
        if token_mode == "interp":
            return InterpTimeTokenEncoder(
                d_model, shrink_globals=shrink_globals,
                tempo_history_token=tempo_history_token,
            )
        return TimeConditioningEncoder(d_model, shrink_globals=shrink_globals)
    if variant == "mlp":
        encoder = ContinuousTimeEncoder(
            d_model, featurize=featurize, shrink_globals=shrink_globals,
        )
    else:
        raise ValueError(
            f"Unknown time_encoder variant: {variant!r}. Known: {TIME_ENCODER_VARIANTS}"
        )

    # Init-discipline assertion for continuous-broadcast variants: the final
    # projection must have nonzero weight std, or it collapses (dead encoder).
    final_linear = encoder.mlp[-1]
    assert final_linear.weight.detach().std().item() > 1e-4, (
        f"{variant} time encoder's final projection has near-zero init std — "
        f"will likely die under broadcast-add. See ContinuousTimeEncoder docstring."
    )
    return encoder


def _build_with_rng_snapshot(builder_fn):
    """Run `builder_fn()` without letting it shift the global RNG state.

    Used to keep backbone init bit-identical across encoder variants that
    consume different numbers of random draws (e.g., 14-anchor vs 2-endpoint
    ELO, MLP vs Fourier time). Without this, swapping an encoder at the same
    seed silently re-rolls every later submodule's weights and contaminates
    the ablation.
    """
    rng_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    module = builder_fn()
    torch.set_rng_state(rng_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return module


class AttentionPooling(nn.Module):
    """
    Attention-based pooling that learns which squares matter for value/time prediction.

    Unlike mean pooling which weights all 64 squares equally, this learns to
    focus on tactically relevant squares (e.g., the hanging queen, not empty squares).
    
    Optionally conditioned on global context (ELO might focus on different features).
    """
    def __init__(self, d_model, context_dim=None):
        super().__init__()
        self.d_model = d_model
        self.use_context = context_dim is not None

        if self.use_context:
            # Query is generated from context
            self.query_proj = nn.Linear(context_dim, d_model)
        else:
            # Learned query vector
            self.query = nn.Parameter(torch.randn(d_model) * 0.02)

        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, x, context=None):
        """
        Args:
            x: (B, 64, d_model) - token representations
            context: (B, context_dim) - optional global context
        Returns:
            pooled: (B, d_model) - weighted combination of tokens
        """
        B = x.shape[0]
        keys = self.key_proj(x)  # (B, 64, d_model)

        if self.use_context and context is not None:
            query = self.query_proj(context)  # (B, d_model)
        else:
            query = self.query.unsqueeze(0).expand(B, -1)  # (B, d_model)

        # Compute attention scores
        attn_scores = (keys * query.unsqueeze(1)).sum(dim=-1) * self.scale  # (B, 64)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, 64)

        # Weighted sum
        pooled = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)
        return pooled


class BinaryHead(nn.Module):
    """
    Binary classification head (used for resign and flag prediction).

    Mean-pool over the 64 squares, then a small MLP:
    mean-pool -> LayerNorm -> Linear(d_model, 128) -> ReLU -> Linear(128, 1).
    Returns a single logit per sample, trained with BCE-with-logits.
    """
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 64, d_model)
        Returns:
            logit: (B,) binary logits
        """
        pooled = x.mean(dim=1)          # (B, d_model)
        return self.mlp(self.norm(pooled)).squeeze(-1)


# =============================================================================
# CONFIGS — mirror github.com/CSSLab/maia3 maia3/model_registry.py
# =============================================================================

BASE_MAIA3_CONFIG = {
    "history": 8,
    "num_blocks": 8,
    "mlp_ratio": 2.0,
    "dim_emb": 128,            # Elo embedding dim
    "dropout": 0.0,
    "use_gab": True,
    "use_rms_norm": True,
    "omit_qkv_biases": True,
    "activation": "gelu",
    # GAB Q/K-rotation ablation (default = additive-only baseline, Arm A):
    #   gab_additive_bias: GAB emits the additive B[i,j] bias (standard Maia3).
    #   gab_qk_rotation:   GAB also emits per-square, per-head 2D-plane rotation
    #     angles applied to q/k before attention — a dynamic, board-conditional
    #     generalization of RoPE: logit[i,j] = (R_i q_i)·(R_j k_j). Zero-init -> the
    #     rotation is identity at step 0. Arm B = both; Arm C = rotation only.
    "gab_additive_bias": True,
    "gab_qk_rotation": False,
    # Time/elo/extra-input knobs we add on top of upstream:
    "time_encoder": "tokens",
    # "interp" = continuous endpoint-interpolation time tokens (no log-bin
    # quantization; 1s vs 2s distinguishable). "bins" = legacy 16-bin embeddings.
    "time_token_encoder": "interp",
    "time_featurize": "log1p",
    # Default config routes the full 8-ply time history through one tempo token
    # and drops the per-position clock channels (see InterpTimeTokenEncoder).
    "tempo_history_token": True,
    "per_position_clocks": False,
    "shrink_global_time": True,
    # When True, the per-position channels carry per-move clock *deltas* (move
    # durations / tempo) instead of absolute remaining clocks. Opt-in via
    # --per-position-time-deltas; see _per_position_clock_deltas. Default False
    # so the baseline (absolute clocks) is unchanged.
    "per_position_time_deltas": False,
    "no_time_context": False,
    "num_piece_types": 13,     # our dataset uses 13 (12 pieces + empty)
    "gradient_checkpointing": False,
}

CONFIG_MAIA3_3M = {
    **BASE_MAIA3_CONFIG,
    "dim_vit": 192,
    "head_hid_dim": 192,
    "num_heads": 6,
    "gab_gen_size": 64,
    "gab_per_square_dim": 0,
    "gab_intermediate_dim": 64,
}

CONFIG_MAIA3_5M = {
    **BASE_MAIA3_CONFIG,
    "dim_vit": 256,
    "head_hid_dim": 256,
    "num_heads": 8,
    "gab_gen_size": 64,
    "gab_per_square_dim": 0,
    "gab_intermediate_dim": 64,
}

CONFIG_MAIA3_23M = {
    **BASE_MAIA3_CONFIG,
    "dim_vit": 512,
    "head_hid_dim": 512,
    "num_heads": 16,
    "gab_gen_size": 128,
    "gab_per_square_dim": 32,
    "gab_intermediate_dim": 128,
}

CONFIG_MAIA3_79M = {
    **BASE_MAIA3_CONFIG,
    "dim_vit": 1024,
    "head_hid_dim": 1024,
    "num_heads": 32,
    "gab_gen_size": 128,
    "gab_per_square_dim": 32,
    "gab_intermediate_dim": 128,
}


# =============================================================================
# Elo encoder — Maia3-faithful interpolation between two endpoint embeddings
# =============================================================================

class Maia3EloEncoder(nn.Module):
    """Linear interpolation between two learned embeddings at Elo=0 and Elo=5000.

    Mirrors `interpolate_elo` in the upstream MAIA3Model. Separate per-side
    anchor pairs for player vs opponent.

    Returns (B, 2 * dim_emb): the per-sample (player, opponent) concatenation
    that the input projection consumes.
    """
    ELO_MAX = 5000.0

    def __init__(self, dim_emb: int):
        super().__init__()
        self.dim_emb = dim_emb
        self.player_low = nn.Embedding(1, dim_emb)
        self.player_high = nn.Embedding(1, dim_emb)
        self.opp_low = nn.Embedding(1, dim_emb)
        self.opp_high = nn.Embedding(1, dim_emb)
        # Match upstream init — xavier_normal_ on each anchor weight.
        nn.init.xavier_normal_(self.player_low.weight)
        nn.init.xavier_normal_(self.player_high.weight)
        nn.init.xavier_normal_(self.opp_low.weight)
        nn.init.xavier_normal_(self.opp_high.weight)
        for emb in (self.player_low, self.player_high, self.opp_low, self.opp_high):
            emb._no_reinit = True

    def _interp(self, elo: torch.Tensor, low: nn.Embedding, high: nn.Embedding) -> torch.Tensor:
        elo = torch.clamp(elo, 0.0, self.ELO_MAX)
        w_low = (elo / self.ELO_MAX).unsqueeze(-1)
        w_high = 1.0 - w_low
        zero_idx = torch.zeros_like(elo, dtype=torch.long)
        e_low = low(zero_idx)
        e_high = high(zero_idx)
        return w_low * e_low + w_high * e_high

    def forward(self, player_elo: torch.Tensor, opp_elo: torch.Tensor) -> torch.Tensor:
        e_p = self._interp(player_elo, self.player_low, self.player_high)
        e_o = self._interp(opp_elo, self.opp_low, self.opp_high)
        return torch.cat([e_p, e_o], dim=-1)  # (B, 2 * dim_emb)


# =============================================================================
# Time conditioning ablations — three approaches that try to recover the
# bandwidth of the prepended-tokens variant without paying its O(L^2) cost.
#
# All three share the same input signals as TimeConditioningEncoder /
# ContinuousTimeEncoder: 6 inputs (base_time, increment, my_clock, opp_clock,
# my_last_time, opp_last_time) or 2 inputs (base_time, increment) when
# shrink_globals=True. Time is in seconds.
# =============================================================================

class _TimeFeaturizer(nn.Module):
    """Shared utility: featurize raw time inputs for MLP-style encoders.

    Mirrors ContinuousTimeEncoder's featurize options so the new variants can
    be compared on the same input transformation.
    """
    def __init__(self, featurize: str, num_time_inputs: int):
        super().__init__()
        if featurize not in TIME_FEATURIZE_VARIANTS:
            raise ValueError(f"Unknown featurize: {featurize!r}")
        self.featurize = featurize
        self.num_time_inputs = num_time_inputs
        if featurize == "fourier":
            self.feat_dim = num_time_inputs * FOURIER_NUM_FREQS * 2
        else:
            self.feat_dim = num_time_inputs

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        """raw: (B, num_time_inputs) raw seconds -> (B, feat_dim)."""
        t = raw.clamp(min=0)
        if self.featurize == "log1p":
            return torch.log1p(t) / _LOG_BIN_TIME_DENOM
        if self.featurize == "sqrt":
            return torch.sqrt(t) / math.sqrt(LOG_BIN_TIME_MAX_SECONDS)
        if self.featurize == "linear":
            return t / LOG_BIN_TIME_MAX_SECONDS
        if self.featurize == "fourier":
            return _fourier_features(t)
        raise AssertionError(f"unreachable: featurize={self.featurize}")


def _stack_time_inputs(
    base_time: torch.Tensor,
    increment: torch.Tensor,
    my_clock: Optional[torch.Tensor] = None,
    opp_clock: Optional[torch.Tensor] = None,
    my_last_time: Optional[torch.Tensor] = None,
    opp_last_time: Optional[torch.Tensor] = None,
    shrink_globals: bool = False,
) -> torch.Tensor:
    """Stack the time inputs into a (B, N) tensor. N=2 or N=6 depending on
    shrink_globals."""
    if shrink_globals:
        return torch.stack([base_time, increment], dim=-1)
    return torch.stack([
        base_time, increment, my_clock, opp_clock, my_last_time, opp_last_time,
    ], dim=-1)


class TimeChannelEncoder(nn.Module):
    """Per-signal log-binned embeddings, concatenated into per-square channels
    (option 1: preserves per-signal distinguishability without growing the
    sequence; still uniform across squares).

    Each time signal gets its own small Embedding(NUM_TIME_LOG_BINS,
    channel_dim) — the same discrete log-binned representation as the tokens
    variant, just routed into the input encoder's channel dim rather than
    prepended as sequence tokens. The 6 (or 2) per-signal embeddings are
    concatenated into a (B, N_signals * channel_dim) vector that the input
    encoder broadcasts onto each square and mixes through state_proj.

    Returns: (B, N_signals * channel_dim) — to be passed to Maia3InputEncoder.
    """
    def __init__(self, channel_dim: int = 16, shrink_globals: bool = False):
        super().__init__()
        self.channel_dim = channel_dim
        self.shrink_globals = shrink_globals
        self.num_time_inputs = 2 if shrink_globals else 6

        if shrink_globals:
            self.base_time_emb = nn.Embedding(NUM_TIME_LOG_BINS, channel_dim)
            self.inc_emb = nn.Embedding(NUM_INC_CATEGORIES, channel_dim)
            embs = (self.base_time_emb, self.inc_emb)
        else:
            self.base_time_emb = nn.Embedding(NUM_TIME_LOG_BINS, channel_dim)
            self.my_clock_emb = nn.Embedding(NUM_TIME_LOG_BINS, channel_dim)
            self.opp_clock_emb = nn.Embedding(NUM_TIME_LOG_BINS, channel_dim)
            self.inc_emb = nn.Embedding(NUM_INC_CATEGORIES, channel_dim)
            self.my_last_emb = nn.Embedding(NUM_TIME_LOG_BINS, channel_dim)
            self.opp_last_emb = nn.Embedding(NUM_TIME_LOG_BINS, channel_dim)
            embs = (
                self.base_time_emb, self.my_clock_emb, self.opp_clock_emb,
                self.inc_emb, self.my_last_emb, self.opp_last_emb,
            )

        for emb in embs:
            nn.init.normal_(emb.weight, std=0.02)
            emb._no_reinit = True

    @property
    def out_dim(self) -> int:
        return self.num_time_inputs * self.channel_dim

    def forward(
        self,
        base_time: torch.Tensor,
        increment: torch.Tensor,
        my_clock: Optional[torch.Tensor] = None,
        opp_clock: Optional[torch.Tensor] = None,
        my_last_time: Optional[torch.Tensor] = None,
        opp_last_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bt = self.base_time_emb(log_bin_time(base_time))
        inc = self.inc_emb(bin_increment(increment))
        if self.shrink_globals:
            return torch.cat([bt, inc], dim=-1)
        mc = self.my_clock_emb(log_bin_time(my_clock))
        oc = self.opp_clock_emb(log_bin_time(opp_clock))
        ml = self.my_last_emb(log_bin_time(my_last_time))
        ol = self.opp_last_emb(log_bin_time(opp_last_time))
        return torch.cat([bt, mc, oc, inc, ml, ol], dim=-1)


class TimeFiLMEncoder(nn.Module):
    """FiLM modulation (option 2): emit per-sample (gamma, beta) of shape
    (B, d_model) each, applied as x = x * (1 + gamma) + beta.

    Multiplicative modulation gives time signals a per-channel gate (different
    feature channels can be amplified or suppressed by time pressure) without
    paying any sequence-length cost. Output projections start at small nonzero
    std so the modulation is near-identity at init but the gradient signal
    stays alive (matches the ContinuousTimeEncoder dead-encoder reasoning).
    """
    FINAL_INIT_STD = 0.02

    def __init__(
        self,
        d_model: int,
        featurize: str = "log1p",
        hidden_dim: int = 128,
        shrink_globals: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.shrink_globals = shrink_globals
        num_time_inputs = 2 if shrink_globals else 6
        self.featurizer = _TimeFeaturizer(featurize, num_time_inputs)

        self.trunk = nn.Sequential(
            nn.Linear(self.featurizer.feat_dim, hidden_dim),
            nn.SiLU(),
        )
        self.gamma_head = nn.Linear(hidden_dim, d_model)
        self.beta_head = nn.Linear(hidden_dim, d_model)

        nn.init.normal_(self.trunk[0].weight, std=0.02)
        nn.init.zeros_(self.trunk[0].bias)
        nn.init.normal_(self.gamma_head.weight, std=self.FINAL_INIT_STD)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.normal_(self.beta_head.weight, std=self.FINAL_INIT_STD)
        nn.init.zeros_(self.beta_head.bias)
        for m in (self.trunk[0], self.gamma_head, self.beta_head):
            m._no_reinit = True

    def forward(
        self,
        base_time: torch.Tensor,
        increment: torch.Tensor,
        my_clock: Optional[torch.Tensor] = None,
        opp_clock: Optional[torch.Tensor] = None,
        my_last_time: Optional[torch.Tensor] = None,
        opp_last_time: Optional[torch.Tensor] = None,
    ):
        raw = _stack_time_inputs(
            base_time, increment, my_clock, opp_clock,
            my_last_time, opp_last_time, shrink_globals=self.shrink_globals,
        )
        h = self.trunk(self.featurizer(raw))
        gamma = self.gamma_head(h)                                            # (B, d_model)
        beta = self.beta_head(h)                                              # (B, d_model)
        return gamma, beta


class TimeAttnBiasEncoder(nn.Module):
    """Time-conditioned attention bias (option 3): emit (B, n_heads, 64, 64)
    additive bias from time signals, added to the GAB bias inside each block.

    Factorized the same way Maia3's GAB head is: a small per-sample trunk
    produces (B, n_heads, gen_size) coefficients, expanded against a shared
    learnable (64*64, gen_size) template matrix to (B, n_heads, 64, 64). This
    keeps the param count modest while still giving square-pair-conditional
    routing as a function of time. The bias is computed once per forward and
    reused across all blocks (cheaper than per-block recomputation, same
    expressivity ceiling per sample).
    """
    FINAL_INIT_STD = 0.02

    def __init__(
        self,
        n_heads: int,
        featurize: str = "log1p",
        hidden_dim: int = 64,
        gen_size: int = 64,
        shrink_globals: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.gen_size = gen_size
        self.shrink_globals = shrink_globals
        num_time_inputs = 2 if shrink_globals else 6
        self.featurizer = _TimeFeaturizer(featurize, num_time_inputs)

        self.trunk = nn.Sequential(
            nn.Linear(self.featurizer.feat_dim, hidden_dim),
            nn.SiLU(),
        )
        self.per_head_head = nn.Linear(hidden_dim, n_heads * gen_size)
        # Shared expansion template (matches Maia3 GAB's gab_weight structure).
        self.bias_template = nn.Parameter(torch.empty(64 * 64, gen_size))

        nn.init.normal_(self.trunk[0].weight, std=0.02)
        nn.init.zeros_(self.trunk[0].bias)
        nn.init.normal_(self.per_head_head.weight, std=self.FINAL_INIT_STD)
        nn.init.zeros_(self.per_head_head.bias)
        nn.init.xavier_normal_(self.bias_template)
        for m in (self.trunk[0], self.per_head_head):
            m._no_reinit = True
        self.bias_template._no_reinit = True

    def forward(
        self,
        base_time: torch.Tensor,
        increment: torch.Tensor,
        my_clock: Optional[torch.Tensor] = None,
        opp_clock: Optional[torch.Tensor] = None,
        my_last_time: Optional[torch.Tensor] = None,
        opp_last_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raw = _stack_time_inputs(
            base_time, increment, my_clock, opp_clock,
            my_last_time, opp_last_time, shrink_globals=self.shrink_globals,
        )
        B = raw.shape[0]
        h = self.trunk(self.featurizer(raw))
        per_head = self.per_head_head(h).view(B, self.n_heads, self.gen_size)
        bias = torch.einsum("bhi,oi->bho", per_head, self.bias_template)
        return bias.view(B, self.n_heads, 64, 64)


# =============================================================================
# Input encoder — board history + auxiliary planes + Elo, projected to dim_vit
# =============================================================================

def _reconstruct_per_position_clocks(
    time_history: torch.Tensor,
    my_clock_now: torch.Tensor,
    opp_clock_now: torch.Tensor,
    my_inc: torch.Tensor,
    opp_inc: torch.Tensor,
) -> torch.Tensor:
    """Mirror of PerSquareInputEncoder._reconstruct_per_position_clocks.

    Returns (B, H, 2): per-history-step (active_clock, opp_clock) in seconds.
    """
    B, H = time_history.shape
    device = time_history.device
    dtype = time_history.dtype

    idx = torch.arange(H, device=device)
    opp_moved = (idx % 2 == 0).to(dtype)
    my_moved = 1.0 - opp_moved

    opp_step = (time_history - opp_inc.unsqueeze(1)) * opp_moved
    my_step = (time_history - my_inc.unsqueeze(1)) * my_moved

    zeros = torch.zeros(B, 1, device=device, dtype=dtype)
    opp_cum = torch.cat([zeros, opp_step.cumsum(dim=1)[:, :-1]], dim=1)
    my_cum = torch.cat([zeros, my_step.cumsum(dim=1)[:, :-1]], dim=1)

    active_clocks = my_clock_now.unsqueeze(1) + my_cum
    opp_clocks = opp_clock_now.unsqueeze(1) + opp_cum
    return torch.stack([active_clocks, opp_clocks], dim=-1)


def _per_position_clock_deltas(
    time_history: torch.Tensor,
    my_inc: torch.Tensor,
    opp_inc: torch.Tensor,
) -> torch.Tensor:
    """Per-history-step clock *delta* for each side: how far that player's clock
    moved on the move that produced the position (spent - increment), in seconds.

    This is the consecutive-clock difference of `_reconstruct_per_position_clocks`
    exposed directly, so the model reads move *durations* (tempo) instead of
    having to finite-difference broadcast absolute clocks. Newest-first: index 0
    is the opponent's last move, index 1 the active player's last move,
    alternating; a side's delta is zero on the other side's plies. Signed —
    negative means the clock rose (moved faster than the increment).

    Returns (B, H, 2): per step (active_delta, opp_delta) in seconds.
    """
    B, H = time_history.shape
    idx = torch.arange(H, device=time_history.device)
    opp_moved = (idx % 2 == 0).to(time_history.dtype)
    my_moved = 1.0 - opp_moved
    opp_delta = (time_history - opp_inc.unsqueeze(1)) * opp_moved
    my_delta = (time_history - my_inc.unsqueeze(1)) * my_moved
    return torch.stack([my_delta, opp_delta], dim=-1)


class Maia3InputEncoder(nn.Module):
    """Build (B, 64, dim_vit) square tokens.

    Maia3-style: concatenate everything per-square (board history one-hots,
    auxiliary planes, broadcast Elo embeddings) and project with a single
    Linear. No per-square absolute position embedding — GAB carries position.
    """
    def __init__(self, config, time_channels_dim: int = 0):
        super().__init__()
        self.history = config["history"]
        self.num_piece_types = config["num_piece_types"]
        self.dim_emb = config["dim_emb"]
        self.dim_vit = config["dim_vit"]
        self.per_position_clocks = bool(config.get("per_position_clocks", False))
        self.per_position_time_deltas = bool(config.get("per_position_time_deltas", False))
        self.time_channels_dim = time_channels_dim

        # Per-square channels:
        #   piece one-hots:       history * num_piece_types
        #   ep mask:              1
        #   castling broadcast:   4
        #   rep_flags broadcast:  history
        #   per-position clocks:  2 * history (when enabled)
        #   elo per side:         2 * dim_emb
        #   time channels:        time_channels_dim (only when time_encoder='channels')
        self.board_dim = self.history * self.num_piece_types + 1
        self.state_dim = 4 + self.history
        if self.per_position_clocks:
            self.state_dim += 2 * self.history
        self.state_dim += 2 * self.dim_emb
        self.state_dim += time_channels_dim

        total_in = self.board_dim + self.state_dim
        self.token_projection = nn.Linear(total_in, self.dim_vit)

    def forward(
        self,
        board_history: torch.Tensor,    # (B, history, 64) int64
        rep_flags: torch.Tensor,        # (B, history) float
        castling: torch.Tensor,         # (B, 4) float
        ep_mask: torch.Tensor,          # (B, 64) float
        elo_per_side: torch.Tensor,     # (B, 2*dim_emb) float
        time_history: Optional[torch.Tensor] = None,
        my_clock_now: Optional[torch.Tensor] = None,
        opp_clock_now: Optional[torch.Tensor] = None,
        my_inc: Optional[torch.Tensor] = None,
        opp_inc: Optional[torch.Tensor] = None,
        time_channels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = board_history.shape[0]

        board_onehot = F.one_hot(board_history, num_classes=self.num_piece_types).float()
        board_onehot = board_onehot.permute(0, 2, 1, 3).reshape(B, 64, -1)

        ep_per_square = ep_mask.unsqueeze(-1)                              # (B, 64, 1)
        rep_broadcast = rep_flags.unsqueeze(1).expand(-1, 64, -1)          # (B, 64, history)
        castle_broadcast = castling.unsqueeze(1).expand(-1, 64, -1)        # (B, 64, 4)

        board_input = torch.cat([board_onehot, ep_per_square], dim=-1)
        state_parts = [castle_broadcast, rep_broadcast]

        if self.per_position_clocks:
            if self.per_position_time_deltas:
                # Per-move clock deltas (tempo). Signed log so a rising clock
                # (moved faster than the increment) stays representable.
                delta_per_pos = _per_position_clock_deltas(
                    time_history, my_inc, opp_inc,
                )                                                          # (B, history, 2)
                clock_feat = (
                    torch.sign(delta_per_pos)
                    * torch.log1p(delta_per_pos.abs())
                    / _LOG_BIN_TIME_DENOM
                )
            else:
                clock_per_pos = _reconstruct_per_position_clocks(
                    time_history, my_clock_now, opp_clock_now, my_inc, opp_inc,
                )                                                          # (B, history, 2)
                clock_feat = torch.log1p(clock_per_pos.clamp(min=0)) / _LOG_BIN_TIME_DENOM
            clock_flat = clock_feat.permute(0, 2, 1).reshape(B, 2 * self.history)
            state_parts.append(clock_flat.unsqueeze(1).expand(-1, 64, -1))

        state_parts.append(elo_per_side.unsqueeze(1).expand(-1, 64, -1))

        if self.time_channels_dim > 0:
            assert time_channels is not None, "time_channels required when time_channels_dim>0"
            state_parts.append(time_channels.unsqueeze(1).expand(-1, 64, -1))

        full_input = torch.cat([board_input] + state_parts, dim=-1)        # (B, 64, total_in)
        return self.token_projection(full_input)                           # (B, 64, dim_vit)


# =============================================================================
# Geometric Attention Bias (GAB) — shared weight produces (B, H, 64, 64) bias
# =============================================================================

def apply_gab_rotation(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Apply per-square, per-head 2D-plane rotations to q or k (RoPE-style).

    A dynamic, board-conditional generalization of RoPE: instead of fixed
    position-dependent rotations, the angles are emitted by GAB from the board
    state. Each adjacent channel pair (2d, 2d+1) is rotated by angle[..., d].
    Norm-preserving, so it composes cleanly with the surrounding norms.

    Args:
        x: (B, H, L, head_dim) - q or k.
        angles: (B, H, L, head_dim/2) - rotation angle per 2D plane. Positions
            without a GAB angle (e.g. conditioning tokens) carry angle 0, which
            makes the rotation an identity there.
    Returns:
        (B, H, L, head_dim) rotated tensor.
    """
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack((rx1, rx2), dim=-1).flatten(-2)


class Maia3GAB(nn.Module):
    """Per-block GAB head that consumes the 64 square tokens and emits
    (B, num_heads, 64, 64) attention biases and/or per-square Q/K rotation angles.

    Mirrors `MHA._sq_bias` in upstream. The final shared parameters
    `gab_weight: (64*64, gab_gen_size)` (additive bias) and, when rotation is on,
    `gab_rot_weight: (64*rot_planes, gab_gen_size)` are owned by `Maia3Net` and
    passed in by reference so they are shared across all encoder blocks.

    Two front-end modes (controlled by `gab_per_square_dim`):
      - > 0: per-square projection then flatten the 64 squares (Maia3 23M/79M).
      - 0:   mean-pool over the 64 squares (Maia3 3M/5M).
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        gab_gen_size: int,
        gab_per_square_dim: int,
        gab_intermediate_dim: int,
        gab_weight: Optional[nn.Parameter],
        gab_rot_weight: Optional[nn.Parameter] = None,
        rot_planes: int = 0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.gen_size = gab_gen_size
        self.per_square_dim = gab_per_square_dim
        self.gab_weight = gab_weight            # shared additive-bias param (or None)
        self.gab_rot_weight = gab_rot_weight    # shared rotation-angle param (or None)
        self.rot_planes = rot_planes
        self.emit_bias = gab_weight is not None
        self.emit_rotation = gab_rot_weight is not None

        if gab_per_square_dim == 0:
            self.sm1 = None
            self.sm2 = nn.Linear(d_model, gab_intermediate_dim)
        else:
            self.sm1 = nn.Linear(d_model, gab_per_square_dim)
            self.sm2 = nn.Linear(64 * gab_per_square_dim, gab_intermediate_dim)
        self.ln1 = nn.LayerNorm(gab_intermediate_dim)
        self.sm3 = nn.Linear(gab_intermediate_dim, num_heads * gab_gen_size)
        self.ln2 = nn.LayerNorm(num_heads * gab_gen_size)
        self.act = nn.GELU()

    def forward(self, squares: torch.Tensor):
        """squares: (B, 64, d_model) — board square tokens only.

        Returns (bias, angles):
          bias:   (B, H, 64, 64) additive GAB bias, or None.
          angles: (B, H, 64, rot_planes) per-square rotation angles, or None.
        """
        B = squares.size(0)
        if self.sm1 is not None:
            y = self.sm1(squares)            # (B, 64, per_sq)
            y = y.reshape(B, -1)             # (B, 64*per_sq)
            y = self.act(self.sm2(y))        # (B, gab_intermediate_dim)
        else:
            y = squares.mean(dim=1)          # (B, d_model)
            y = self.act(self.sm2(y))        # (B, gab_intermediate_dim)
        y = self.ln1(y)
        y = self.act(self.sm3(y))            # (B, H*gen)
        y = self.ln2(y).view(B, self.num_heads, self.gen_size)

        bias = None
        if self.emit_bias:
            # einsum: (B, H, gen) x (64*64, gen) -> (B, H, 64*64)
            b = torch.einsum("bhi,oi->bho", y, self.gab_weight)
            bias = b.view(B, self.num_heads, 64, 64)

        angles = None
        if self.emit_rotation:
            # einsum: (B, H, gen) x (64*rot_planes, gen) -> (B, H, 64*rot_planes)
            a = torch.einsum("bhi,oi->bho", y, self.gab_rot_weight)
            angles = a.view(B, self.num_heads, 64, self.rot_planes)

        return bias, angles


# =============================================================================
# Multi-head attention — direct SDPA so a 4D float GAB bias rides the
# memory-efficient attention backend instead of the manual fallback that
# nn.MultiheadAttention forces with a 3D float attn_mask.
# =============================================================================

class Maia3MHA(nn.Module):
    """MHA matching nn.MultiheadAttention's parameter layout (combined in_proj
    + out_proj) so init scales stay identical to upstream Maia3, but the
    forward path calls F.scaled_dot_product_attention directly with a 4D
    float bias. SDPA selects the memory-efficient backend, which is much
    faster than the python-level manual attention nn.MultiheadAttention falls
    back to whenever a per-batch-per-head float attn_mask is supplied.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float, omit_qkv_biases: bool):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout
        bias = not omit_qkv_biases

        # Layout mirrors nn.MultiheadAttention: a single in_proj_weight of
        # shape (3*d_model, d_model) and an out_proj Linear.
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Match upstream init exactly. nn.MultiheadAttention._reset_parameters
        # does xavier_uniform_ on in_proj_weight, zeros on biases; out_proj
        # gets nn.Linear's default kaiming_uniform_.
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.zeros_(self.in_proj_bias)
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        gab_rotation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: (B, L, d_model). attn_bias: (B, num_heads, L, L) additive float bias or None.
        gab_rotation: (B, num_heads, L, head_dim/2) per-square rotation angles or None."""
        B, L, _ = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)        # (B, L, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)
        # (B, L, d_model) -> (B, num_heads, L, head_dim)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Dynamic GAB rotation: rotate q_i and k_j by their own square's angle
        # before SDPA -> the bilinear (R_i q_i)·(R_j k_j) term. Norm-preserving,
        # so it stays compatible with the memory-efficient attention backend.
        if gab_rotation is not None:
            q = apply_gab_rotation(q, gab_rotation)
            k = apply_gab_rotation(k, gab_rotation)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.dropout_p if self.training else 0.0,
        )                                                                 # (B, H, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


# =============================================================================
# Encoder block — post-LayerNorm order (upstream EncoderOnlyBlock)
# =============================================================================

class Maia3EncoderBlock(nn.Module):
    def __init__(
        self,
        config: dict,
        gab_weight: Optional[nn.Parameter],
        gab_rot_weight: Optional[nn.Parameter] = None,
        rot_planes: int = 0,
    ):
        super().__init__()
        d_model = config["dim_vit"]
        num_heads = config["num_heads"]
        dropout = config["dropout"]
        use_gab = bool(config["use_gab"])
        d_ff = int(d_model * config["mlp_ratio"])

        self.use_gab = use_gab
        self.gab = None
        if use_gab:
            self.gab = Maia3GAB(
                d_model=d_model,
                num_heads=num_heads,
                gab_gen_size=config["gab_gen_size"],
                gab_per_square_dim=config["gab_per_square_dim"],
                gab_intermediate_dim=config["gab_intermediate_dim"],
                gab_weight=gab_weight,
                gab_rot_weight=gab_rot_weight,
                rot_planes=rot_planes,
            )

        self.self_attn = Maia3MHA(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            omit_qkv_biases=bool(config["omit_qkv_biases"]),
        )
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout_sa = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        Norm = nn.RMSNorm if config["use_rms_norm"] else nn.LayerNorm
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)

        act_name = config["activation"].lower()
        if act_name == "gelu":
            self.activation = F.gelu
        elif act_name == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unknown activation: {act_name}")

    def forward(
        self,
        x: torch.Tensor,
        num_cond_tokens: int = 0,
        time_attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Compute GAB outputs from the 64 square tokens: an additive bias and/or
        # per-square Q/K rotation angles. Optionally add the time-conditioned
        # attention bias (option 3 ablation) to the additive path. Both operate on
        # the 64-square block; pad to full seq if cond tokens are prepended (bias
        # padded with 0, angles with 0 -> identity rotation on cond tokens).
        attn_bias = None
        gab_rotation = None
        if self.use_gab or time_attn_bias is not None:
            squares = x[:, num_cond_tokens:, :] if num_cond_tokens > 0 else x
            bias_64 = None
            angles_64 = None
            if self.use_gab:
                bias_64, angles_64 = self.gab(squares)
            if time_attn_bias is not None:
                bias_64 = time_attn_bias if bias_64 is None else bias_64 + time_attn_bias
            if num_cond_tokens > 0:
                B, L = x.shape[0], x.shape[1]
                if bias_64 is not None:
                    H = bias_64.shape[1]
                    attn_bias = torch.zeros(B, H, L, L, device=x.device, dtype=bias_64.dtype)
                    attn_bias[:, :, num_cond_tokens:, num_cond_tokens:] = bias_64
                if angles_64 is not None:
                    H, P = angles_64.shape[1], angles_64.shape[3]
                    gab_rotation = torch.zeros(B, H, L, P, device=x.device, dtype=angles_64.dtype)
                    gab_rotation[:, :, num_cond_tokens:, :] = angles_64
            else:
                attn_bias = bias_64
                gab_rotation = angles_64

        # Post-LN: norm AFTER residual
        sa_out = self.self_attn(x, attn_bias=attn_bias, gab_rotation=gab_rotation)
        x = self.norm1(x + self.dropout_sa(sa_out))
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout_ff(ff_out))
        return x


# =============================================================================
# Main model
# =============================================================================

class Maia3Net(nn.Module):
    """Maia3 baseline with our temporal conditioning + resign/flag/time heads.

    Forward signature and return-tuple layout match Chessformer for drop-in
    compatibility with train.py's loss code.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.no_time_context = bool(config.get("no_time_context", False))
        self.gradient_checkpointing = bool(config.get("gradient_checkpointing", False))

        dim_vit = config["dim_vit"]
        dim_emb = config["dim_emb"]
        head_hid_dim = config["head_hid_dim"]

        # Elo conditioning (Maia3-style endpoint interpolation).
        self.elo_encoder = Maia3EloEncoder(dim_emb)

        # Time conditioning (ours, optional). Sequence length grows by
        # `num_cond_tokens` when the 'tokens' variant is in use.
        self.time_encoder_variant = config.get("time_encoder", "tokens")
        self.time_featurize = config.get("time_featurize", "log1p")
        self.per_position_clocks = bool(config.get("per_position_clocks", False))
        self.shrink_global_time = (
            self.per_position_clocks and bool(config.get("shrink_global_time", True))
        )
        # Tempo-history token: route the full 8-ply time_history through one
        # interp-token (replaces my_last/opp_last); pair with --no-per-position-clocks.
        self.tempo_history_token = bool(config.get("tempo_history_token", False))
        # Ablation: blank the per-move time_history (zeros) so the model keeps
        # current clocks + TC + the time head, but loses ALL tempo/move-time-history
        # info (my_last/opp_last tokens and per-position channels go constant).
        self.zero_time_history = bool(config.get("zero_time_history", False))
        # Time-blind ablation: build the model exactly as the time-conditioned
        # baseline (encoder, time head, flag head all present and identically
        # shaped) but feed constant (zero) time signals in forward(), so the
        # model cannot condition on time. The only difference is the information,
        # not the parameters — so the time head still trains and is comparable.
        # Distinct from no_time_context, which removes the heads.
        self.no_time_conditioning = bool(config.get("no_time_conditioning", False))

        # Time conditioning variants:
        #   'tokens'   — prepend log-binned tokens to the sequence (baseline)
        #   'mlp'      — broadcast-add a single (B, d_model) vector to all squares
        #   'channels' — per-signal log-binned embeddings concatenated as input channels
        #   'interp_channels' — Elo-style scalar endpoint interpolation concatenated as
        #                input channels (continuous; the closest mirror of elo conditioning)
        #   'film'     — per-sample (gamma, beta) modulation applied after input encoding
        #   'bias'     — (B, n_heads, 64, 64) additive attention bias added to GAB each block
        time_channels_dim = 0
        if self.no_time_context:
            self.time_encoder = None
            self.num_cond_tokens = 0
        elif self.time_encoder_variant == "tokens":
            self.time_encoder = build_time_encoder(
                "tokens", dim_vit, shrink_globals=self.shrink_global_time,
                token_mode=config.get("time_token_encoder", "bins"),
                tempo_history_token=self.tempo_history_token,
            )
            self.num_cond_tokens = self.time_encoder.num_tokens
        elif self.time_encoder_variant == "mlp":
            self.time_encoder = _build_with_rng_snapshot(
                lambda: build_time_encoder(
                    "mlp", dim_vit,
                    featurize=self.time_featurize,
                    shrink_globals=self.shrink_global_time,
                )
            )
            self.num_cond_tokens = 0
        elif self.time_encoder_variant == "channels":
            channel_dim = int(config.get("time_channel_dim", 16))
            self.time_encoder = _build_with_rng_snapshot(
                lambda: TimeChannelEncoder(
                    channel_dim=channel_dim,
                    shrink_globals=self.shrink_global_time,
                )
            )
            time_channels_dim = self.time_encoder.out_dim
            self.num_cond_tokens = 0
        elif self.time_encoder_variant == "interp_channels":
            # Elo-style scalar interpolation routed into per-square input
            # channels (see InterpTimeChannelEncoder). emb_dim is the per-signal
            # channel width — the knob a too-small earlier attempt got wrong.
            # Default to dim_vit so each signal gets the SAME representational
            # width it would as a prepended token (clean tokens-vs-channels
            # comparison: identical information, identical scaling, identical
            # per-signal capacity — only the routing mechanism differs).
            _cfg_dim = config.get("time_interp_channel_dim", None)
            emb_dim = int(_cfg_dim) if _cfg_dim else dim_vit
            self.time_encoder = _build_with_rng_snapshot(
                lambda: InterpTimeChannelEncoder(
                    emb_dim=emb_dim,
                    shrink_globals=self.shrink_global_time,
                    tempo_history_token=self.tempo_history_token,
                )
            )
            time_channels_dim = self.time_encoder.out_dim
            self.num_cond_tokens = 0
        elif self.time_encoder_variant == "film":
            self.time_encoder = _build_with_rng_snapshot(
                lambda: TimeFiLMEncoder(
                    dim_vit,
                    featurize=self.time_featurize,
                    hidden_dim=int(config.get("time_film_hidden", 128)),
                    shrink_globals=self.shrink_global_time,
                )
            )
            self.num_cond_tokens = 0
        elif self.time_encoder_variant == "bias":
            self.time_encoder = _build_with_rng_snapshot(
                lambda: TimeAttnBiasEncoder(
                    n_heads=config["num_heads"],
                    featurize=self.time_featurize,
                    hidden_dim=int(config.get("time_bias_hidden", 128)),
                    shrink_globals=self.shrink_global_time,
                )
            )
            self.num_cond_tokens = 0
        else:
            raise ValueError(
                f"Unknown time_encoder variant: {self.time_encoder_variant!r}. "
                f"Known: tokens, mlp, channels, interp_channels, film, bias"
            )

        self.total_seq_len = self.num_cond_tokens + NUM_SQUARES

        # Input encoder (knows about time_channels_dim when variant='channels').
        self.input_encoder = Maia3InputEncoder(config, time_channels_dim=time_channels_dim)

        # GAB ablation knobs. use_gab gates the GAB head; within it, the additive
        # bias and the Q/K rotation are independently toggleable (Arms A/B/C).
        use_gab = bool(config["use_gab"])
        self.gab_additive_bias = use_gab and bool(config.get("gab_additive_bias", True))
        self.gab_qk_rotation = use_gab and bool(config.get("gab_qk_rotation", False))
        if use_gab:
            assert self.gab_additive_bias or self.gab_qk_rotation, \
                "GAB must emit at least one of additive bias / Q-K rotation"
        self.rot_planes = (config["dim_vit"] // config["num_heads"]) // 2

        # Shared GAB additive-bias weight across all blocks (upstream owns this on
        # the top-level MAIA3Model). Init matches upstream's xavier_normal_.
        if self.gab_additive_bias:
            self.gab_shared_weight = nn.Parameter(torch.empty(64 * 64, config["gab_gen_size"]))
            nn.init.xavier_normal_(self.gab_shared_weight)
        else:
            self.gab_shared_weight = None

        # Shared GAB rotation-angle weight (zero-init -> identity rotation at step
        # 0, so enabling it leaves a bias-equipped model identical to the baseline
        # at init). Also shared across all blocks.
        if self.gab_qk_rotation:
            self.gab_rot_shared_weight = nn.Parameter(
                torch.zeros(64 * self.rot_planes, config["gab_gen_size"])
            )
        else:
            self.gab_rot_shared_weight = None

        # Transformer stack.
        self.layers = nn.ModuleList([
            Maia3EncoderBlock(
                config,
                gab_weight=self.gab_shared_weight,
                gab_rot_weight=self.gab_rot_shared_weight,
                rot_planes=self.rot_planes,
            )
            for _ in range(config["num_blocks"])
        ])
        # Final LayerNorm (upstream uses LayerNorm at the end of the encoder
        # regardless of `use_rms_norm` for the blocks).
        self.last_ln_trunk = nn.LayerNorm(dim_vit)

        # --- Output heads ---

        # Policy: bilinear from/to.
        self.proj_sq_from = nn.Linear(dim_vit, head_hid_dim, bias=False)
        self.proj_sq_to = nn.Linear(dim_vit, head_hid_dim, bias=False)
        self.policy_scale = 1.0 / math.sqrt(head_hid_dim)

        # Promotion: per-rank-8 file bias, output reshaped to (B, 8, 4)
        # to match train.py's loss-code expectation.
        self.promo_bias_proj = nn.Linear(head_hid_dim, 4, bias=False)

        # WDL: mean-pool → MLP (matches upstream).
        self.fc_value_hid = nn.Linear(dim_vit, head_hid_dim)
        self.fc_value = nn.Linear(head_hid_dim, NUM_VALUE_CLASSES)
        self.last_ln_pool = nn.LayerNorm(dim_vit)

        # Resign / flag binary heads (ours).
        self.resign_head = BinaryHead(dim_vit)
        # Flag (time-forfeit) head: dropped whenever the model can't see the
        # clock. Flagging is almost purely a clock phenomenon, so without time
        # signals the head can only learn the base rate — degenerate, and its
        # gradients would just inject noise into the shared trunk. (The move-time
        # head, by contrast, captures board-intrinsic think-time and is kept.)
        self.use_flag_head = not (self.no_time_context or self.no_time_conditioning)
        self.flag_head = BinaryHead(dim_vit) if self.use_flag_head else None

        # Move-conditional time head (ours).
        if not self.no_time_context:
            self.time_baseline_pool = AttentionPooling(dim_vit, context_dim=None)
            self.time_baseline_head = nn.Sequential(
                nn.Linear(dim_vit, 128),
                Mish(),
                nn.Linear(128, NUM_TIME_BINS),
            )
            self.time_from_proj = nn.Linear(dim_vit, NUM_TIME_BINS, bias=False)
            self.time_to_proj = nn.Linear(dim_vit, NUM_TIME_BINS, bias=False)

        # Init policy: match upstream Maia3 — explicit xavier_normal_ on the
        # Elo anchor embeddings and the shared GAB weight (both done inside
        # their respective submodules), and PyTorch defaults everywhere else.
        # This is essential because nn.MultiheadAttention stores in_proj_weight
        # as a Parameter (not nn.Linear), so a model-wide apply(N(0, 0.02))
        # would overwrite out_proj but leave in_proj_weight at its xavier
        # default — a ~4x mismatch that hurts convergence.

    def forward(self, batch, return_promo: bool = False):
        board_history = batch["board_history"]
        rep_flags = batch["rep_flags"]
        castling = batch["castling"]
        ep_mask = batch["ep_mask"]
        scalars = batch["scalars"]
        legal_mask = batch.get("legal_mask", None)

        B = board_history.shape[0]

        # Scalars layout (same as Chessformer):
        # 0 elo_norm  1 opp_elo_norm  2 ply_norm
        # 3 active_clock_s  4 opp_clock_s  5 active_inc_s  6 opp_inc_s
        # 7 hmc_norm  8 base_time_s
        player_elo_raw = scalars[:, 0] * 700 + 1900
        opp_elo_raw = scalars[:, 1] * 700 + 1900
        elo_per_side = self.elo_encoder(player_elo_raw, opp_elo_raw)

        time_history = batch["time_history"]
        if self.zero_time_history or self.no_time_conditioning:
            time_history = torch.zeros_like(time_history)
        my_clock_raw = scalars[:, 3]
        opp_clock_raw = scalars[:, 4]
        my_inc_raw = scalars[:, 5]
        opp_inc_raw = scalars[:, 6]
        increment_raw = my_inc_raw
        base_time_raw = scalars[:, 8]
        if self.no_time_conditioning:
            # Time-blind: blank every time-derived signal before it reaches the
            # time encoder / per-position-clock input encoder. The encoder and
            # tokens still exist (constant output -> a learned bias), so the
            # architecture is unchanged; the model simply has no time info to
            # condition on. The time + flag heads are untouched below.
            zero = torch.zeros_like(base_time_raw)
            my_clock_raw = zero
            opp_clock_raw = zero
            my_inc_raw = zero
            opp_inc_raw = zero
            increment_raw = zero
            base_time_raw = zero
        my_last_time_raw = time_history[:, 1]
        opp_last_time_raw = time_history[:, 0]

        # Build the time-encoder kwargs once; encoders ignore the clock/last
        # args when shrink_global_time is on.
        time_kwargs = dict(base_time=base_time_raw, increment=increment_raw)
        if not self.shrink_global_time:
            time_kwargs.update(
                my_clock=my_clock_raw, opp_clock=opp_clock_raw,
                my_last_time=my_last_time_raw, opp_last_time=opp_last_time_raw,
            )
        # The tempo-history token (interp encoder) consumes the full 8-ply history.
        if self.tempo_history_token and self.time_encoder_variant in ("tokens", "interp_channels"):
            time_kwargs["time_history"] = time_history

        # 'channels'/'interp_channels' variants feed into the input encoder, so encode first.
        time_channels = None
        if (not self.no_time_context) and self.time_encoder_variant in ("channels", "interp_channels"):
            time_channels = self.time_encoder(**time_kwargs)

        if self.per_position_clocks:
            x = self.input_encoder(
                board_history, rep_flags, castling, ep_mask, elo_per_side,
                time_history=time_history,
                my_clock_now=my_clock_raw,
                opp_clock_now=opp_clock_raw,
                my_inc=my_inc_raw,
                opp_inc=opp_inc_raw,
                time_channels=time_channels,
            )
        else:
            x = self.input_encoder(
                board_history, rep_flags, castling, ep_mask, elo_per_side,
                time_channels=time_channels,
            )

        # 'bias' variant produces a square-pair bias added to GAB each block.
        time_attn_bias = None
        if not self.no_time_context:
            if self.time_encoder_variant == "tokens":
                cond_tokens = self.time_encoder(**time_kwargs)
                x = torch.cat([cond_tokens, x], dim=1)
            elif self.time_encoder_variant == "mlp":
                time_repr = self.time_encoder(**time_kwargs)
                x = x + time_repr.unsqueeze(1)
            elif self.time_encoder_variant == "film":
                gamma, beta = self.time_encoder(**time_kwargs)
                x = x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            elif self.time_encoder_variant == "bias":
                time_attn_bias = self.time_encoder(**time_kwargs)              # (B, H, 64, 64)
            # 'channels' was handled before the input encoder.

        NC = self.num_cond_tokens
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = gradient_checkpoint(
                    layer, x, NC, time_attn_bias, use_reentrant=False,
                )
            else:
                x = layer(x, NC, time_attn_bias=time_attn_bias)
        x = self.last_ln_trunk(x)

        x_squares = x[:, NC:, :]                                              # (B, 64, dim_vit)

        # --- Policy head: bilinear ---
        sq_from = self.proj_sq_from(x_squares)                                # (B, 64, H)
        sq_to = self.proj_sq_to(x_squares)                                    # (B, 64, H)
        move_logits = torch.einsum("bid,bjd->bij", sq_from, sq_to) * self.policy_scale
        move_logits = move_logits.reshape(B, 64 * 64)                         # (B, 4096)
        if legal_mask is not None:
            move_logits = move_logits.masked_fill(~legal_mask, float("-inf"))

        # --- Promotion head ---
        # Per-file bias from the rank-8 to-square keys. Upstream scales this
        # by sqrt(head_hid_dim) because they add it to the unscaled base score;
        # we emit the bias-only signal so our trainer's (B,8,4)-indexed loss
        # works unchanged.
        rank8_keys = sq_to[:, 56:64, :]                                       # (B, 8, H)
        promo_logits = self.promo_bias_proj(rank8_keys)                       # (B, 8, 4)

        # --- WDL head ---
        pooled = self.last_ln_pool(x_squares.mean(dim=1))                     # (B, dim_vit)
        wdl_logits = self.fc_value(F.relu(self.fc_value_hid(pooled)))         # (B, 3)

        # --- Resign / flag ---
        resign_logit = self.resign_head(x_squares)
        flag_logit = self.flag_head(x_squares) if self.use_flag_head else None

        # --- Move-conditional time head ---
        if self.no_time_context:
            time_baseline = None
            time_from_logits = None
            time_to_logits = None
        else:
            time_pooled = self.time_baseline_pool(x_squares)
            time_baseline = self.time_baseline_head(time_pooled)              # (B, T)
            time_from_logits = self.time_from_proj(x_squares)                 # (B, 64, T)
            time_to_logits = self.time_to_proj(x_squares)                     # (B, 64, T)

        if return_promo:
            return (move_logits, wdl_logits, resign_logit, flag_logit,
                    time_baseline, time_from_logits, time_to_logits, promo_logits)
        return (move_logits, wdl_logits, resign_logit, flag_logit,
                time_baseline, time_from_logits, time_to_logits)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Dummy batch. Scalars layout matches dataset.py / model.py:
    #   0 elo_norm  1 opp_elo_norm  2 ply_norm
    #   3 my_clock_s  4 opp_clock_s  5 my_inc_s  6 opp_inc_s
    #   7 hmc_norm  8 base_time_s
    B = 2
    scalars = torch.tensor([
        [(1500 - 1900) / 700, (1700 - 1900) / 700, 0.5,
         60.0, 60.0, 2.0, 2.0, 0.1, 180.0],
        [(2000 - 1900) / 700, (1900 - 1900) / 700, 0.7,
         300.0, 280.0, 0.0, 0.0, 0.2, 600.0],
    ])
    dummy_batch = {
        'board_history': torch.randint(0, 13, (B, 8, 64)),
        'time_history': torch.rand(B, 8) * 60,
        'rep_flags': torch.randint(0, 2, (B, 8)).float(),
        'castling': torch.randint(0, 2, (B, 4)).float(),
        'ep_mask': torch.zeros(B, 64),
        'scalars': scalars,
        'legal_mask': torch.ones(B, 4096, dtype=torch.bool),
    }

    print("=" * 60)
    print("MAIA3 MODEL CONFIGS")
    print("=" * 60)

    configs = [
        ("MAIA3_3M", CONFIG_MAIA3_3M),
        ("MAIA3_5M", CONFIG_MAIA3_5M),
        ("MAIA3_23M", CONFIG_MAIA3_23M),
        ("MAIA3_79M", CONFIG_MAIA3_79M),
    ]

    for name, base_config in configs:
        has_time = not base_config.get('no_time_context')
        # tokens (baseline) + 4 time-conditioning ablations + an mlp variant
        if has_time:
            runs = [
                ('tokens', None),
                ('mlp', 'log1p'),
                ('channels', None),
                ('film', 'log1p'),
                ('bias', 'log1p'),
            ]
        else:
            runs = [('tokens', None)]
        for variant, feat in runs:
            config = dict(base_config)
            config['time_encoder'] = variant
            if feat is not None:
                config['time_featurize'] = feat
            tag = f"time_encoder={variant}" + (f", featurize={feat}" if feat else "")
            print(f"\n--- CONFIG_{name} ({tag}) ---")
            model = Maia3Net(config)
            print(f"Parameters: {count_parameters(model):,}")
            out = model(dummy_batch)
            (move_logits, wdl_logits, resign_logit, flag_logit,
             time_baseline, time_from_logits, time_to_logits) = out
            flag_shape = flag_logit.shape if flag_logit is not None else "disabled"
            print(f"Forward pass: OK (move_logits={move_logits.shape}, wdl={wdl_logits.shape}, "
                  f"resign={resign_logit.shape}, flag={flag_shape})")
            if time_baseline is not None:
                print(f"  time_baseline={time_baseline.shape}, time_from={time_from_logits.shape}, time_to={time_to_logits.shape}")
            else:
                print("  time head: disabled (no_time_context)")
