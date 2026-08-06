from __future__ import annotations

import chess


# Settings (Defaults) copied from `inference/app.py`.
DEFAULT_GAME_SETTINGS: dict = {
    "temperature": 0.9,
    "time_temperature": 0.9,
    "top_p": 0.95,
    "time_top_p": 0.95,
    # Dynamic temperature decay (temp_start/temp_min/temp_decay_rate) is NOT
    # set here — it only applies to the UCI engine path (lichess-bot configs).
    # When these keys are absent, dynamic_temperature() falls through to the
    # plain "temperature" value, which is what the web app slider controls.
    "active_elo": 1800,
    "opponent_elo": 1800,
    "human_elo": 1800,
    "engine_elo": 1800,
    "use_real_ratings": False,
    "match_active_elo_to_opponent": False,
    "match_active_elo_offset": 0,
    "human_color": chess.WHITE,
    "auto_play": False,
    "simulate_thinking_time": False,
    "internal_clock": False,
    "use_real_time": False,
    "use_mode_time": False,
    "use_expected_time": True,
    # Multiplicative stretch on the model's predicted move time. The time head
    # under-predicts (mass on 0-2s), so >1.0 makes think times human-length
    # while preserving the sampled distribution's shape/variance. 1.0 = raw.
    "time_scale": 1.0,
    # Small floor (seconds) to avoid literal 0.0s replies. Keep it LOW — a large
    # floor flattens every move to a constant and looks robotic. 0.0 disables.
    "min_move_time_s": 0.0,
    # Safety cap: the floor above never consumes more than this fraction of the
    # remaining clock, so the bot won't flag itself when it is low on time.
    "min_move_time_clock_fraction": 0.3,
    "start_clock_s": 300.0,
    "inc_s": 0,
    "analysis_use_clock_overrides": False,
    "analysis_active_clock_s": 300.0,
    "analysis_opponent_clock_s": 300.0,
    # Device selection for model inference: 'auto' (default), 'cuda', or 'cpu'
    "device": "cuda",

    # MCTS (disabled by default)
    "use_mcts": False,
    "mcts_simulations": 256,
    "mcts_c_puct": 1.2,
    # lc0-style dynamic PUCT: cpuct = log((N + base + 1) / base) * factor + c_puct
    # Set cpuct_factor=0 to disable (static c_puct only).
    "mcts_cpuct_base": 19768,
    "mcts_cpuct_factor": 2.0,
    "mcts_max_children": 16,
    "mcts_root_dirichlet_alpha": 0.0,
    "mcts_root_exploration_frac": 0.0,

    "mcts_final_temperature": 0.0,
    "mcts_final_top_p": 0.95,  # Top-p (nucleus) sampling for final move selection (1.0 = disabled)
    "mcts_max_depth": 24,
    "mcts_adaptive": False,
    "mcts_adaptive_scale": 30.0,
    "mcts_contempt": 0.15,  # Penalize draws to avoid drawish positions when ahead
    "mcts_fpu_reduction": 0.33,  # First Play Urgency reduction (lc0-style)
    "mcts_batch_size": 16,  # Batched NN evaluation (1 = unbatched, 16-32 = ~5x faster on GPU)
    "mcts_simulate_time": False,  # Simulate remaining thinking time after MCTS completes
    "mcts_start_ply": 0,  # Ply at which MCTS kicks in (0 = from start, higher = skip opening for variety)
    "mcts_tree_reuse": True,  # Reuse search tree from previous position for better follow-up moves
    "mcts_kl_regularized": False,  # KL-regularized move selection: blend Q-values with human prior
    "mcts_kl_coeff": 1.0,  # c in lambda = c / sqrt(N_sim); higher = more human-like
    "show_mcts_stats": True,

    # Pondering (thinking during opponent's time)
    "ponder": False,

    # Overlays

    # Arrow for top move
    "show_arrows": True,
}


# Fixed Context (copied from `inference/app.py`).
START_CLOCK_S = 300.0
INC_S = 0


# App-level RNG seed (copied from `inference/app.py`).
DEFAULT_RNG_SEED = 67
