# The UCI engine

`python -m inference.uci_engine`, or through the launchers in `scripts/`
(`marvin-uci.bat` on Windows, `marvin-uci.sh` elsewhere), which any chess GUI can
point at directly. It runs the exported ONNX weights, so it needs
`requirements-engine.txt` but not PyTorch, and it picks up whichever `.onnx`
files are in this directory. The engine reads the clock from the GUI's `go`
command (`wtime`, `btime`, `winc`, `binc`) and builds the model's temporal
inputs from it, so a timed game needs no configuration. Every option below is
listed by `uci`, with the defaults shown here.

## Strength

| Option | Type | Default | |
|---|---|---|---|
| `ActiveElo` | spin 1200-2400 | 1800 | Rating the engine plays as |
| `OpponentElo` | spin 1200-2400 | 1800 | Rating it treats you as, which shapes its choices |
| `UseRealRatings` | check | false | Take both ratings from the GUI when it supplies them |
| `MatchActiveEloToOpponent` | check | false | Set `ActiveElo` from `OpponentElo` |
| `MatchActiveEloOffset` | spin -1000-1000 | 0 | Offset applied when matching |

## Move sampling

| Option | Type | Default | |
|---|---|---|---|
| `Temperature` | string | 0.9 | Policy sampling temperature; lower is more deterministic |
| `TopP` | string | 0.95 | Nucleus cutoff over legal moves |
| `TempStart` | string | 0.9 | Temperature at the start of the game |
| `TempMin` | string | 0.9 | Floor the temperature decays toward |
| `TempDecayRate` | string | 0.1 | How fast temperature decays from `TempStart` to `TempMin` |

## Thinking time

The model predicts a distribution over how long a human would think. These
options choose what is taken from it, and whether the engine actually waits.

| Option | Type | Default | |
|---|---|---|---|
| `UseModeTime` | check | false | Use the most likely bin. Takes precedence over `UseExpectedTime` |
| `UseExpectedTime` | check | true | Use the mean of the distribution |
| `UseRealTime` | check | false | Reconstruct the opponent's actual move times from GUI clock deltas and feed those as the tempo history, instead of the model's own estimates |
| `TimeTemperature` | string | 0.9 | Temperature on the time distribution |
| `TimeTopP` | string | 0.95 | Nucleus cutoff on the time distribution |
| `SimulateThinkingTime` | check | false | Actually wait the predicted time before replying |
| `TimeScale` | string | 1.0 | Multiplier on the predicted time |
| `MinMoveTime` | string | 0.0 | Floor on seconds per move |
| `MinMoveTimeClockFraction` | string | 0.3 | Skip that floor below this fraction of the clock, so the engine does not flag itself |

With both `UseModeTime` and `UseExpectedTime` off, the think time is sampled
from the distribution under `TimeTemperature` and `TimeTopP`. Those two also
shape the expected time, which is computed over the temperature-scaled,
top-p-filtered bins. The mode is an argmax over the unscaled distribution, so
neither affects it.

## Clock source

| Option | Type | Default | |
|---|---|---|---|
| `InternalClock` | check | false | Run an internal clock instead of reading the GUI's |
| `GameBaseTime` | string | 0 | Base time in seconds for the internal clock |
| `GameIncrement` | string | 0 | Increment in seconds for the internal clock |
| `StaticTimeHistory` | check | false | Feed one fixed per-move time instead of the real history |
| `StaticTimeHistoryValue` | string | 5.0 | The fixed value, in seconds |

## Resignation and flagging

| Option | Type | Default | |
|---|---|---|---|
| `EnableResign` | check | false | Allow the engine to resign |
| `ResignThreshold` | string | 0.98 | Resignation-head probability required |
| `MinResignPly` | spin 0-1000 | 20 | Never resign before this ply |
| `EnableFlag` | check | false | Allow the time-forfeit head to end the game |
| `FlagThreshold` | string | 0.98 | Flag-head probability required |

## Search

Off by default: the engine plays the raw policy, which is what the paper
evaluates. Search makes it stronger but less human, since it moves play away
from the cloned distribution.

| Option | Type | Default | |
|---|---|---|---|
| `UseMCTS` | check | false | Enable MCTS |
| `MCTSKLRegularized` | check | false | Regularize search toward the policy prior |
| `MCTSKLCoeff` | string | 1.0 | Strength of that regularization |
| `MCTSSimulations` | spin 0-200000 | 256 | Simulations per move |
| `MCTSBatchSize` | spin 1-256 | 16 | Positions evaluated per network call |
| `MCTSMaxChildren` | spin 1-4096 | 16 | Moves expanded per node |
| `MCTSMaxDepth` | spin 1-512 | 24 | Depth limit |
| `MCTSStartPly` | spin 0-100 | 0 | Ply at which search switches on |
| `MCTSCpuct` | string | 1.2 | Exploration constant |
| `MCTSCpuctBase` | string | 19768 | Visit count at which `cpuct` starts growing |
| `MCTSCpuctFactor` | string | 2.0 | How fast `cpuct` grows past that point |
| `MCTSFPU` | string | 0.33 | First-play-urgency value for unvisited moves |
| `MCTSContempt` | string | 0.15 | Draw aversion |
| `MCTSRootDirichletAlpha` | string | 0.0 | Dirichlet noise at the root |
| `MCTSRootExplorationFrac` | string | 0.0 | Weight of that noise |
| `MCTSFinalTemperature` | string | 0.0 | Temperature over final visit counts |
| `MCTSFinalTopP` | string | 0.95 | Nucleus cutoff over final visit counts |
| `MCTSAdaptive` | check | false | Scale simulations with position complexity |
| `MCTSAdaptiveScale` | string | 30.0 | Strength of that scaling |
| `MCTSSimulateTime` | check | false | Budget search by the predicted think time |
| `MCTSTreeReuse` | check | true | Keep the subtree from the previous move |

## Diagnostics

| Option | Type | Default | |
|---|---|---|---|
| `DebugClocks` | check | false | Log the temporal inputs each move |
| `LogTimeHistory` | check | false | Log the tempo history |
| `LogResignProbs` | check | false | Log resignation-head output |
| `LogMctsStats` | check | false | Log search statistics |
| `Ponder` | check | false | Declared for GUI compatibility |
