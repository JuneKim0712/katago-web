# KataGo Upstream Core Provenance

## Pinned Upstream Reference
- **Upstream Repository**: [lightvector/KataGo](https://github.com/lightvector/KataGo)
- **Pinned Version / Tag**: `v1.18.1`
- **Release Reference Commit**: `4f8849b77526a798544d673f8e65839ff9be83be`
- **License**: MIT License (David J. Wu / Lightvector)

## Algorithmic References
The modules in `src/core-contract/` and `src/core-runtime/` conform directly to the upstream C++ implementation in KataGo v1.18.1:

1. **Board Logic & Liberties**:
   - `cpp/game/board.h` & `cpp/game/board.cpp`
   - Strict liberty counting, multi-stone group tracking, suicide rule handling.

2. **Rulesets & Enums**:
   - `cpp/game/rules.h` & `cpp/game/rules.cpp`
   - Ko rules: `KO_SIMPLE`, `KO_POSITIONAL`, `KO_SITUATIONAL`.
   - Scoring rules: `SCORING_AREA`, `SCORING_TERRITORY`.
   - Tax rules: `TAX_NONE`, `TAX_SEKI`, `TAX_ALL`.
   - Multi-stone suicide flag: `hasButton`, `suicideLegal`.
   - Presets:
     - `Tromp-Taylor`: positional superko, suicide legal, area scoring, tax none.
     - `Chinese`: positional superko, suicide illegal, area scoring, tax none.
     - `Japanese`: simple ko, suicide illegal, territory scoring, tax seki.
     - `Korean`: simple ko, suicide illegal, territory scoring, tax seki.
     - `AGA`: situational superko, suicide illegal, area scoring with pass stones.

3. **Game State & Board History**:
   - `cpp/game/boardhistory.h` & `cpp/game/boardhistory.cpp`
   - Canonical move history preservation (untruncated).
   - Zobrist position hashing for situational and positional superko.
   - Consecutive pass tracking and terminal state detection.
   - Initial board snapshot + consecutive moves.

4. **Neural Network Input Features**:
   - `cpp/neuralnet/nninputs.h` & `cpp/neuralnet/nninputs.cpp`
   - Input format: 22 spatial channels (size $H \times W \times 22$ or $22 \times H \times W$), 19 global features.
   - Spatial channels:
     - 0: 1.0 everywhere (constant plane)
     - 1: Current player's stones
     - 2: Opponent's stones
     - 3: 1-liberty stones (current player)
     - 4: 2-liberty stones (current player)
     - 5: 3-liberty stones (current player)
     - 6: 1-liberty stones (opponent)
     - 7: 2-liberty stones (opponent)
     - 8: 3-liberty stones (opponent)
     - 9-13: History planes (moves 1 step ago, 2 steps ago, 3, 4, 5 steps ago)
     - 14: Ko legality / illegal moves (suicide / ko / superko restrictions for current player)
     - 15: Opponent illegal moves
     - 16-21: Additional spatial cues (e.g. ladder / territory / pass-alive status if enabled)
   - Global features (19 scalars):
     - Komi, ruleset flags, handicap, pass history, capture counts, self-play / playout counts.

5. **MCTS & Search Execution**:
   - `cpp/search/searchpuct.cpp` & `cpp/search/searchnnhelpers.cpp`
   - Upstream PUCT formula with parent visit weighting and dynamic FPU (first play urgency).
   - Backend-neutral evaluation interface decoupling search tree mechanics from tensor execution runtime.
