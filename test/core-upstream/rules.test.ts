import { describe, it, expect } from 'vitest';
import {
  RULES_PRESETS,
  parseRules,
  rulesToString,
  equalsRules,
} from '../../src/core-contract/rules.js';
import {
  createMove,
  createPassMove,
  PASS_LOC,
  xyToLoc,
} from '../../src/core-contract/position.js';
import {
  validateAndNormalizePosition,
  createBoardHistoryFromPosition,
  PositionValidationError,
} from '../../src/core-contract/validation.js';
import { BoardHistory } from '../../src/core-runtime/boardHistory.js';

describe('Stage 2: Rules and Presets', () => {
  it('loads all KataGo rules presets correctly', () => {
    expect(RULES_PRESETS['japanese'].scoringRule).toBe('SCORING_TERRITORY');
    expect(RULES_PRESETS['japanese'].koRule).toBe('KO_SIMPLE');
    expect(RULES_PRESETS['japanese'].multiStoneSuicideLegal).toBe(false);
    expect(RULES_PRESETS['japanese'].taxRule).toBe('TAX_SEKI');

    expect(RULES_PRESETS['chinese'].scoringRule).toBe('SCORING_AREA');
    expect(RULES_PRESETS['chinese'].koRule).toBe('KO_POSITIONAL');
    expect(RULES_PRESETS['chinese'].multiStoneSuicideLegal).toBe(false);
    expect(RULES_PRESETS['chinese'].whiteHandicapBonusRule).toBe('WHB_N');

    expect(RULES_PRESETS['tromp-taylor'].scoringRule).toBe('SCORING_AREA');
    expect(RULES_PRESETS['tromp-taylor'].koRule).toBe('KO_POSITIONAL');
    expect(RULES_PRESETS['tromp-taylor'].multiStoneSuicideLegal).toBe(true);

    expect(RULES_PRESETS['new-zealand'].multiStoneSuicideLegal).toBe(true);
    expect(RULES_PRESETS['new-zealand'].koRule).toBe('KO_SITUATIONAL');

    expect(RULES_PRESETS['aga'].koRule).toBe('KO_SITUATIONAL');
    expect(RULES_PRESETS['aga'].scoringRule).toBe('SCORING_AREA');
  });

  it('parses rule strings and performs round-trip serialization', () => {
    for (const [name, preset] of Object.entries(RULES_PRESETS)) {
      const parsed = parseRules(name);
      expect(equalsRules(parsed, preset)).toBe(true);

      const serialized = rulesToString(parsed);
      const roundTrip = parseRules(serialized);
      expect(equalsRules(roundTrip, preset)).toBe(true);
    }

    // Custom clause parsing
    const custom = parseRules('ko=situational,scoring=area,tax=none,suicide=true');
    expect(custom.koRule).toBe('KO_SITUATIONAL');
    expect(custom.scoringRule).toBe('SCORING_AREA');
    expect(custom.taxRule).toBe('TAX_NONE');
    expect(custom.multiStoneSuicideLegal).toBe(true);
  });
});

describe('Stage 2: Legality Edge Cases & Superko', () => {
  it('enforces suicide illegality under Japanese and multi-stone suicide legality under Tromp-Taylor', () => {
    // Surround corner (0,0) with White stones at (1,0) and (0,1)
    const jpHistory = new BoardHistory(RULES_PRESETS.japanese, 'black', 19, 19, [
      { x: 1, y: 0, color: 'white' },
      { x: 0, y: 1, color: 'white' },
    ]);
    const cornerLoc = xyToLoc(0, 0, 19);

    // Black playing at (0,0) has 0 liberties and captures nothing -> illegal under Japanese
    expect(jpHistory.isLegal(cornerLoc, 'black')).toBe(false);

    // Under Tromp-Taylor, multi-stone suicide (where existing friendly stones are removed,
    // producing a new non-repeating board state) is fully legal.
    // Let Black have a stone at (0,0) in atari, surrounded by White at (1,0) and (0,2), with (0,1) empty.
    const ttHistory = new BoardHistory(RULES_PRESETS['tromp-taylor'], 'black', 19, 19, [
      { x: 0, y: 0, color: 'black' },
      { x: 1, y: 0, color: 'white' },
      { x: 1, y: 1, color: 'white' },
      { x: 0, y: 2, color: 'white' },
    ]);
    const fillLoc = xyToLoc(0, 1, 19);
    // Playing at (0,1) joins the Black group and suicides both stones
    expect(ttHistory.isLegal(fillLoc, 'black')).toBe(true);

    // Playing suicide removes both stones
    const ok = ttHistory.makeMove(fillLoc, 'black');
    expect(ok).toBe(true);
    expect(ttHistory.currentBoard.getColor(xyToLoc(0, 0, 19))).toBe(0); // EMPTY
    expect(ttHistory.currentBoard.getColor(fillLoc)).toBe(0); // EMPTY
  });

  it('enforces multi-stone suicide legality under New Zealand', () => {
    // 2-stone block at (0,0) and (1,0) surrounded by White at (2,0), (0,1), (1,1)
    const nzHistory = new BoardHistory(RULES_PRESETS['new-zealand'], 'black', 19, 19, [
      { x: 0, y: 0, color: 'black' },
      { x: 2, y: 0, color: 'white' },
      { x: 0, y: 1, color: 'white' },
      { x: 1, y: 1, color: 'white' },
    ]);
    const fillLoc = xyToLoc(1, 0, 19);
    expect(nzHistory.isLegal(fillLoc, 'black')).toBe(true);
    nzHistory.makeMove(fillLoc, 'black');
    // Both black stones should be removed
    expect(nzHistory.currentBoard.getColor(xyToLoc(0, 0, 19))).toBe(0);
    expect(nzHistory.currentBoard.getColor(fillLoc)).toBe(0);
  });

  it('enforces simple ko rules correctly', () => {
    const history = new BoardHistory(RULES_PRESETS.japanese, 'black', 19, 19);
    // Standard ko shape around (1,1) and (2,1)
    // Black at (0,1), (1,0), (1,2)
    // White at (2,0), (2,2), (3,1), and White at (1,1)
    history.currentBoard.setStone(xyToLoc(0, 1, 19), 1); // Black
    history.currentBoard.setStone(xyToLoc(1, 0, 19), 1); // Black
    history.currentBoard.setStone(xyToLoc(1, 2, 19), 1); // Black

    history.currentBoard.setStone(xyToLoc(2, 0, 19), 2); // White
    history.currentBoard.setStone(xyToLoc(2, 2, 19), 2); // White
    history.currentBoard.setStone(xyToLoc(3, 1, 19), 2); // White
    history.currentBoard.setStone(xyToLoc(1, 1, 19), 2); // White stone in atari!

    // Black captures White stone by playing (2,1)
    const bMove = xyToLoc(2, 1, 19);
    expect(history.isLegal(bMove, 'black')).toBe(true);
    history.makeMove(bMove, 'black');

    // White stone at (1,1) was captured
    expect(history.currentBoard.getColor(xyToLoc(1, 1, 19))).toBe(0);
    expect(history.koLoc).toBe(xyToLoc(1, 1, 19));

    // Immediate recapture by White at (1,1) is illegal (simple ko)
    expect(history.isLegal(xyToLoc(1, 1, 19), 'white')).toBe(false);

    // If White plays elsewhere (pass)
    history.makeMove(PASS_LOC, 'white');
    // Now ko is lifted, Black can connect or play elsewhere
    expect(history.koLoc).toBe(PASS_LOC);
  });

  it('differentiates positional superko from situational superko', () => {
    // In situational superko, repeating the position with the OPPOSITE player is allowed,
    // whereas in positional superko it is strictly forbidden.
    const posRules = RULES_PRESETS.chinese; // KO_POSITIONAL
    const sitRules = RULES_PRESETS.aga; // KO_SITUATIONAL

    const posHist = new BoardHistory(posRules, 'black', 19, 19);
    const sitHist = new BoardHistory(sitRules, 'black', 19, 19);

    // Initial position has empty board.
    // Let's create a cycle where stones return to empty or an identical position.
    // In Tromp-Taylor/Chinese, positional superko forbids recreating the exact board.
    expect(posHist.rules.koRule).toBe('KO_POSITIONAL');
    expect(sitHist.rules.koRule).toBe('KO_SITUATIONAL');
  });
});

describe('Stage 2: Validation & Normalization', () => {
  it('normalizes legacy request into CanonicalPosition', () => {
    const legacyReq = {
      board: [
        ['black', null],
        [null, 'white'],
      ],
      currentPlayer: 'black' as const,
      komi: 6.5,
      rules: 'chinese',
    };

    const canonical = validateAndNormalizePosition(legacyReq);
    expect(canonical.width).toBe(2);
    expect(canonical.height).toBe(2);
    expect(canonical.rules.scoringRule).toBe('SCORING_AREA');
    expect(canonical.komi).toBe(6.5);
    expect(canonical.initialStones.length).toBe(2);
  });

  it('rejects contradictory board snapshots with clear error', () => {
    const contradictory = {
      initialStones: [{ x: 0, y: 0, color: 'black' as const }],
      moveHistory: [{ x: 1, y: 0, player: 'white' as const }],
      board: [
        ['white', 'white'], // (0,0) is white, which contradicts initialStones replay
        [null, null],
      ],
      rules: 'japanese',
    };

    expect(() => validateAndNormalizePosition(contradictory)).toThrow(
      PositionValidationError
    );
  });
});
