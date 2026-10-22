import { describe, it, expect } from 'vitest';
import {
  extractFeatures,
  NUM_SPATIAL_CHANNELS,
  NUM_GLOBAL_CHANNELS,
  applySymmetry,
} from '../../src/core-runtime/features.js';
import { BoardHistory } from '../../src/core-runtime/boardHistory.js';
import { RULES_PRESETS } from '../../src/core-contract/rules.js';
import { xyToLoc, PASS_LOC } from '../../src/core-contract/position.js';

describe('Stage 3: Feature Generation', () => {
  it('generates spatial and global tensors with exact expected dimensions', () => {
    const history = new BoardHistory(RULES_PRESETS.japanese, 'black', 19, 19);
    const features = extractFeatures(history, 6.5, { layout: 'NHWC' });

    expect(features.spatial.length).toBe(19 * 19 * NUM_SPATIAL_CHANNELS);
    expect(features.global.length).toBe(NUM_GLOBAL_CHANNELS);
    expect(features.layout).toBe('NHWC');

    // Plane 0 must be 1.0 everywhere
    for (let pos = 0; pos < 19 * 19; pos++) {
      expect(features.spatial[pos * NUM_SPATIAL_CHANNELS + 0]).toBe(1.0);
    }
  });

  it('correctly populates stone planes and liberty planes', () => {
    // Place a single Black stone at (3,3) -> has 4 liberties
    // Place a single White stone at (0,0) -> in corner, has 2 liberties
    const history = new BoardHistory(RULES_PRESETS.japanese, 'black', 19, 19, [
      { x: 3, y: 3, color: 'black' },
      { x: 0, y: 0, color: 'white' },
    ]);

    const features = extractFeatures(history, 6.5, { layout: 'NHWC' });

    const bIdx = (3 * 19 + 3) * NUM_SPATIAL_CHANNELS;
    const wIdx = (0 * 19 + 0) * NUM_SPATIAL_CHANNELS;

    // Current player is black
    expect(features.spatial[bIdx + 1]).toBe(1.0); // friendly stone
    expect(features.spatial[bIdx + 2]).toBe(0.0); // opponent stone

    expect(features.spatial[wIdx + 1]).toBe(0.0); // friendly stone
    expect(features.spatial[wIdx + 2]).toBe(1.0); // opponent stone
    expect(features.spatial[wIdx + 4]).toBe(1.0); // 2 liberties! (corner)
  });

  it('correctly encodes untruncated move history and ko locations into planes 6 and 9..13', () => {
    const history = new BoardHistory(RULES_PRESETS.japanese, 'black', 19, 19);

    // Play a sequence of moves
    history.makeMove(xyToLoc(3, 3, 19), 'black');
    history.makeMove(xyToLoc(15, 15, 19), 'white');
    history.makeMove(xyToLoc(3, 15, 19), 'black');

    // Now current player is White.
    // Last move was Black at (3, 15) -> plane 9
    // 2 moves ago was White at (15, 15) -> plane 10
    // 3 moves ago was Black at (3, 3) -> plane 11
    const features = extractFeatures(history, 6.5, { layout: 'NHWC' });

    const m1Idx = (15 * 19 + 3) * NUM_SPATIAL_CHANNELS;
    const m2Idx = (15 * 19 + 15) * NUM_SPATIAL_CHANNELS;
    const m3Idx = (3 * 19 + 3) * NUM_SPATIAL_CHANNELS;

    expect(features.spatial[m1Idx + 9]).toBe(1.0); // 1 step ago
    expect(features.spatial[m2Idx + 10]).toBe(1.0); // 2 steps ago
    expect(features.spatial[m3Idx + 11]).toBe(1.0); // 3 steps ago
  });

  it('correctly maps both NHWC and NCHW layouts', () => {
    const history = new BoardHistory(RULES_PRESETS.japanese, 'black', 19, 19, [
      { x: 5, y: 5, color: 'black' },
    ]);

    const nhwc = extractFeatures(history, 7.5, { layout: 'NHWC' });
    const nchw = extractFeatures(history, 7.5, { layout: 'NCHW' });

    // For stone at (5,5), friendly stone is in channel 1
    const nhwcVal = nhwc.spatial[(5 * 19 + 5) * NUM_SPATIAL_CHANNELS + 1];
    const nchwVal = nchw.spatial[(1 * 19 + 5) * 19 + 5];

    expect(nhwcVal).toBe(1.0);
    expect(nchwVal).toBe(1.0);
  });

  it('correctly applies board symmetries', () => {
    const [fx, fy] = applySymmetry(2, 3, 1, 19, 19); // flip X
    expect(fx).toBe(19 - 1 - 2);
    expect(fy).toBe(3);

    const [tx, ty] = applySymmetry(2, 3, 4, 19, 19); // transpose
    expect(tx).toBe(3);
    expect(ty).toBe(2);
  });

  it('populates global features with correct rule and komi indicators', () => {
    const historyJp = new BoardHistory(RULES_PRESETS.japanese, 'black', 19, 19);
    const featJp = extractFeatures(historyJp, 6.5);
    // Black to move with komi 6.5 -> selfKomi = -6.5 -> global[5] = -6.5 / 20.0
    expect(featJp.global[5]).toBeCloseTo(-6.5 / 20.0);
    expect(featJp.global[9]).toBe(1.0); // territory scoring
    expect(featJp.global[10]).toBe(1.0); // seki tax

    const historyCh = new BoardHistory(RULES_PRESETS.chinese, 'white', 19, 19);
    const featCh = extractFeatures(historyCh, 7.5);
    // White to move with komi 7.5 -> selfKomi = 7.5 -> global[5] = 7.5 / 20.0
    expect(featCh.global[5]).toBeCloseTo(7.5 / 20.0);
    expect(featCh.global[9]).toBe(0.0); // area scoring
    expect(featCh.global[12]).toBe(1.0); // WHB_N
  });
});
