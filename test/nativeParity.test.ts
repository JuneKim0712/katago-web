import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { pathToFileURL } from 'node:url';
import * as tf from '@tensorflow/tfjs';
import fixture from './fixtures/katago-model17-native.json';
import {
  parseKataGoModelV8,
  KataGoModelV8Tf,
  MctsSearch,
  playMove,
  type BoardState,
  type GameRules,
  type Move,
  type Player,
} from '../src/index.js';

const modelPath = process.env.KATAGO_TEST_MODEL ?? 'C:/Users/junwo/Downloads/kata1-tf2-b10c384-s2941M-d5872M.bin.gz';

describe.skipIf(!existsSync(modelPath))('native KataGo v1.18.1 raw-output parity', () => {
  let model: KataGoModelV8Tf;
  beforeAll(async () => {
    const compressed = readFileSync(modelPath);
    expect(createHash('sha256').update(compressed).digest('hex')).toBe(fixture.modelSha256);
    if (process.env.KATAGO_WEBGPU_MODULE) {
      const { create, globals } = await import(/* @vite-ignore */ pathToFileURL(process.env.KATAGO_WEBGPU_MODULE).href);
      Object.assign(globalThis, globals);
      Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { gpu: create(['backend=d3d12']) } });
      await import('@tensorflow/tfjs-backend-webgpu');
      if (!await tf.setBackend('webgpu')) throw new Error('WebGPU unavailable');
    } else await tf.setBackend('cpu');
    await tf.ready();
    model = new KataGoModelV8Tf(parseKataGoModelV8(gunzipSync(compressed)), { fuseTransformerProjections: process.env.KATAGO_FUSED_PROJECTIONS === '1' });
  });
  afterAll(() => {
    model?.dispose();
    if (process.env.KATAGO_WEBGPU_MODULE) {
      tf.removeBackend('webgpu');
      Reflect.deleteProperty(globalThis, 'navigator');
    }
  });

  for (const position of fixture.cases) {
    it(position.id, async () => {
      const state = { stones: new Uint8Array(361), koPoint: -1 };
      const snapshot = (): BoardState => Array.from({ length: 19 }, (_, y) => Array.from({ length: 19 }, (_, x) => {
        const stone = state.stones[y * 19 + x];
        return stone === 1 ? 'black' : stone === 2 ? 'white' : null;
      }));
      const history: Move[] = [], boards: BoardState[] = [snapshot()];
      for (const [player, coordinate] of position.moves) {
        const pass = coordinate === 'pass';
        const x = pass ? -1 : 'ABCDEFGHJKLMNOPQRST'.indexOf(coordinate[0]), y = pass ? -1 : 19 - Number(coordinate.slice(1));
        playMove(state, pass ? 361 : y * 19 + x, player === 'black' ? 1 : 2, []);
        history.push({ x, y, player: player as Player });
        boards.push(snapshot());
      }
      const search = await MctsSearch.create({
        model, board: boards.at(-1)!, previousBoard: boards.at(-2), previousPreviousBoard: boards.at(-3), previousPreviousPreviousBoard: boards.at(-4),
        currentPlayer: position.currentPlayer as Player, moveHistory: history, koPoint: state.koPoint,
        komi: position.komi, rules: position.rules as GameRules, nnRandomize: false, conservativePass: false,
        maxChildren: 361, ownershipMode: 'root', wideRootNoise: 0, rootSymmetrySamples: 1,
      });
      // No search iterations: compare the production feature / network / postprocess path.
      const actual = search.getAnalysis({ topK: 1, analysisPvLen: 1, cloneBuffers: false });
      expect(Math.abs(actual.rootWinRate - position.native.whiteLoss), 'black win probability').toBeLessThan(0.0002);
      expect(Math.abs(actual.rootScoreLead + position.native.whiteLead), 'score lead').toBeLessThan(0.01);
      expect(Math.abs(actual.rootScoreSelfplay + position.native.whiteScoreSelfplay), 'score mean').toBeLessThan(0.01);
      const expectedPolicy = [...position.native.policy, position.native.policyPass];
      let policyError = 0, ownershipError = 0;
      for (let i = 0; i < 362; i++) {
        if (expectedPolicy[i] === null) expect(actual.policy[i], `illegal move ${i}`).toBeLessThan(0);
        else policyError = Math.max(policyError, Math.abs(actual.policy[i] - expectedPolicy[i]!));
        if (i < 361) ownershipError = Math.max(ownershipError, Math.abs(actual.ownership[i] + position.native.whiteOwnership[i]));
      }
      expect(policyError, 'policy max absolute error').toBeLessThan(0.0003);
      expect(ownershipError, 'ownership max absolute error').toBeLessThan(0.002);
    }, 60000);
  }
});
