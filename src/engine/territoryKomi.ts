import type { GameRules } from '../types/index.js';
import { BLACK, PASS_MOVE, WHITE } from './fastBoard.js';
import type { RecentMove } from './featuresV7Fast';

/** Native BoardHistory::whiteBonusScore, normal play (not Japanese encore).
 * Count moves played, including captured stones, rather than stones remaining.
 * Full history is required; a history-free setup uses its starting stones.
 */
export function getNNKomi(
  komi: number,
  rules: GameRules,
  moves: readonly RecentMove[],
  setupStones?: Uint8Array,
  movesCount?: number
): number {
  if (rules === 'chinese') return komi;
  let bonus = 0;
  if (setupStones) for (const stone of setupStones) bonus += stone === BLACK ? 1 : stone === WHITE ? -1 : 0;
  const count = movesCount ?? moves.length;
  for (let i = 0; i < count; i++) {
    const move = moves[i]!;
    if (move.move !== PASS_MOVE) bonus += move.player === 'black' ? 1 : -1;
  }
  return komi + bonus;
}
