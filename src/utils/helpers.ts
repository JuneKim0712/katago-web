import type { Player } from '../types/index.js';

export const getOpponent = (player: Player): Player => (player === 'black' ? 'white' : 'black');

export const getAnimationNow = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());
