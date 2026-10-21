import { getAnimationNow } from './helpers.js';

export type ClockSource = () => number;

export const DefaultClockSource: ClockSource = getAnimationNow;

/**
 * Scripted clock for deterministic deadline / cancellation qualification.
 */
export class ScriptedClock {
  private time: number;

  constructor(initialTime = 0) {
    this.time = initialTime;
  }

  now = (): number => this.time;

  advance(deltaMs: number): void {
    this.time += deltaMs;
  }

  set(newTime: number): void {
    this.time = newTime;
  }
}
