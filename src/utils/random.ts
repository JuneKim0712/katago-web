export interface RandomSource {
  next(): number;
}

export const DefaultRandomSource: RandomSource = {
  next: () => Math.random(),
};

/**
 * Deterministic finite random tape for replay and exact parity qualification.
 */
export class RandomTape implements RandomSource {
  private readonly values: Float64Array | number[];
  private index = 0;

  constructor(values: Float64Array | number[]) {
    this.values = values;
  }

  next(): number {
    if (this.index >= this.values.length) {
      throw new Error(`RandomTape exhausted at index ${this.index} (total length: ${this.values.length})`);
    }
    return this.values[this.index++];
  }

  get consumedCount(): number {
    return this.index;
  }

  get remainingCount(): number {
    return this.values.length - this.index;
  }

  reset(): void {
    this.index = 0;
  }
}

/**
 * Recording proxy that intercepts calls to an underlying RandomSource and records the tape.
 */
export class RecordingRandomSource implements RandomSource {
  readonly tape: number[] = [];

  constructor(private readonly source: RandomSource = DefaultRandomSource) {}

  next(): number {
    const val = this.source.next();
    this.tape.push(val);
    return val;
  }

  toRandomTape(): RandomTape {
    return new RandomTape(this.tape);
  }
}

/**
 * Seedable lightweight linear congruential / splitmix PRNG for test tape generation.
 */
export function createSeededRandomSource(seed: number): RandomSource {
  let s = (seed >>> 0) || 1;
  return {
    next(): number {
      s = (s + 0x9e3779b9) | 0;
      let z = s;
      z = Math.imul(z ^ (z >>> 16), 0x21f0aaad);
      z = Math.imul(z ^ (z >>> 15), 0x735a2d97);
      z = z ^ (z >>> 15);
      return (z >>> 0) / 4294967296;
    },
  };
}

export class Rand {
  private spare: number | null = null;
  private readonly source: RandomSource;

  constructor(source: RandomSource = DefaultRandomSource) {
    this.source = source;
  }

  nextBool(p: number): boolean {
    return this.source.next() < p;
  }

  nextGaussian(): number {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return v;
    }

    let u = 0;
    let v = 0;
    let s = 0;
    while (s === 0 || s >= 1) {
      u = this.source.next() * 2 - 1;
      v = this.source.next() * 2 - 1;
      s = u * u + v * v;
    }
    const mul = Math.sqrt((-2 * Math.log(s)) / s);
    this.spare = v * mul;
    return u * mul;
  }
}
