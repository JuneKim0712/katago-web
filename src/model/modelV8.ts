import * as tf from '@tensorflow/tfjs';
import { projectedRotaryWebGpu } from './rotaryWebGpu.js';
import { transformerRmsWebGpu } from './rmsNormWebGpu.js';
import { BOARD_SIZE } from '../engine/fastBoard.js';
import type {
  ActivationKind,
  ParsedBatchNorm,
  ParsedConv2d,
  ParsedMatBias,
  ParsedMatMul,
  ParsedRMSNormLayer,
  ParsedTransformerRMSNorm,
  ParsedTransformerAttention,
  ParsedTransformerFFN,
} from './binModelParser.js';

const INPUT_CHANNELS = 22;
const GLOBAL_FEATURES = 19;

async function writeTensorData(tensor: tf.Tensor, data: Float32Array): Promise<void> {
  const writer = (tensor as tf.Tensor & { write?: (values: Float32Array) => Promise<void> }).write;
  if (typeof writer === 'function') {
    await writer.call(tensor, data);
    return;
  }
  throw new Error('Tensor.write is unavailable on this TF.js backend');
}

/** Reusable spatial/global input tensors to avoid per-batch GPU allocation. */
export class ModelInputBuffers {
  get supportsWrites(): boolean {
    return typeof (tf.Tensor.prototype as tf.Tensor & { write?: unknown }).write === 'function';
  }
  private spatial: tf.Tensor4D | null = null;
  private global: tf.Tensor2D | null = null;
  private capacity = 0;

  async uploadBatch(
    spatialData: Float32Array,
    globalData: Float32Array,
    batch: number
  ): Promise<{ spatial: tf.Tensor4D; global: tf.Tensor2D; disposeSlice: boolean }> {
    if (!this.spatial || this.capacity !== batch) {
      this.dispose();
      this.capacity = batch;
      this.spatial = tf.zeros([batch, BOARD_SIZE, BOARD_SIZE, INPUT_CHANNELS]) as tf.Tensor4D;
      this.global = tf.zeros([batch, GLOBAL_FEATURES]) as tf.Tensor2D;
    }

    const spatial = this.spatial!;
    const global = this.global!;
    const spatialBytes = batch * BOARD_SIZE * BOARD_SIZE * INPUT_CHANNELS;
    const globalBytes = batch * GLOBAL_FEATURES;
    await writeTensorData(spatial, spatialData.subarray(0, spatialBytes));
    await writeTensorData(global, globalData.subarray(0, globalBytes));

    return { spatial, global, disposeSlice: false };
  }

  dispose(): void {
    this.spatial?.dispose();
    this.global?.dispose();
    this.spatial = null;
    this.global = null;
    this.capacity = 0;
  }
}

type TfBn = { scale: tf.Tensor4D; bias: tf.Tensor4D };

type TfRMSNormLayer = {
  gamma: tf.Tensor4D;
  beta: tf.Tensor4D;
  epsilon: number;
  spatial: boolean;
};

type TfTransformerRMSNorm = {
  fuse: boolean;
  weight: tf.Tensor4D;
  epsilon: number;
  channels: number;
};

type TfTransformerAttention = {
  fuseRotary: boolean;
  kind: 'transformer_attention';
  numHeads: number;
  numKVHeads: number;
  qHeadDim: number;
  vHeadDim: number;
  preLN: TfTransformerRMSNorm;
  qkv: TfProjectionGroup;
  outProj: TfMatMul;
  rope?: {
    cos: tf.Tensor4D;
    sin: tf.Tensor4D;
  };
  scale: number;
};

type TfTransformerFFN = {
  kind: 'transformer_ffn';
  useSwiGLU: boolean;
  preLN: TfTransformerRMSNorm;
  input: TfProjectionGroup;
  linear2: TfMatMul;
};

type TfConv = {
  kernelY: number;
  kernelX: number;
  inChannels: number;
  outChannels: number;
  dilationY: number;
  dilationX: number;
  filter: tf.Tensor4D;
};

type TfMatMul = {
  inChannels: number;
  outChannels: number;
  w: tf.Tensor2D;
};

type TfProjectionGroup = { layers: TfMatMul[]; sizes: number[]; fused: boolean };

function makeProjectionGroup(layers: ParsedMatMul[], fused: boolean): TfProjectionGroup {
  const sizes = layers.map(layer => layer.outChannels);
  if (!fused || layers.length === 1) return { layers: layers.map(makeMatMul), sizes, fused: false };
  const inChannels = layers[0].inChannels;
  if (layers.some(layer => layer.inChannels !== inChannels)) throw new Error('Projection input dimensions differ');
  const outChannels = sizes.reduce((sum, size) => sum + size, 0);
  const weights = new Float32Array(inChannels * outChannels);
  for (let row = 0; row < inChannels; row++) {
    let column = 0;
    for (const layer of layers) {
      weights.set(layer.weights.subarray(row * layer.outChannels, (row + 1) * layer.outChannels), row * outChannels + column);
      column += layer.outChannels;
    }
  }
  return { layers: [{ inChannels, outChannels, w: tf.tensor2d(weights, [inChannels, outChannels]) }], sizes, fused: true };
}

function projectGroup(input: tf.Tensor2D, group: TfProjectionGroup): tf.Tensor2D[] {
  if (!group.fused) return group.layers.map(layer => tf.matMul(input, layer.w) as tf.Tensor2D);
  return tf.split(tf.matMul(input, group.layers[0].w), group.sizes, 1) as tf.Tensor2D[];
}

type TfMatBias = {
  channels: number;
  b: tf.Tensor2D; // [1,channels]
};

function makeBn(bn: ParsedBatchNorm): TfBn {
  const scale = tf.tensor4d(bn.mergedScale, [1, 1, 1, bn.channels]);
  const bias = tf.tensor4d(bn.mergedBias, [1, 1, 1, bn.channels]);
  return { scale, bias };
}

function makeRMSNormLayer(rn: ParsedRMSNormLayer): TfRMSNormLayer {
  const gamma = tf.tensor4d(rn.gamma, [1, 1, 1, rn.channels]);
  const beta = tf.tensor4d(rn.beta, [1, 1, 1, rn.channels]);
  return { gamma, beta, epsilon: rn.epsilon, spatial: rn.spatial };
}

function makeTransformerRMSNorm(tr: ParsedTransformerRMSNorm, fuse: boolean): TfTransformerRMSNorm {
  const weight = tf.tensor4d(tr.weights, [1, 1, 1, tr.channels]);
  return { weight, epsilon: tr.epsilon, channels: tr.channels, fuse };
}

function precomputeRope(
  attn: ParsedTransformerAttention,
  H = BOARD_SIZE,
  W = BOARD_SIZE
): { cos: tf.Tensor4D; sin: tf.Tensor4D } | undefined {
  if (!attn.useRope) return undefined;
  const numPairs = Math.floor(attn.qHeadDim / 2);
  const S = H * W;

  if (attn.learnableRope && attn.ropeFreqs) {
    const cosTable = new Float32Array(attn.numKVHeads * S * numPairs);
    const sinTable = new Float32Array(attn.numKVHeads * S * numPairs);
    for (let h = 0; h < attn.numKVHeads; h++) {
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const xy = y * W + x;
          for (let p = 0; p < numPairs; p++) {
            const freqX = attn.ropeFreqs[(h * numPairs + p) * 2 + 0];
            const freqY = attn.ropeFreqs[(h * numPairs + p) * 2 + 1];
            const angle = x * freqX + y * freqY;
            const idx = (h * S + xy) * numPairs + p;
            cosTable[idx] = Math.cos(angle);
            sinTable[idx] = Math.sin(angle);
          }
        }
      }
    }
    return {
      cos: tf.tensor4d(cosTable, [1, attn.numKVHeads, S, numPairs]),
      sin: tf.tensor4d(sinTable, [1, attn.numKVHeads, S, numPairs]),
    };
  }

  if (!attn.learnableRope && attn.ropeTheta) {
    const numPairsPerDim = Math.floor(numPairs / 2);
    const dimHalf = Math.floor(attn.qHeadDim / 2);
    const cosTable = new Float32Array(S * numPairs);
    const sinTable = new Float32Array(S * numPairs);

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const xy = y * W + x;
        for (let p = 0; p < numPairs; p++) {
          let angle: number;
          if (p < numPairsPerDim) {
            const freq = 1.0 / Math.pow(attn.ropeTheta, (2 * p) / dimHalf);
            angle = y * freq;
          } else {
            const pAdj = p - numPairsPerDim;
            const freq = 1.0 / Math.pow(attn.ropeTheta, (2 * pAdj) / dimHalf);
            angle = x * freq;
          }
          const idx = xy * numPairs + p;
          cosTable[idx] = Math.cos(angle);
          sinTable[idx] = Math.sin(angle);
        }
      }
    }
    return {
      cos: tf.tensor4d(cosTable, [1, 1, S, numPairs]),
      sin: tf.tensor4d(sinTable, [1, 1, S, numPairs]),
    };
  }

  return undefined;
}

function makeConv(conv: ParsedConv2d): TfConv {
  // File weights are in [kY,kX,inC,outC] which matches tf.conv2d filter format.
  const filter = tf.tensor4d(conv.weights, [conv.kernelY, conv.kernelX, conv.inChannels, conv.outChannels]);
  return {
    kernelY: conv.kernelY,
    kernelX: conv.kernelX,
    inChannels: conv.inChannels,
    outChannels: conv.outChannels,
    dilationY: conv.dilationY,
    dilationX: conv.dilationX,
    filter,
  };
}

function makeMatMul(mm: ParsedMatMul): TfMatMul {
  const w = tf.tensor2d(mm.weights, [mm.inChannels, mm.outChannels]);
  return { inChannels: mm.inChannels, outChannels: mm.outChannels, w };
}

function makeMatBias(bias: ParsedMatBias): TfMatBias {
  const b = tf.tensor2d(bias.weights, [1, bias.channels]);
  return { channels: bias.channels, b };
}

function applyActivation4D(x: tf.Tensor4D, kind: ActivationKind): tf.Tensor4D {
  if (kind === 'identity') return x;
  if (kind === 'relu') return tf.relu(x) as tf.Tensor4D;
  if (kind === 'silu') return tf.mul(x, tf.sigmoid(x)) as tf.Tensor4D;
  // mish: x * tanh(softplus(x))
  return tf.mul(x, tf.tanh(tf.softplus(x))) as tf.Tensor4D;
}

function applyActivation2D(x: tf.Tensor2D, kind: ActivationKind): tf.Tensor2D {
  if (kind === 'identity') return x;
  if (kind === 'relu') return tf.relu(x) as tf.Tensor2D;
  if (kind === 'silu') return tf.mul(x, tf.sigmoid(x)) as tf.Tensor2D;
  return tf.mul(x, tf.tanh(tf.softplus(x))) as tf.Tensor2D;
}

function bnAct(x: tf.Tensor4D, bn: TfBn, activation: ActivationKind): tf.Tensor4D {
  const y = tf.add(tf.mul(x, bn.scale), bn.bias) as tf.Tensor4D;
  return applyActivation4D(y, activation);
}

function rmsNormAct(x: tf.Tensor4D, rn: TfRMSNormLayer, activation: ActivationKind): tf.Tensor4D {
  const meanSq = rn.spatial ? tf.mean(tf.square(x), [1, 2, 3], true) : tf.mean(tf.square(x), -1, true);
  const r = tf.rsqrt(tf.add(meanSq, rn.epsilon));
  const norm = tf.add(tf.mul(tf.mul(x, r), rn.gamma), rn.beta) as tf.Tensor4D;
  return applyActivation4D(norm, activation);
}

function applyTransformerRMSNorm(x: tf.Tensor4D, tr: TfTransformerRMSNorm): tf.Tensor4D {
  if (tr.fuse && x.shape[3] <= 2048 && tf.getBackend() === 'webgpu') return transformerRmsWebGpu(x, tr.weight, tr.epsilon);
  const meanSq = tf.mean(tf.square(x), -1, true);
  const r = tf.rsqrt(tf.add(meanSq, tr.epsilon));
  return tf.mul(tf.mul(x, r), tr.weight) as tf.Tensor4D;
}

function forwardAttention(x: tf.Tensor4D, attn: TfTransformerAttention): tf.Tensor4D {
  const N = x.shape[0];
  const H = x.shape[1];
  const W = x.shape[2];
  const C = x.shape[3];
  const S = H * W;

  const normX = applyTransformerRMSNorm(x, attn.preLN);
  const flat = normX.reshape([N * S, C]) as tf.Tensor2D;
  const [qFlat, kFlat, vFlat] = projectGroup(flat, attn.qkv);

  const fusedRotary = attn.fuseRotary && !!attn.rope && tf.getBackend() === 'webgpu';
  let Q = fusedRotary
    ? projectedRotaryWebGpu(qFlat, attn.rope!.cos, attn.rope!.sin, N, attn.numHeads, S, attn.qHeadDim)
    : qFlat.reshape([N, S, attn.numHeads, attn.qHeadDim]).transpose([0, 2, 1, 3]) as tf.Tensor4D;
  let K = fusedRotary
    ? projectedRotaryWebGpu(kFlat, attn.rope!.cos, attn.rope!.sin, N, attn.numKVHeads, S, attn.qHeadDim)
    : kFlat.reshape([N, S, attn.numKVHeads, attn.qHeadDim]).transpose([0, 2, 1, 3]) as tf.Tensor4D;
  let V = vFlat.reshape([N, S, attn.numKVHeads, attn.vHeadDim]).transpose([0, 2, 1, 3]) as tf.Tensor4D;

  if (attn.rope && !fusedRotary) {
    const numPairs = Math.floor(attn.qHeadDim / 2);

    let cosQ = attn.rope.cos;
    let sinQ = attn.rope.sin;
    if (attn.numHeads !== attn.numKVHeads && cosQ.shape[1] === attn.numKVHeads) {
      const g = Math.floor(attn.numHeads / attn.numKVHeads);
      cosQ = cosQ.reshape([1, attn.numKVHeads, 1, S, numPairs]).tile([1, 1, g, 1, 1]).reshape([1, attn.numHeads, S, numPairs]) as tf.Tensor4D;
      sinQ = sinQ.reshape([1, attn.numKVHeads, 1, S, numPairs]).tile([1, 1, g, 1, 1]).reshape([1, attn.numHeads, S, numPairs]) as tf.Tensor4D;
    }

    const qPair = Q.reshape([N, attn.numHeads, S, numPairs, 2]);
    const q0 = qPair.slice([0, 0, 0, 0, 0], [-1, -1, -1, -1, 1]).reshape([N, attn.numHeads, S, numPairs]);
    const q1 = qPair.slice([0, 0, 0, 0, 1], [-1, -1, -1, -1, 1]).reshape([N, attn.numHeads, S, numPairs]);
    const qRot0 = tf.sub(tf.mul(q0, cosQ), tf.mul(q1, sinQ));
    const qRot1 = tf.add(tf.mul(q0, sinQ), tf.mul(q1, cosQ));
    Q = tf.stack([qRot0, qRot1], -1).reshape([N, attn.numHeads, S, attn.qHeadDim]) as tf.Tensor4D;

    const cosK = attn.rope.cos;
    const sinK = attn.rope.sin;
    const kPair = K.reshape([N, attn.numKVHeads, S, numPairs, 2]);
    const k0 = kPair.slice([0, 0, 0, 0, 0], [-1, -1, -1, -1, 1]).reshape([N, attn.numKVHeads, S, numPairs]);
    const k1 = kPair.slice([0, 0, 0, 0, 1], [-1, -1, -1, -1, 1]).reshape([N, attn.numKVHeads, S, numPairs]);
    const kRot0 = tf.sub(tf.mul(k0, cosK), tf.mul(k1, sinK));
    const kRot1 = tf.add(tf.mul(k0, sinK), tf.mul(k1, cosK));
    K = tf.stack([kRot0, kRot1], -1).reshape([N, attn.numKVHeads, S, attn.qHeadDim]) as tf.Tensor4D;
  }

  if (attn.numHeads !== attn.numKVHeads) {
    const g = Math.floor(attn.numHeads / attn.numKVHeads);
    K = K.reshape([N, attn.numKVHeads, 1, S, attn.qHeadDim]).tile([1, 1, g, 1, 1]).reshape([N, attn.numHeads, S, attn.qHeadDim]) as tf.Tensor4D;
    V = V.reshape([N, attn.numKVHeads, 1, S, attn.vHeadDim]).tile([1, 1, g, 1, 1]).reshape([N, attn.numHeads, S, attn.vHeadDim]) as tf.Tensor4D;
  }

  const scores = tf.matMul(Q, K, false, true).mul(attn.scale);
  const weights = tf.softmax(scores, -1);
  const context = tf.matMul(weights, V);

  const ctxFlat = context.transpose([0, 2, 1, 3]).reshape([N * S, attn.numHeads * attn.vHeadDim]);
  const outFlat = tf.matMul(ctxFlat, attn.outProj.w);
  const out = outFlat.reshape([N, H, W, C]) as tf.Tensor4D;
  return tf.add(x, out) as tf.Tensor4D;
}

function forwardFFN(x: tf.Tensor4D, ffn: TfTransformerFFN): tf.Tensor4D {
  const N = x.shape[0];
  const H = x.shape[1];
  const W = x.shape[2];
  const C = x.shape[3];
  const S = H * W;

  const normX = applyTransformerRMSNorm(x, ffn.preLN);
  const flat = normX.reshape([N * S, C]) as tf.Tensor2D;
  const [linear, gate] = projectGroup(flat, ffn.input);
  let mid = linear;
  mid = tf.mul(mid, tf.sigmoid(mid)); // SiLU(linear1)
  if (ffn.useSwiGLU && gate) {
    mid = tf.mul(mid, gate);
  }

  const outFlat = tf.matMul(mid, ffn.linear2.w);
  const out = outFlat.reshape([N, H, W, C]) as tf.Tensor4D;
  return tf.add(x, out) as tf.Tensor4D;
}

function conv2d(x: tf.Tensor4D, conv: TfConv): tf.Tensor4D {
  return tf.conv2d(x, conv.filter, 1, 'same', 'NHWC', [conv.dilationY, conv.dilationX]) as tf.Tensor4D;
}

function poolRowsGPool(x: tf.Tensor4D): tf.Tensor2D {
  // KataGo gpool: concat(mean, mean * (sqrt(div)-14)*0.1, max).
  const boardSize = x.shape[1] ?? 19;
  const factor = (boardSize - 14) * 0.1;
  const mean = tf.mean(x, [1, 2]) as tf.Tensor2D; // [N,C]
  const max = tf.max(x, [1, 2]) as tf.Tensor2D; // [N,C]
  return tf.concat([mean, mean.mul(factor), max], 1) as tf.Tensor2D;
}

function poolRowsValueHead(x: tf.Tensor4D): tf.Tensor2D {
  // KataGo value pooling: concat(mean, mean * (sqrt(div)-14)*0.1, mean * (((sqrt(div)-14)^2)*0.01 - 0.1)).
  const boardSize = x.shape[1] ?? 19;
  const base = boardSize - 14;
  const factor1 = base * 0.1;
  const factor2 = base * base * 0.01 - 0.1;
  const mean = tf.mean(x, [1, 2]) as tf.Tensor2D; // [N,C]
  return tf.concat([mean, mean.mul(factor1), mean.mul(factor2)], 1) as tf.Tensor2D;
}

export type ParsedTrunkBlock =
  | {
      kind: 'ordinary';
      preBN: ParsedBatchNorm;
      preActivation: ActivationKind;
      w1: ParsedConv2d;
      midBN: ParsedBatchNorm;
      midActivation: ActivationKind;
      w2: ParsedConv2d;
    }
  | {
      kind: 'gpool';
      preBN: ParsedBatchNorm;
      preActivation: ActivationKind;
      w1a: ParsedConv2d;
      w1b: ParsedConv2d;
      gpoolBN: ParsedBatchNorm;
      gpoolActivation: ActivationKind;
      w1r: ParsedMatMul;
      midBN: ParsedBatchNorm;
      midActivation: ActivationKind;
      w2: ParsedConv2d;
    }
  | {
      kind: 'nested_bottleneck';
      numBlocks: number;
      preBN: ParsedBatchNorm;
      preActivation: ActivationKind;
      preConv: ParsedConv2d;
      blocks: ParsedTrunkBlock[];
      postBN: ParsedBatchNorm;
      postActivation: ActivationKind;
      postConv: ParsedConv2d;
    }
  | ParsedTransformerAttention
  | ParsedTransformerFFN;

export type ParsedKataGoModelV8 = {
  modelName: string;
  modelVersion: number;
  numInputChannels: number;
  numInputGlobalChannels: number;
  metaEncoderVersion: number;
  postProcessParams: {
    tdScoreMultiplier: number;
    scoreMeanMultiplier: number;
    scoreStdevMultiplier: number;
    leadMultiplier: number;
    varianceTimeMultiplier: number;
    shorttermValueErrorMultiplier: number;
    shorttermScoreErrorMultiplier: number;
    outputScaleMultiplier: number;
  };
  policyOutChannels: number;
  scoreValueChannels: number;
  trunk: {
    numBlocks: number;
    trunkNumChannels: number;
    midNumChannels: number;
    regularNumChannels: number;
    gpoolNumChannels: number;
    conv1: ParsedConv2d;
    ginput: ParsedMatMul;
    blocks: ParsedTrunkBlock[];
    tipBN?: ParsedBatchNorm;
    tipRMSNorm?: ParsedRMSNormLayer;
    tipActivation: ActivationKind;
  };
  policy: {
    p1: ParsedConv2d;
    g1: ParsedConv2d;
    g1BN: ParsedBatchNorm;
    g1Activation: ActivationKind;
    gpoolToBias: ParsedMatMul;
    p1BN: ParsedBatchNorm;
    p1Activation: ActivationKind;
    p2: ParsedConv2d;
    passMul: ParsedMatMul;
    passBias?: ParsedMatBias;
    passActivation?: ActivationKind;
    passMul2?: ParsedMatMul;
  };
  value: {
    v1: ParsedConv2d;
    v1BN: ParsedBatchNorm;
    v1Activation: ActivationKind;
    v2: ParsedMatMul;
    v2Bias: ParsedMatBias;
    v2Activation: ActivationKind;
    v3: ParsedMatMul;
    v3Bias: ParsedMatBias;
    sv3: ParsedMatMul;
    sv3Bias: ParsedMatBias;
    ownership: ParsedConv2d;
  };
};

type TfTrunkBlock =
  | {
      kind: 'ordinary';
      preBN: TfBn;
      preActivation: ActivationKind;
      w1: TfConv;
      midBN: TfBn;
      midActivation: ActivationKind;
      w2: TfConv;
    }
  | {
      kind: 'gpool';
      preBN: TfBn;
      preActivation: ActivationKind;
      w1a: TfConv;
      w1b: TfConv;
      gpoolBN: TfBn;
      gpoolActivation: ActivationKind;
      w1r: TfMatMul;
      midBN: TfBn;
      midActivation: ActivationKind;
      w2: TfConv;
    }
  | {
      kind: 'nested_bottleneck';
      numBlocks: number;
      preBN: TfBn;
      preActivation: ActivationKind;
      preConv: TfConv;
      blocks: TfTrunkBlock[];
      postBN: TfBn;
      postActivation: ActivationKind;
      postConv: TfConv;
    }
  | TfTransformerAttention
  | TfTransformerFFN;

export class KataGoModelV8Tf {
  readonly modelName: string;
  readonly modelVersion: number;
  readonly postProcessParams: ParsedKataGoModelV8['postProcessParams'];
  readonly policyOutChannels: number;
  readonly scoreValueChannels: number;
  readonly inputBuffers = new ModelInputBuffers();

  private readonly trunkConv1: TfConv;
  private readonly trunkGInput: TfMatMul;
  private readonly trunkBlocks: TfTrunkBlock[];
  private readonly trunkTipBN?: TfBn;
  private readonly trunkTipRMSNorm?: TfRMSNormLayer;
  private readonly trunkTipActivation: ActivationKind;

  private readonly p1: TfConv;
  private readonly g1: TfConv;
  private readonly g1BN: TfBn;
  private readonly g1Activation: ActivationKind;
  private readonly gpoolToBias: TfMatMul;
  private readonly p1BN: TfBn;
  private readonly p1Activation: ActivationKind;
  private readonly p2: TfConv;
  private readonly passMul: TfMatMul;
  private readonly passBias?: TfMatBias;
  private readonly passActivation?: ActivationKind;
  private readonly passMul2?: TfMatMul;

  private readonly v1: TfConv;
  private readonly v1BN: TfBn;
  private readonly v1Activation: ActivationKind;
  private readonly v2: TfMatMul;
  private readonly v2Bias: TfMatBias;
  private readonly v2Activation: ActivationKind;
  private readonly v3: TfMatMul;
  private readonly v3Bias: TfMatBias;
  private readonly sv3: TfMatMul;
  private readonly sv3Bias: TfMatBias;
  private readonly ownership: TfConv;

  constructor(parsed: ParsedKataGoModelV8, options: { fuseTransformerProjections?: boolean; fuseRotary?: boolean; fuseRmsNorm?: boolean } = {}) {
    const fuseProjections = options.fuseTransformerProjections ?? false;
    this.modelName = parsed.modelName;
    this.modelVersion = parsed.modelVersion;
    this.postProcessParams = parsed.postProcessParams;
    this.policyOutChannels = parsed.policyOutChannels;
    this.scoreValueChannels = parsed.scoreValueChannels;

    this.trunkConv1 = makeConv(parsed.trunk.conv1);
    this.trunkGInput = makeMatMul(parsed.trunk.ginput);
    const toTfBlock = (b: ParsedTrunkBlock): TfTrunkBlock => {
      if (b.kind === 'ordinary') {
        return {
          kind: 'ordinary',
          preBN: makeBn(b.preBN),
          preActivation: b.preActivation,
          w1: makeConv(b.w1),
          midBN: makeBn(b.midBN),
          midActivation: b.midActivation,
          w2: makeConv(b.w2),
        };
      }
      if (b.kind === 'gpool') {
        return {
          kind: 'gpool',
          preBN: makeBn(b.preBN),
          preActivation: b.preActivation,
          w1a: makeConv(b.w1a),
          w1b: makeConv(b.w1b),
          gpoolBN: makeBn(b.gpoolBN),
          gpoolActivation: b.gpoolActivation,
          w1r: makeMatMul(b.w1r),
          midBN: makeBn(b.midBN),
          midActivation: b.midActivation,
          w2: makeConv(b.w2),
        };
      }
      if (b.kind === 'nested_bottleneck') {
        return {
          kind: 'nested_bottleneck',
          numBlocks: b.numBlocks,
          preBN: makeBn(b.preBN),
          preActivation: b.preActivation,
          preConv: makeConv(b.preConv),
          blocks: b.blocks.map(toTfBlock),
          postBN: makeBn(b.postBN),
          postActivation: b.postActivation,
          postConv: makeConv(b.postConv),
        };
      }
      if (b.kind === 'transformer_attention') {
        return {
          kind: 'transformer_attention',
          fuseRotary: options.fuseRotary ?? true,
          numHeads: b.numHeads,
          numKVHeads: b.numKVHeads,
          qHeadDim: b.qHeadDim,
          vHeadDim: b.vHeadDim,
          preLN: makeTransformerRMSNorm(b.preLN, options.fuseRmsNorm ?? true),
          qkv: makeProjectionGroup([b.qProj, b.kProj, b.vProj], fuseProjections),
          outProj: makeMatMul(b.outProj),
          rope: precomputeRope(b),
          scale: 1.0 / Math.sqrt(b.qHeadDim),
        };
      }
      if (b.kind === 'transformer_ffn') {
        return {
          kind: 'transformer_ffn',
          useSwiGLU: b.useSwiGLU,
          preLN: makeTransformerRMSNorm(b.preLN, options.fuseRmsNorm ?? true),
          input: makeProjectionGroup(b.linearGate ? [b.linear1, b.linearGate] : [b.linear1], fuseProjections),
          linear2: makeMatMul(b.linear2),
        };
      }
      throw new Error(`Unknown block kind ${(b as { kind: string }).kind}`);
    };
    this.trunkBlocks = parsed.trunk.blocks.map(toTfBlock);
    this.trunkTipBN = parsed.trunk.tipBN ? makeBn(parsed.trunk.tipBN) : undefined;
    this.trunkTipRMSNorm = parsed.trunk.tipRMSNorm ? makeRMSNormLayer(parsed.trunk.tipRMSNorm) : undefined;
    this.trunkTipActivation = parsed.trunk.tipActivation;

    this.p1 = makeConv(parsed.policy.p1);
    this.g1 = makeConv(parsed.policy.g1);
    this.g1BN = makeBn(parsed.policy.g1BN);
    this.g1Activation = parsed.policy.g1Activation;
    this.gpoolToBias = makeMatMul(parsed.policy.gpoolToBias);
    this.p1BN = makeBn(parsed.policy.p1BN);
    this.p1Activation = parsed.policy.p1Activation;
    this.p2 = makeConv(parsed.policy.p2);
    this.passMul = makeMatMul(parsed.policy.passMul);
    this.passBias = parsed.policy.passBias ? makeMatBias(parsed.policy.passBias) : undefined;
    this.passActivation = parsed.policy.passActivation;
    this.passMul2 = parsed.policy.passMul2 ? makeMatMul(parsed.policy.passMul2) : undefined;

    this.v1 = makeConv(parsed.value.v1);
    this.v1BN = makeBn(parsed.value.v1BN);
    this.v1Activation = parsed.value.v1Activation;
    this.v2 = makeMatMul(parsed.value.v2);
    this.v2Bias = makeMatBias(parsed.value.v2Bias);
    this.v2Activation = parsed.value.v2Activation;
    this.v3 = makeMatMul(parsed.value.v3);
    this.v3Bias = makeMatBias(parsed.value.v3Bias);
    this.sv3 = makeMatMul(parsed.value.sv3);
    this.sv3Bias = makeMatBias(parsed.value.sv3Bias);
    this.ownership = makeConv(parsed.value.ownership);
  }

  forward(spatial: tf.Tensor4D, global: tf.Tensor2D): {
    policy: tf.Tensor4D;
    policyPass: tf.Tensor2D;
    value: tf.Tensor2D;
    scoreValue: tf.Tensor2D;
    ownership: tf.Tensor4D;
  } {
    return tf.tidy(() => {
      const trunk = this.forwardTrunk(spatial, global);

      // Policy head
      let p1Out = conv2d(trunk, this.p1);
      const g1Out = conv2d(trunk, this.g1);
      const g1Out2 = bnAct(g1Out, this.g1BN, this.g1Activation);
      const g1Concat = poolRowsGPool(g1Out2); // [N, g1C*3]
      const g1Bias = tf.matMul(g1Concat, this.gpoolToBias.w) as tf.Tensor2D; // [N, p1C]
      p1Out = p1Out.add(g1Bias.reshape([g1Bias.shape[0], 1, 1, g1Bias.shape[1]])) as tf.Tensor4D;
      const p1Out2 = bnAct(p1Out, this.p1BN, this.p1Activation);

      const policy = conv2d(p1Out2, this.p2); // [N,19,19,policyOutChannels]
      const policyPass = this.forwardPolicyPass(g1Concat); // [N,policyOutChannels]

      // Value head
      const v1Out = conv2d(trunk, this.v1);
      const v1Out2 = bnAct(v1Out, this.v1BN, this.v1Activation);
      const v1Mean = poolRowsValueHead(v1Out2); // [N,96]
      let v2Out = tf.matMul(v1Mean, this.v2.w) as tf.Tensor2D; // [N,64]
      v2Out = v2Out.add(this.v2Bias.b) as tf.Tensor2D;
      v2Out = applyActivation2D(v2Out, this.v2Activation);
      let value = tf.matMul(v2Out, this.v3.w) as tf.Tensor2D; // [N,3]
      value = value.add(this.v3Bias.b) as tf.Tensor2D;
      let scoreValue = tf.matMul(v2Out, this.sv3.w) as tf.Tensor2D; // [N,scoreValueChannels]
      scoreValue = scoreValue.add(this.sv3Bias.b) as tf.Tensor2D;
      if (this.scoreValueChannels > 4) {
        scoreValue = scoreValue.slice([0, 0], [scoreValue.shape[0], 4]) as tf.Tensor2D;
      }

      const ownership = conv2d(v1Out2, this.ownership); // [N,19,19,1]

      return { policy, policyPass, value, scoreValue, ownership };
    });
  }

  forwardPolicyValue(spatial: tf.Tensor4D, global: tf.Tensor2D): {
    policy: tf.Tensor4D;
    policyPass: tf.Tensor2D;
    value: tf.Tensor2D;
    scoreValue: tf.Tensor2D;
  } {
    return tf.tidy(() => {
      const trunk = this.forwardTrunk(spatial, global);

      let p1Out = conv2d(trunk, this.p1);
      const g1Out = conv2d(trunk, this.g1);
      const g1Out2 = bnAct(g1Out, this.g1BN, this.g1Activation);
      const g1Concat = poolRowsGPool(g1Out2);
      const g1Bias = tf.matMul(g1Concat, this.gpoolToBias.w) as tf.Tensor2D;
      p1Out = p1Out.add(g1Bias.reshape([g1Bias.shape[0], 1, 1, g1Bias.shape[1]])) as tf.Tensor4D;
      const p1Out2 = bnAct(p1Out, this.p1BN, this.p1Activation);

      const policy = conv2d(p1Out2, this.p2);
      const policyPass = this.forwardPolicyPass(g1Concat);

      const v1Out = conv2d(trunk, this.v1);
      const v1Out2 = bnAct(v1Out, this.v1BN, this.v1Activation);
      const v1Mean = poolRowsValueHead(v1Out2);
      let v2Out = tf.matMul(v1Mean, this.v2.w) as tf.Tensor2D;
      v2Out = v2Out.add(this.v2Bias.b) as tf.Tensor2D;
      v2Out = applyActivation2D(v2Out, this.v2Activation);
      let value = tf.matMul(v2Out, this.v3.w) as tf.Tensor2D;
      value = value.add(this.v3Bias.b) as tf.Tensor2D;
      let scoreValue = tf.matMul(v2Out, this.sv3.w) as tf.Tensor2D;
      scoreValue = scoreValue.add(this.sv3Bias.b) as tf.Tensor2D;
      if (this.scoreValueChannels > 4) {
        scoreValue = scoreValue.slice([0, 0], [scoreValue.shape[0], 4]) as tf.Tensor2D;
      }

      return { policy, policyPass, value, scoreValue };
    });
  }

  forwardValueOnly(
    spatial: tf.Tensor4D,
    global: tf.Tensor2D
  ): {
    value: tf.Tensor2D;
    scoreValue: tf.Tensor2D;
  } {
    return tf.tidy(() => {
      const trunk = this.forwardTrunk(spatial, global);
      const v1Out = conv2d(trunk, this.v1);
      const v1Out2 = bnAct(v1Out, this.v1BN, this.v1Activation);
      const v1Mean = poolRowsValueHead(v1Out2);
      let v2Out = tf.matMul(v1Mean, this.v2.w) as tf.Tensor2D;
      v2Out = v2Out.add(this.v2Bias.b) as tf.Tensor2D;
      v2Out = applyActivation2D(v2Out, this.v2Activation);
      let value = tf.matMul(v2Out, this.v3.w) as tf.Tensor2D;
      value = value.add(this.v3Bias.b) as tf.Tensor2D;
      let scoreValue = tf.matMul(v2Out, this.sv3.w) as tf.Tensor2D;
      scoreValue = scoreValue.add(this.sv3Bias.b) as tf.Tensor2D;
      if (this.scoreValueChannels > 4) {
        scoreValue = scoreValue.slice([0, 0], [scoreValue.shape[0], 4]) as tf.Tensor2D;
      }
      return { value, scoreValue };
    });
  }

  private forwardTrunk(spatial: tf.Tensor4D, global: tf.Tensor2D): tf.Tensor4D {
    let trunk = conv2d(spatial, this.trunkConv1);
    const ginput = tf.matMul(global, this.trunkGInput.w) as tf.Tensor2D;
    trunk = trunk.add(ginput.reshape([ginput.shape[0], 1, 1, ginput.shape[1]])) as tf.Tensor4D;

    trunk = this.applyBlockStack(trunk, this.trunkBlocks);
    if (this.trunkTipRMSNorm) {
      return rmsNormAct(trunk, this.trunkTipRMSNorm, this.trunkTipActivation);
    }
    return bnAct(trunk, this.trunkTipBN!, this.trunkTipActivation);
  }

  private forwardPolicyPass(gpool: tf.Tensor2D): tf.Tensor2D {
    let pass = tf.matMul(gpool, this.passMul.w) as tf.Tensor2D;
    if (this.passBias && this.passActivation && this.passMul2) {
      pass = pass.add(this.passBias.b) as tf.Tensor2D;
      pass = applyActivation2D(pass, this.passActivation);
      pass = tf.matMul(pass, this.passMul2.w) as tf.Tensor2D;
    }
    return pass;
  }

  private applyBlockStack(trunk: tf.Tensor4D, blocks: TfTrunkBlock[]): tf.Tensor4D {
    for (const block of blocks) {
      // Release attention matrices and other temporaries after each block,
      // rather than retaining every layer until the outer forward tidy exits.
      trunk = tf.tidy(() => {
        if (block.kind === 'ordinary') {
          const a = bnAct(trunk, block.preBN, block.preActivation);
          const b = conv2d(a, block.w1);
          const c = bnAct(b, block.midBN, block.midActivation);
          const d = conv2d(c, block.w2);
          return trunk.add(d) as tf.Tensor4D;
        }

        if (block.kind === 'gpool') {
          const a = bnAct(trunk, block.preBN, block.preActivation);
          let regularOut = conv2d(a, block.w1a);
          const gpoolOut = conv2d(a, block.w1b);
          const gpoolOut2 = bnAct(gpoolOut, block.gpoolBN, block.gpoolActivation);
          const gpoolConcat = poolRowsGPool(gpoolOut2);
          const gpoolBias = tf.matMul(gpoolConcat, block.w1r.w) as tf.Tensor2D;
          regularOut = regularOut.add(gpoolBias.reshape([gpoolBias.shape[0], 1, 1, gpoolBias.shape[1]])) as tf.Tensor4D;
          const c = bnAct(regularOut, block.midBN, block.midActivation);
          const d = conv2d(c, block.w2);
          return trunk.add(d) as tf.Tensor4D;
        }

        if (block.kind === 'nested_bottleneck') {
          const a = bnAct(trunk, block.preBN, block.preActivation);
          const mid = this.applyBlockStack(conv2d(a, block.preConv), block.blocks);
          const c = bnAct(mid, block.postBN, block.postActivation);
          const d = conv2d(c, block.postConv);
          return trunk.add(d) as tf.Tensor4D;
        }

        if (block.kind === 'transformer_attention') {
          return forwardAttention(trunk, block);
        }

        if (block.kind === 'transformer_ffn') {
          return forwardFFN(trunk, block);
        }
        throw new Error('Unsupported trunk block');
      });
    }
    return trunk;
  }

  dispose(): void {
    this.inputBuffers.dispose();
    const tensors: tf.Tensor[] = [
      this.trunkConv1.filter,
      this.trunkGInput.w,
      ...(this.trunkTipBN ? [this.trunkTipBN.scale, this.trunkTipBN.bias] : []),
      ...(this.trunkTipRMSNorm ? [this.trunkTipRMSNorm.gamma, this.trunkTipRMSNorm.beta] : []),
      this.p1.filter,
      this.g1.filter,
      this.g1BN.scale,
      this.g1BN.bias,
      this.gpoolToBias.w,
      this.p1BN.scale,
      this.p1BN.bias,
      this.p2.filter,
      this.passMul.w,
      ...(this.passBias ? [this.passBias.b] : []),
      ...(this.passMul2 ? [this.passMul2.w] : []),
      this.v1.filter,
      this.v1BN.scale,
      this.v1BN.bias,
      this.v2.w,
      this.v2Bias.b,
      this.v3.w,
      this.v3Bias.b,
      this.sv3.w,
      this.sv3Bias.b,
      this.ownership.filter,
    ];

    const pushBlockTensors = (block: TfTrunkBlock): void => {
      if (block.kind === 'ordinary') {
        tensors.push(block.preBN.scale, block.preBN.bias, block.w1.filter, block.midBN.scale, block.midBN.bias, block.w2.filter);
        return;
      }
      if (block.kind === 'gpool') {
        tensors.push(
          block.preBN.scale,
          block.preBN.bias,
          block.w1a.filter,
          block.w1b.filter,
          block.gpoolBN.scale,
          block.gpoolBN.bias,
          block.w1r.w,
          block.midBN.scale,
          block.midBN.bias,
          block.w2.filter
        );
        return;
      }
      if (block.kind === 'nested_bottleneck') {
        tensors.push(block.preBN.scale, block.preBN.bias, block.preConv.filter);
        for (const inner of block.blocks) pushBlockTensors(inner);
        tensors.push(block.postBN.scale, block.postBN.bias, block.postConv.filter);
        return;
      }
      if (block.kind === 'transformer_attention') {
        tensors.push(
          block.preLN.weight,
          ...block.qkv.layers.map(layer => layer.w),
          block.outProj.w
        );
        if (block.rope) {
          tensors.push(block.rope.cos, block.rope.sin);
        }
        return;
      }
      if (block.kind === 'transformer_ffn') {
        tensors.push(
          block.preLN.weight,
          ...block.input.layers.map(layer => layer.w),
          block.linear2.w
        );
        return;
      }
    };

    for (const block of this.trunkBlocks) pushBlockTensors(block);

    tf.dispose(tensors);
  }
}
