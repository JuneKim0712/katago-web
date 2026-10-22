/**
 * Tensor layout adapters for converting between KataGo TFJS layout (NHWC)
 * and KataGo ONNX Runtime layout (NCHW).
 *
 * KataGo TFJS formats:
 *   - Spatial input: [batch, height, width, channels] = [N, 19, 19, 22] (NHWC)
 *   - Global input:  [batch, channels] = [N, 19]
 *   - Policy output: [batch, height, width, channels] = [N, 19, 19, 2] (NHWC)
 *   - Pass output:   [batch, channels] = [N, 2]
 *   - Value output:  [batch, 3]
 *   - Score output:  [batch, 6]
 *   - Ownership output: [batch, height, width, 1] = [N, 19, 19, 1] (NHWC)
 *
 * KataGo ONNX formats (documented in docs/ONNX_Model_Files.md):
 *   - InputSpatial:  [batch, channels, height, width] = [N, 22, 19, 19] (NCHW)
 *   - InputGlobal:   [batch, channels, 1, 1] = [N, 19, 1, 1] (NCHW)
 *   - InputMask:     [batch, 1, height, width] = [N, 1, 19, 19] (channel 0 of InputSpatial)
 *   - OutputPolicy:  [batch, channels, height, width] = [N, 2, 19, 19] (NCHW)
 *   - OutputPolicyPass: [batch, channels, 1, 1] = [N, 2, 1, 1]
 *   - OutputValue:   [batch, 3, 1, 1]
 *   - OutputScoreValue: [batch, 6, 1, 1]
 *   - OutputOwnership: [batch, 1, height, width] = [N, 1, 19, 19] (NCHW)
 */

export function nhwcToNchw(nhwc, batch, height, width, channels) {
  const nchw = new Float32Array(batch * channels * height * width);
  for (let b = 0; b < batch; b++) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        for (let c = 0; c < channels; c++) {
          const nhwcIdx = ((b * height + y) * width + x) * channels + c;
          const nchwIdx = ((b * channels + c) * height + y) * width + x;
          nchw[nchwIdx] = nhwc[nhwcIdx];
        }
      }
    }
  }
  return nchw;
}

export function extractMask(nhwc, batch, height, width, channels) {
  const mask = new Float32Array(batch * 1 * height * width);
  for (let b = 0; b < batch; b++) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const nhwcIdx = ((b * height + y) * width + x) * channels + 0;
        const maskIdx = (b * height + y) * width + x;
        mask[maskIdx] = nhwc[nhwcIdx];
      }
    }
  }
  return mask;
}

export function nchwToNhwc(nchw, batch, channels, height, width) {
  const nhwc = new Float32Array(batch * height * width * channels);
  for (let b = 0; b < batch; b++) {
    for (let c = 0; c < channels; c++) {
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const nchwIdx = ((b * channels + c) * height + y) * width + x;
          const nhwcIdx = ((b * height + y) * width + x) * channels + c;
          nhwc[nhwcIdx] = nchw[nchwIdx];
        }
      }
    }
  }
  return nhwc;
}

/**
 * Self-verification test for adapters using a labeled coordinate tensor.
 */
export function verifyAdapters() {
  const B = 2, H = 3, W = 4, C = 5;
  const original = new Float32Array(B * H * W * C);
  for (let b = 0; b < B; b++) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        for (let c = 0; c < C; c++) {
          original[((b * H + y) * W + x) * C + c] = b * 1000 + y * 100 + x * 10 + c;
        }
      }
    }
  }

  const nchw = nhwcToNchw(original, B, H, W, C);
  for (let b = 0; b < B; b++) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        for (let c = 0; c < C; c++) {
          const expected = b * 1000 + y * 100 + x * 10 + c;
          const actual = nchw[((b * C + c) * H + y) * W + x];
          if (actual !== expected) {
            throw new Error(`nhwcToNchw failed at b=${b}, c=${c}, y=${y}, x=${x}`);
          }
        }
      }
    }
  }

  const mask = extractMask(original, B, H, W, C);
  for (let b = 0; b < B; b++) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const expected = b * 1000 + y * 100 + x * 10 + 0;
        const actual = mask[(b * H + y) * W + x];
        if (actual !== expected) {
          throw new Error(`extractMask failed at b=${b}, y=${y}, x=${x}`);
        }
      }
    }
  }

  const roundTrip = nchwToNhwc(nchw, B, C, H, W);
  for (let i = 0; i < original.length; i++) {
    if (roundTrip[i] !== original[i]) {
      throw new Error(`nchwToNhwc roundtrip failed at index ${i}`);
    }
  }
  return true;
}
