import { ModelArtifact, ModelManifest, ModelSource } from '../api/types';
import { failRequest } from '../api/validation';

export const BASELINE_MODEL_ID = 'b10c384-default';
export const BASELINE_MODEL_SHA256 = 'c306a252a2909859c52d4d2ab433ac89ba051f8f2606850df2f747203e5974e3';
export const BASELINE_MODEL_BYTES = 64605132;

export const BASELINE_B10C384_MANIFEST: ModelManifest = {
  schemaVersion: 1,
  modelId: BASELINE_MODEL_ID,
  sourceModelSha256: '8015d49bdf45854c94e5f9e22b2b70e8738d3e56a6edca730e5b82',
  variants: [
    {
      variantId: 'onnx-webgpu-default',
      format: 'onnx',
      entryPath: 'model.onnx',
      artifacts: [
        {
          path: 'model.onnx',
          sha256: BASELINE_MODEL_SHA256,
          byteLength: BASELINE_MODEL_BYTES,
          source: {
            kind: 'url',
            url: '/test/fixtures/model.onnx',
          },
        },
      ],
    },
  ],
};

export function validateManifest(manifest: unknown, requestId: string | null = null): ModelManifest {
  if (!manifest || typeof manifest !== 'object') {
    failRequest('MODEL_UNSUPPORTED', 'Manifest must be an object', requestId);
  }
  const m = manifest as Record<string, unknown>;
  if (m.schemaVersion !== 1) {
    failRequest('MODEL_UNSUPPORTED', `Unsupported manifest schemaVersion: ${String(m.schemaVersion)}`, requestId);
  }
  if (typeof m.modelId !== 'string' || m.modelId.trim().length === 0) {
    failRequest('MODEL_UNSUPPORTED', 'Manifest modelId must be non-empty string', requestId);
  }
  if (!Array.isArray(m.variants) || m.variants.length === 0) {
    failRequest('MODEL_UNSUPPORTED', 'Manifest must declare at least one variant', requestId);
  }

  for (const v of m.variants) {
    if (!v || typeof v !== 'object') {
      failRequest('MODEL_UNSUPPORTED', 'Invalid variant entry', requestId);
    }
    const varObj = v as Record<string, unknown>;
    if (!['onnx', 'katago-bin', 'katago-bin-gzip'].includes(varObj.format as string)) {
      failRequest('MODEL_UNSUPPORTED', `Unsupported format: ${String(varObj.format)}`, requestId);
    }
    if (!Array.isArray(varObj.artifacts) || varObj.artifacts.length === 0) {
      failRequest('MODEL_UNSUPPORTED', `Variant ${String(varObj.variantId)} has no artifacts`, requestId);
    }
    for (const a of varObj.artifacts) {
      if (!a.path || a.path.includes('..') || a.path.startsWith('/')) {
        failRequest('MODEL_UNSUPPORTED', `Illegal artifact path: ${String(a.path)}`, requestId);
      }
      if (typeof a.sha256 !== 'string' || a.sha256.length !== 64) {
        failRequest('MODEL_UNSUPPORTED', `Invalid artifact sha256: ${String(a.sha256)}`, requestId);
      }
      if (!Number.isInteger(a.byteLength) || a.byteLength <= 0) {
        failRequest('MODEL_UNSUPPORTED', `Invalid artifact byteLength: ${String(a.byteLength)}`, requestId);
      }
    }
  }

  return manifest as ModelManifest;
}

export async function computeSha256(bytes: Uint8Array): Promise<string> {
  if (typeof crypto !== 'undefined' && crypto.subtle) {
    const copy = new Uint8Array(bytes) as unknown as BufferSource;
    const hashBuf = await crypto.subtle.digest('SHA-256', copy);
    const hashArr = Array.from(new Uint8Array(hashBuf));
    return hashArr.map(b => b.toString(16).padStart(2, '0')).join('');
  }
  // Node.js fallback
  const { createHash } = await import('crypto');
  return createHash('sha256').update(bytes).digest('hex');
}

export function computeModelFingerprint(manifest: ModelManifest, variantId: string): string {
  const variant = manifest.variants.find(v => v.variantId === variantId);
  if (!variant) return manifest.modelId;
  const artifactSignatures = variant.artifacts.map(a => `${a.path}:${a.sha256}:${a.byteLength}`).sort().join(';');
  return `${manifest.modelId}:${variant.variantId}:${artifactSignatures}`;
}
