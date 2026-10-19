export function isGpuDeviceLostError(err: unknown): boolean {
  const message = err instanceof Error ? err.message : String(err);
  return /mapAsync|GPUBuffer|external Instance|device lost|Device was lost|Invalid GPU device/i.test(message);
}

export function formatKataGoUserError(err: unknown): string {
  if (isGpuDeviceLostError(err)) {
    return 'GPU ran out of memory or lost the device. Try a smaller model or wait for automatic CPU fallback.';
  }
  return err instanceof Error ? err.message : String(err);
}
