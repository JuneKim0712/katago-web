import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './test/browser',
  timeout: 120000,
  use: {
    browserName: 'chromium',
    channel: 'chrome',
    headless: true,
    launchOptions: {
      args: [
        '--enable-unsafe-webgpu',
        '--js-flags=--experimental-wasm-jspi',
      ],
    },
  },
});
