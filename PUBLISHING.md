# Publishing & Release Guide

This document describes the release engineering and publishing process for **`katago-web`**. It ensures that every release published to npm and pushed to GitHub is strictly typed, thoroughly tested, securely signed, and reproducible.

---

## Table of Contents

- [1. Release Principles](#1-release-principles)
- [2. Pre-Release Quality Gates](#2-pre-release-quality-gates)
- [3. Versioning & Package Preparation](#3-versioning--package-preparation)
- [4. Pushing to GitHub](#4-pushing-to-github)
- [5. Publishing to npm](#5-publishing-to-npm)
- [6. Troubleshooting 2FA and Access Errors](#6-troubleshooting-2fa-and-access-errors)
- [7. Automated CI/CD Release (GitHub Actions)](#7-automated-cicd-release-github-actions)

---

## 1. Release Principles

1. **Semantic Versioning (SemVer 2.0.0)**:
   - `MAJOR` (`x.0.0`): Breaking public API changes (Contract v1 interface modifications, removed exports).
   - `MINOR` (`1.x.0`): New backward-compatible features (new search parameters, ruleset presets, tree reuse optimizations).
   - `PATCH` (`1.0.x`): Backward-compatible bug fixes, performance improvements, internal refactoring.
2. **Immutable Artifacts**: Once published to the npm registry, a version cannot be overwritten or replaced.
3. **Clean Packaging**: Only the compiled `dist/` directory, `README.md`, `LICENSE`, and `package.json` are packaged. Raw C++ sources, test fixtures, benchmarks, and configuration files are excluded via the `"files"` field in `package.json`.

---

## 2. Pre-Release Quality Gates

Before tagging or publishing any release, execute the full quality verification pipeline:

```bash
# 1. Typecheck the entire project with strict compiler flags
npm run typecheck

# 2. Compile clean TypeScript declaration files and distribution bundles
npm run build

# 3. Run core unit tests, contract validations, and search reuse suites
npm test

# 4. Verify tarball contents and size without publishing (Dry Run)
npm pack --dry-run
```

### Inspecting the Dry-Run Output
Ensure that:
- **Tarball size**: Approximately $500\text{--}550\text{ KB}$ (unpacked $\approx 2.0\text{ MB}$, primarily containing `dist/wasm/katago.wasm`).
- **Files**: Confirmed that `test/`, `src/`, `native/`, `.github/`, and local temporary files are **not** present in the tarball contents.

---

## 3. Versioning & Package Preparation

Use the standard `npm version` command, which automatically updates `package.json`, `package-lock.json`, and creates a signed Git tag:

```bash
# For a bug fix or performance update:
npm version patch -m "chore(release): %s"

# For new backward-compatible features:
npm version minor -m "chore(release): %s"

# For breaking API changes:
npm version major -m "chore(release): %s"
```

---

## 4. Pushing to GitHub

Push your commits and the newly created Git release tag to the `origin` repository:

```bash
# Push master branch commits
git push origin master

# Push tags to GitHub
git push origin --tags
```

### Creating GitHub Releases (Optional but Recommended)
1. Navigate to [`https://github.com/JuneKim0712/katago-web/releases`](https://github.com/JuneKim0712/katago-web/releases).
2. Click **Draft a new release**.
3. Select the Git tag you just pushed (e.g. `v1.0.1`).
4. Click **Generate release notes** to populate the changelog.
5. Click **Publish release**.

---

## 5. Publishing to npm

### Step 5.1: Authenticate with npm
Verify that your local CLI session is authenticated:

```bash
npm whoami
```

If not logged in, authenticate:
```bash
npm login
```
*(Enter your username, password, email verification code, and 6-digit authenticator code).*

### Step 5.2: Publish Package
Because npm enforces Two-Factor Authentication (2FA) for all package writes, provide your live 6-digit TOTP code from your authenticator app (Google Authenticator, Authy, Apple Passwords, etc.):

```bash
npm publish --access public --otp=YOUR_6_DIGIT_OTP
```

*Note: The `prepublishOnly` lifecycle script configured in `package.json` automatically triggers `npm run typecheck && npm run build` before packaging, guaranteeing that stale artifacts can never be published.*

### Step 5.3: Verify Live Registry Status
Confirm that the package is live on npm:

```bash
npm view katago-web version
```

Visit the npm package page:
`https://www.npmjs.com/package/katago-web`

---

## 6. Troubleshooting 2FA and Access Errors

### Error: `403 Forbidden - Two-factor authentication or granular access token with bypass 2fa enabled is required`
- **Cause**: npm requires 2FA on publisher accounts.
- **Solution**:
  1. Open [npmjs.com/settings/teresta/2fa](https://www.npmjs.com/settings/teresta/2fa).
  2. Enable 2FA using an Authenticator app.
  3. Include `--otp=XXXXXX` when running `npm publish`.

### Error: `403 Forbidden - Your account has been temporarily suspended due to a recent security-sensitive action`
- **Cause**: npm places an automated 5-to-10 minute security cooldown hold on publish operations immediately after enabling 2FA or changing credentials.
- **Solution**:
  1. Check your email for any security confirmation links from npm.
  2. Wait 10 minutes for the automated security cooldown to expire.
  3. Retry publishing with a fresh 6-digit OTP.

### Error: `403 Forbidden - You cannot publish over the previously published versions` (`EPUBLISHCONFLICT`)
- **Cause**: The version specified in `package.json` has already been published to npm.
- **Solution**: Bump the version with `npm version patch` before publishing.

---

## 7. Automated CI/CD Release (GitHub Actions)

To automate publishing on every GitHub Release, create `.github/workflows/publish.yml`:

```yaml
name: Publish to npm

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write # Required for npm provenance
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 20
          registry-url: 'https://registry.npmjs.org'

      - name: Install dependencies
        run: npm ci

      - name: Run verification gates
        run: |
          npm run typecheck
          npm run build
          npm test

      - name: Publish to npm with Provenance
        run: npm publish --access public --provenance
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

*(To use this, generate an npm Granular Access Token with **Read and write** permissions for `katago-web`, select **Bypass 2FA**, and add it as a secret named `NPM_TOKEN` under GitHub repository settings $\to$ Secrets and variables $\to$ Actions).*
