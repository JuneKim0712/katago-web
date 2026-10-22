import {
  API_VERSION,
  AnalyzeRequest,
  Capabilities,
  Dimensions,
  EngineFailure,
  KataGoEngineError,
  Move,
  Position,
  Rules,
  RulesInput,
  StonePlacement,
} from './types';

export function makeEngineFailure(
  code: EngineFailure['code'],
  message: string,
  requestId: string | null = null,
  engineGeneration: number = 0,
  details: Record<string, string | number | boolean | null> = {}
): EngineFailure {
  return {
    code,
    message,
    requestId,
    engineGeneration,
    details,
  };
}

export function failRequest(
  code: EngineFailure['code'],
  message: string,
  requestId: string | null = null,
  engineGeneration: number = 0,
  details: Record<string, string | number | boolean | null> = {}
): never {
  throw new KataGoEngineError(makeEngineFailure(code, message, requestId, engineGeneration, details));
}

export function validateIdentifier(id: unknown, fieldName: string, requestId: string | null = null): string {
  if (typeof id !== 'string' || id.trim().length === 0 || id.length > 128) {
    failRequest('INVALID_REQUEST', `${fieldName} must be a non-empty string <= 128 chars`, requestId, 0, { [fieldName]: String(id) });
  }
  return id;
}

export function validateDimensions(dims: unknown, caps: Capabilities, requestId: string | null = null): Dimensions {
  if (!dims || typeof dims !== 'object') {
    failRequest('INVALID_POSITION', 'Dimensions must be an object', requestId);
  }
  const d = dims as Record<string, unknown>;
  const width = d.width;
  const height = d.height;
  if (!Number.isInteger(width) || !Number.isInteger(height)) {
    failRequest('INVALID_POSITION', 'Board dimensions must be integers', requestId, 0, { width: String(width), height: String(height) });
  }
  const w = width as number;
  const h = height as number;
  if (w < 2 || h < 2 || w > caps.maxBoardWidth || h > caps.maxBoardHeight) {
    failRequest('UNSUPPORTED_CAPABILITY', `Board dimensions (${w}x${h}) exceed limits (max ${caps.maxBoardWidth}x${caps.maxBoardHeight})`, requestId, 0, { width: w, height: h });
  }
  if (w !== h && !caps.supportsRectangularBoards) {
    failRequest('UNSUPPORTED_CAPABILITY', `Rectangular boards not supported`, requestId, 0, { width: w, height: h });
  }
  return { width: w, height: h };
}

export function validatePoint(pt: unknown, dims: Dimensions, requestId: string | null = null): { x: number; y: number } {
  if (!pt || typeof pt !== 'object') {
    failRequest('INVALID_POSITION', 'Point must be an object', requestId);
  }
  const p = pt as Record<string, unknown>;
  const x = p.x;
  const y = p.y;
  if (!Number.isInteger(x) || !Number.isInteger(y) || Object.is(x, -0) || Object.is(y, -0)) {
    failRequest('INVALID_POSITION', 'Point coordinates must be non-negative integers', requestId, 0, { x: String(x), y: String(y) });
  }
  const xi = x as number;
  const yi = y as number;
  if (xi < 0 || xi >= dims.width || yi < 0 || yi >= dims.height) {
    failRequest('INVALID_POSITION', `Point (${xi},${yi}) is out of bounds for ${dims.width}x${dims.height} board`, requestId, 0, { x: xi, y: yi });
  }
  return { x: xi, y: yi };
}

export function validateMove(move: unknown, dims: Dimensions, requestId: string | null = null): Move {
  if (!move || typeof move !== 'object') {
    failRequest('INVALID_REQUEST', 'Move must be an object', requestId);
  }
  const m = move as Record<string, unknown>;
  if (m.kind === 'pass') {
    return { kind: 'pass' };
  }
  if (m.kind === 'play') {
    const pt = validatePoint(m.point, dims, requestId);
    return { kind: 'play', point: pt };
  }
  failRequest('INVALID_REQUEST', `Invalid move kind: ${String(m.kind)}`, requestId, 0, { kind: String(m.kind) });
}

export function validateKomi(komi: unknown, requestId: string | null = null): number {
  if (typeof komi !== 'number' || !Number.isFinite(komi)) {
    failRequest('INVALID_REQUEST', 'Komi must be a finite number', requestId, 0, { komi: String(komi) });
  }
  // Check integer or half-integer
  if (Math.floor(komi * 2) !== komi * 2) {
    failRequest('INVALID_REQUEST', 'Komi must be an integer or half-integer', requestId, 0, { komi });
  }
  if (komi < -150 || komi > 150) {
    failRequest('INVALID_REQUEST', `Komi ${komi} is out of realistic supported range`, requestId, 0, { komi });
  }
  return komi;
}

export function resolveRules(rulesInput: RulesInput, caps: Capabilities, requestId: string | null = null): Rules {
  if (rulesInput.kind === 'preset') {
    const name = rulesInput.name;
    if (!caps.supportedRulePresets.includes(name)) {
      failRequest('UNSUPPORTED_CAPABILITY', `Unsupported rule preset: ${name}`, requestId, 0, { preset: name });
    }
    switch (name) {
      case 'japanese':
        return {
          ko: 'simple',
          scoring: 'territory',
          tax: 'seki',
          multiStoneSuicideLegal: false,
          hasButton: false,
          whiteHandicapBonus: '0',
          friendlyPassOk: false,
        };
      case 'korean':
        return {
          ko: 'simple',
          scoring: 'territory',
          tax: 'seki',
          multiStoneSuicideLegal: false,
          hasButton: false,
          whiteHandicapBonus: '0',
          friendlyPassOk: false,
        };
      case 'chinese':
        return {
          ko: 'simple',
          scoring: 'area',
          tax: 'none',
          multiStoneSuicideLegal: false,
          hasButton: false,
          whiteHandicapBonus: 'N',
          friendlyPassOk: true,
        };
      default:
        failRequest('INVALID_REQUEST', `Unknown rules preset: ${name}`, requestId);
    }
  } else if (rulesInput.kind === 'explicit') {
    const r = rulesInput.value;
    if (!['simple', 'positional', 'situational', 'spight'].includes(r.ko)) {
      failRequest('INVALID_REQUEST', `Invalid ko rule: ${r.ko}`, requestId);
    }
    if (!['area', 'territory'].includes(r.scoring)) {
      failRequest('INVALID_REQUEST', `Invalid scoring rule: ${r.scoring}`, requestId);
    }
    if (!['none', 'seki', 'all'].includes(r.tax)) {
      failRequest('INVALID_REQUEST', `Invalid tax rule: ${r.tax}`, requestId);
    }
    return r;
  }
  failRequest('INVALID_REQUEST', 'Invalid rules input', requestId);
}

export function validateAnalyzeRequest(req: unknown, caps: Capabilities, activeIds: Set<string>): AnalyzeRequest {
  if (!req || typeof req !== 'object') {
    failRequest('INVALID_REQUEST', 'AnalyzeRequest must be an object');
  }
  const r = req as Record<string, unknown>;

  if (r.apiVersion !== API_VERSION) {
    failRequest('UNSUPPORTED_API_VERSION', `API version must be ${API_VERSION}`, null, 0, { apiVersion: String(r.apiVersion) });
  }

  const requestId = validateIdentifier(r.requestId, 'requestId');
  if (activeIds.has(requestId)) {
    failRequest('INVALID_REQUEST', `requestId '${requestId}' has already been used`, requestId);
  }

  const position = r.position as Position;
  if (!position || typeof position !== 'object') {
    failRequest('INVALID_POSITION', 'position must be specified', requestId);
  }

  const dims = validateDimensions(position.dimensions, caps, requestId);
  validateKomi(position.komi, requestId);
  resolveRules(position.rules, caps, requestId);

  // Validate search
  const search = r.search as Record<string, unknown>;
  if (!search || typeof search !== 'object') {
    failRequest('INVALID_REQUEST', 'search object is required', requestId);
  }
  if (search.profile !== 'upstream-analysis-v1') {
    failRequest('UNSUPPORTED_CAPABILITY', `Unsupported search profile: ${String(search.profile)}`, requestId);
  }

  // Limits
  const limits = search.limits as Record<string, unknown>;
  if (!limits || typeof limits !== 'object') {
    failRequest('INVALID_REQUEST', 'search.limits is required', requestId);
  }
  if (limits.kind === 'visits') {
    const v = limits.maxVisits;
    if (!Number.isInteger(v) || (v as number) <= 0 || (v as number) > caps.maxVisits) {
      failRequest('INVALID_REQUEST', `maxVisits must be in [1, ${caps.maxVisits}]`, requestId, 0, { maxVisits: String(v) });
    }
  } else if (limits.kind === 'time') {
    const t = limits.maxTimeMs;
    if (typeof t !== 'number' || !Number.isFinite(t) || t <= 0 || t > caps.maxTimeMs) {
      failRequest('INVALID_REQUEST', `maxTimeMs must be in (0, ${caps.maxTimeMs}]`, requestId, 0, { maxTimeMs: String(t) });
    }
  } else if (limits.kind === 'either') {
    const v = limits.maxVisits;
    const t = limits.maxTimeMs;
    if (!Number.isInteger(v) || (v as number) <= 0 || (v as number) > caps.maxVisits) {
      failRequest('INVALID_REQUEST', `maxVisits must be in [1, ${caps.maxVisits}]`, requestId, 0, { maxVisits: String(v) });
    }
    if (typeof t !== 'number' || !Number.isFinite(t) || (t as number) <= 0 || (t as number) > caps.maxTimeMs) {
      failRequest('INVALID_REQUEST', `maxTimeMs must be in (0, ${caps.maxTimeMs}]`, requestId, 0, { maxTimeMs: String(t) });
    }
  } else {
    failRequest('INVALID_REQUEST', `Invalid limit kind: ${String(limits.kind)}`, requestId);
  }

  // batchSize and searchThreads
  if (search.batchSize !== 'auto') {
    const bs = search.batchSize;
    if (!Number.isInteger(bs) || (bs as number) <= 0 || (bs as number) > caps.maxBatchSize) {
      failRequest('INVALID_REQUEST', `batchSize must be 'auto' or in [1, ${caps.maxBatchSize}]`, requestId, 0, { batchSize: String(bs) });
    }
  }
  if (search.searchThreads !== 'auto') {
    const st = search.searchThreads;
    if (!Number.isInteger(st) || (st as number) <= 0 || (st as number) > caps.maxSearchThreads) {
      failRequest('INVALID_REQUEST', `searchThreads must be 'auto' or in [1, ${caps.maxSearchThreads}]`, requestId, 0, { searchThreads: String(st) });
    }
  }

  // wideRootNoise & rootSymmetrySamples
  const wrn = search.wideRootNoise;
  if (typeof wrn !== 'number' || !Number.isFinite(wrn) || wrn < 0 || wrn > 1) {
    failRequest('INVALID_REQUEST', 'wideRootNoise must be in [0, 1]', requestId, 0, { wideRootNoise: String(wrn) });
  }
  const rss = search.rootSymmetrySamples;
  if (!Number.isInteger(rss) || (rss as number) < 1 || (rss as number) > 8) {
    failRequest('INVALID_REQUEST', 'rootSymmetrySamples must be integer in [1, 8]', requestId, 0, { rootSymmetrySamples: String(rss) });
  }

  // allowedRootMoves
  if (search.allowedRootMoves !== null) {
    if (!Array.isArray(search.allowedRootMoves) || search.allowedRootMoves.length === 0) {
      failRequest('INVALID_REQUEST', 'allowedRootMoves must be null or a non-empty array of moves', requestId);
    }
    for (const m of search.allowedRootMoves) {
      validateMove(m, dims, requestId);
    }
  }

  // Output options
  const output = r.output as Record<string, unknown>;
  if (!output || typeof output !== 'object') {
    failRequest('INVALID_REQUEST', 'output object is required', requestId);
  }
  const maxCand = output.maxCandidates;
  const boardArea = dims.width * dims.height;
  if (!Number.isInteger(maxCand) || (maxCand as number) < 0 || (maxCand as number) > boardArea + 1) {
    failRequest('INVALID_REQUEST', `maxCandidates must be integer in [0, ${boardArea + 1}]`, requestId, 0, { maxCandidates: String(maxCand) });
  }
  const pvLen = output.pvLength;
  if (!Number.isInteger(pvLen) || (pvLen as number) < 0 || (pvLen as number) > caps.maxPvLength) {
    failRequest('INVALID_REQUEST', `pvLength must be integer in [0, ${caps.maxPvLength}]`, requestId, 0, { pvLength: String(pvLen) });
  }
  if (output.ownership !== 'none' && output.ownership !== 'root' && output.ownership !== 'tree') {
    failRequest('INVALID_REQUEST', `Invalid ownership mode: ${String(output.ownership)}`, requestId);
  }
  if (output.includeOwnershipStdev && !caps.supportsOwnershipStdev) {
    failRequest('UNSUPPORTED_CAPABILITY', 'Ownership stdev not supported by this engine build', requestId);
  }
  if (output.includeCandidateOwnership && !caps.supportsCandidateOwnership) {
    failRequest('UNSUPPORTED_CAPABILITY', 'Candidate ownership not supported by this engine build', requestId);
  }

  return req as AnalyzeRequest;
}
