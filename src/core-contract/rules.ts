/**
 * Canonical KataGo ruleset representation matching upstream KataGo v1.18.1 (cpp/game/rules.h).
 */

export type KoRule = 'KO_SIMPLE' | 'KO_POSITIONAL' | 'KO_SITUATIONAL';
export type ScoringRule = 'SCORING_AREA' | 'SCORING_TERRITORY';
export type TaxRule = 'TAX_NONE' | 'TAX_SEKI' | 'TAX_ALL';
export type WhiteHandicapBonusRule = 'WHB_ZERO' | 'WHB_N' | 'WHB_N_MINUS_ONE';

export interface Rules {
  readonly koRule: KoRule;
  readonly scoringRule: ScoringRule;
  readonly taxRule: TaxRule;
  readonly multiStoneSuicideLegal: boolean;
  readonly hasButton: boolean;
  readonly whiteHandicapBonusRule: WhiteHandicapBonusRule;
  readonly friendlyPassOk: boolean;
}

export type RulesPresetName =
  | 'tromp-taylor'
  | 'chinese'
  | 'chinese-rulesist'
  | 'japanese'
  | 'korean'
  | 'aga'
  | 'new-zealand'
  | 'stone-counting';

export const RULES_PRESETS: Readonly<Record<RulesPresetName, Readonly<Rules>>> = Object.freeze({
  'tromp-taylor': Object.freeze({
    koRule: 'KO_POSITIONAL',
    scoringRule: 'SCORING_AREA',
    taxRule: 'TAX_NONE',
    multiStoneSuicideLegal: true,
    hasButton: false,
    whiteHandicapBonusRule: 'WHB_ZERO',
    friendlyPassOk: true,
  }),
  chinese: Object.freeze({
    koRule: 'KO_POSITIONAL',
    scoringRule: 'SCORING_AREA',
    taxRule: 'TAX_NONE',
    multiStoneSuicideLegal: false,
    hasButton: false,
    whiteHandicapBonusRule: 'WHB_N',
    friendlyPassOk: true,
  }),
  'chinese-rulesist': Object.freeze({
    koRule: 'KO_POSITIONAL',
    scoringRule: 'SCORING_AREA',
    taxRule: 'TAX_NONE',
    multiStoneSuicideLegal: false,
    hasButton: false,
    whiteHandicapBonusRule: 'WHB_N_MINUS_ONE',
    friendlyPassOk: true,
  }),
  japanese: Object.freeze({
    koRule: 'KO_SIMPLE',
    scoringRule: 'SCORING_TERRITORY',
    taxRule: 'TAX_SEKI',
    multiStoneSuicideLegal: false,
    hasButton: false,
    whiteHandicapBonusRule: 'WHB_ZERO',
    friendlyPassOk: false,
  }),
  korean: Object.freeze({
    koRule: 'KO_SIMPLE',
    scoringRule: 'SCORING_TERRITORY',
    taxRule: 'TAX_SEKI',
    multiStoneSuicideLegal: false,
    hasButton: false,
    whiteHandicapBonusRule: 'WHB_ZERO',
    friendlyPassOk: false,
  }),
  aga: Object.freeze({
    koRule: 'KO_SITUATIONAL',
    scoringRule: 'SCORING_AREA',
    taxRule: 'TAX_NONE',
    multiStoneSuicideLegal: false,
    hasButton: false,
    whiteHandicapBonusRule: 'WHB_N_MINUS_ONE',
    friendlyPassOk: true,
  }),
  'new-zealand': Object.freeze({
    koRule: 'KO_SITUATIONAL',
    scoringRule: 'SCORING_AREA',
    taxRule: 'TAX_NONE',
    multiStoneSuicideLegal: true,
    hasButton: false,
    whiteHandicapBonusRule: 'WHB_ZERO',
    friendlyPassOk: true,
  }),
  'stone-counting': Object.freeze({
    koRule: 'KO_SIMPLE',
    scoringRule: 'SCORING_AREA',
    taxRule: 'TAX_NONE',
    multiStoneSuicideLegal: false,
    hasButton: false,
    whiteHandicapBonusRule: 'WHB_ZERO',
    friendlyPassOk: true,
  }),
});

export const DEFAULT_RULES: Readonly<Rules> = RULES_PRESETS.japanese;

/**
 * Checks equality between two rules objects.
 */
export function equalsRules(a: Rules, b: Rules): boolean {
  return (
    a.koRule === b.koRule &&
    a.scoringRule === b.scoringRule &&
    a.taxRule === b.taxRule &&
    a.multiStoneSuicideLegal === b.multiStoneSuicideLegal &&
    a.hasButton === b.hasButton &&
    a.whiteHandicapBonusRule === b.whiteHandicapBonusRule &&
    a.friendlyPassOk === b.friendlyPassOk
  );
}

/**
 * Parses a string, preset name, or partial object into a complete, canonical Rules object.
 */
export function parseRules(input?: string | Partial<Rules> | null): Rules {
  if (!input) {
    return { ...DEFAULT_RULES };
  }

  if (typeof input === 'object') {
    return {
      koRule: input.koRule ?? DEFAULT_RULES.koRule,
      scoringRule: input.scoringRule ?? DEFAULT_RULES.scoringRule,
      taxRule: input.taxRule ?? DEFAULT_RULES.taxRule,
      multiStoneSuicideLegal: input.multiStoneSuicideLegal ?? DEFAULT_RULES.multiStoneSuicideLegal,
      hasButton: input.hasButton ?? DEFAULT_RULES.hasButton,
      whiteHandicapBonusRule: input.whiteHandicapBonusRule ?? DEFAULT_RULES.whiteHandicapBonusRule,
      friendlyPassOk: input.friendlyPassOk ?? DEFAULT_RULES.friendlyPassOk,
    };
  }

  const str = input.trim().toLowerCase();
  const normalized = str.startsWith('rules:') ? str.slice(6) : str;

  // Check known presets
  if (normalized in RULES_PRESETS) {
    return { ...RULES_PRESETS[normalized as RulesPresetName] };
  }

  // Handle JSON format if present
  if (normalized.startsWith('{')) {
    try {
      const parsed = JSON.parse(normalized);
      return parseRules(parsed);
    } catch {
      // Fall through to clause-based parsing
    }
  }

  // Parse comma- or semicolon-separated clauses (KataGo upstream format e.g. "ko=positional,scoring=area")
  const clauses = normalized.split(/[,;\s]+/);
  let koRule: KoRule = DEFAULT_RULES.koRule;
  let scoringRule: ScoringRule = DEFAULT_RULES.scoringRule;
  let taxRule: TaxRule = DEFAULT_RULES.taxRule;
  let multiStoneSuicideLegal = DEFAULT_RULES.multiStoneSuicideLegal;
  let hasButton = DEFAULT_RULES.hasButton;
  let whiteHandicapBonusRule: WhiteHandicapBonusRule = DEFAULT_RULES.whiteHandicapBonusRule;
  let friendlyPassOk = DEFAULT_RULES.friendlyPassOk;

  for (const clause of clauses) {
    const [k, v] = clause.split('=').map((s) => s.trim());
    if (!k) continue;

    if (k === 'ko') {
      if (v === 'simple') koRule = 'KO_SIMPLE';
      else if (v === 'positional') koRule = 'KO_POSITIONAL';
      else if (v === 'situational') koRule = 'KO_SITUATIONAL';
    } else if (k === 'scoring') {
      if (v === 'area') scoringRule = 'SCORING_AREA';
      else if (v === 'territory') scoringRule = 'SCORING_TERRITORY';
    } else if (k === 'tax') {
      if (v === 'none') taxRule = 'TAX_NONE';
      else if (v === 'seki') taxRule = 'TAX_SEKI';
      else if (v === 'all') taxRule = 'TAX_ALL';
    } else if (k === 'suicide') {
      multiStoneSuicideLegal = v === 'true' || v === '1';
    } else if (k === 'button') {
      hasButton = v === 'true' || v === '1';
    } else if (k === 'whb') {
      if (v === 'zero' || v === '0') whiteHandicapBonusRule = 'WHB_ZERO';
      else if (v === 'n') whiteHandicapBonusRule = 'WHB_N';
      else if (v === 'n-1' || v === 'n_minus_one') whiteHandicapBonusRule = 'WHB_N_MINUS_ONE';
    } else if (k === 'friendlypass') {
      friendlyPassOk = v === 'true' || v === '1';
    }
  }

  return {
    koRule,
    scoringRule,
    taxRule,
    multiStoneSuicideLegal,
    hasButton,
    whiteHandicapBonusRule,
    friendlyPassOk,
  };
}

/**
 * Serializes canonical Rules to a KataGo-compatible string representation.
 */
export function rulesToString(rules: Rules): string {
  // Check if it matches an exact preset
  for (const [name, preset] of Object.entries(RULES_PRESETS)) {
    if (equalsRules(rules, preset)) {
      return name;
    }
  }

  const parts: string[] = [];
  parts.push(`ko=${rules.koRule.slice(3).toLowerCase()}`);
  parts.push(`scoring=${rules.scoringRule.slice(8).toLowerCase()}`);
  parts.push(`tax=${rules.taxRule.slice(4).toLowerCase()}`);
  parts.push(`suicide=${rules.multiStoneSuicideLegal}`);
  parts.push(`button=${rules.hasButton}`);
  parts.push(`whb=${rules.whiteHandicapBonusRule === 'WHB_N_MINUS_ONE' ? 'n-1' : rules.whiteHandicapBonusRule.slice(4).toLowerCase()}`);
  parts.push(`friendlyPass=${rules.friendlyPassOk}`);
  return parts.join(',');
}
