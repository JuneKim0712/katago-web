/**
 * Absolute visit ceiling — exists only as a runaway guard.
 * In practice, analysis is bounded by maxTimeMs (the real cap) long before
 * this value is approached, even on the fastest small models.
 *
 * Set to 2 000 000 so that "0 = uncapped" in the Settings UI actually means
 * uncapped.  The previous value of 50 000 was visible to users who hadn't set
 * any limit, which contradicted the UI label.
 */
export const ENGINE_MAX_VISITS = 2_000_000;
export const ENGINE_MAX_TIME_MS = 300_000;

