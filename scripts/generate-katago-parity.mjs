// Generate independent reference data with an official KataGo executable.
// Usage: node scripts/generate-katago-parity.mjs [katago.exe] [model.bin.gz] [output.json] [positions.json]
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdtempSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';

const defaultExecutable = [
  process.env.KATAGO_BIN,
  'C:/Users/junwo/Downloads/katago-v1.18.1/katago.exe',
  'C:/Users/junwo/AppData/Local/Temp/baduk-katago-native-v1.18.1/katago.exe',
].find(p => p && existsSync(p));

const defaultModel = [
  process.env.KATAGO_TEST_MODEL,
  process.env.KATAGO_MODEL,
  'C:/Users/junwo/Downloads/kata1-tf2-b10c384-s2941M-d5872M.bin.gz',
].find(p => p && existsSync(p));

const executable = process.argv[2] || defaultExecutable;
const model = process.argv[3] || defaultModel;
const output = process.argv[4] || resolve('test/fixtures/katago-model17-native.json');
const positionsPath = process.argv[5] || resolve('test/fixtures/katago-parity-positions.json');

if (!executable || !existsSync(executable)) {
  throw new Error(`KataGo executable not found: ${executable}`);
}
if (!model || !existsSync(model)) {
  throw new Error(`KataGo model not found: ${model}`);
}

const positions = JSON.parse(readFileSync(positionsPath, 'utf8'));
const directory = mkdtempSync(join(tmpdir(), 'katago-parity-'));

try {
  const config = join(directory, 'parity.cfg');
  writeFileSync(
    config,
    'numSearchThreads = 1\nnumEigenThreadsPerModel = 2\nnnMaxBatchSize = 1\nnnCacheSizePowerOfTwo = 10\nnnMutexPoolSizePowerOfTwo = 8\nnnRandomize = false\nlogAllGTPCommunication = false\nlogSearchInfo = false\nlogToStderr = false\nrules = japanese\n'
  );

  const version = spawnSync(executable, ['version'], {
    cwd: dirname(executable),
    encoding: 'utf8',
    timeout: 10000,
    windowsHide: true,
  });
  if (version.status !== 0) throw new Error(version.stderr || String(version.error));

  let id = 0;
  const commands = [];
  const queries = [];
  const send = (command) => {
    commands.push(`${++id} ${command}`);
    return id;
  };

  send('boardsize 19');
  for (const position of positions) {
    send('clear_board');
    send(`kata-set-rules ${position.rules}`);
    send(`komi ${position.komi}`);
    for (const [player, move] of position.moves) {
      send(`play ${player} ${move}`);
    }
    queries.push({ id: send(`kata-raw-nn ${position.currentPlayer} 0 0.2`), position });
  }
  send('quit');

  const result = spawnSync(executable, ['gtp', '-model', model, '-config', config], {
    cwd: dirname(executable),
    input: commands.join('\n') + '\n',
    encoding: 'utf8',
    timeout: 180000,
    maxBuffer: 32 * 1024 * 1024,
    windowsHide: true,
  });

  if (result.status !== 0 || /^\?/m.test(result.stdout)) {
    throw new Error(`${result.error ?? ''}\n${result.stderr}\n${result.stdout}`);
  }

  const replies = new Map(
    result.stdout
      .trim()
      .split(/\r?\n\r?\n/)
      .map((reply) => {
        const match = /^=(\d+)\s*([\s\S]*)$/.exec(reply);
        if (!match) return null;
        return [Number(match[1]), match[2]];
      })
      .filter(Boolean)
  );

  const cases = queries.map(({ id, position }) => {
    const rawTokens = (replies.get(id) ?? '').trim().split(/\s+/);
    const raw = {};
    for (let i = 0; i < rawTokens.length; ) {
      const key = rawTokens[i++];
      const count = key === 'policy' || key === 'whiteOwnership' ? 361 : 1;
      const values = rawTokens.slice(i, i + count).map((value) => (value.toUpperCase() === 'NAN' ? null : Number(value)));
      if (values.length !== count || values.some((value) => value !== null && !Number.isFinite(value))) {
        throw new Error(`Malformed ${key} in query ${id}`);
      }
      raw[key] = count === 1 ? values[0] : values;
      i += count;
    }
    if (!raw.policy || !raw.whiteOwnership || !Number.isFinite(raw.whiteLoss)) {
      throw new Error(`Missing native outputs in query ${id}`);
    }
    return { ...position, native: raw };
  });

  writeFileSync(
    output,
    JSON.stringify(
      {
        engine: version.stdout.trim(),
        modelSha256: createHash('sha256').update(readFileSync(model)).digest('hex'),
        symmetry: 0,
        policyOptimism: 0.2,
        boardSize: 19,
        cases,
      },
      null,
      2
    ) + '\n'
  );

  console.log(`Generated ${cases.length} native reference positions in: ${output}`);
} finally {
  const target = resolve(directory);
  if (dirname(target) !== resolve(tmpdir()) || !basename(target).startsWith('katago-parity-')) {
    throw new Error('Refusing to remove a directory outside temporary workspace');
  }
  rmSync(target, { recursive: true, force: true });
}
