#!/usr/bin/env node
// postinstall: fetch the Python Synth CLI via install.sh
// Honest shim: this does not implement Synth in JS. It runs the same
// install.sh the curl-one-liner uses, then leaves the binary on PATH
// via the user's shell profile (see install.sh output).

const { spawnSync } = require('child_process');
const path = require('path');
const https = require('https');
const fs = require('fs');
const os = require('os');

const INSTALLER_URL = 'https://raw.githubusercontent.com/Mercphobia/Synth/main/install.sh';

function fail(msg) {
  console.error(`[synth-cli] ${msg}`);
  console.error('[synth-cli] postinstall failed. You can install manually:');
  console.error('  curl -fsSL https://raw.githubusercontent.com/Mercphobia/Synth/main/install.sh | sh');
  // Exit 0: npm install should not hard-fail the user's project for this shim.
  process.exit(0);
}

const tmpScript = path.join(os.tmpdir(), `synth-install-${Date.now()}.sh`);

https.get(INSTALLER_URL, (res) => {
  if (res.statusCode !== 200) {
    fs.unlinkSync(tmpScript);
    return fail(`installer download returned HTTP ${res.statusCode}`);
  }
  const file = fs.createWriteStream(tmpScript);
  res.pipe(file);
  file.on('finish', () => {
    file.close();
    const result = spawnSync('sh', [tmpScript], { stdio: 'inherit' });
    fs.unlinkSync(tmpScript);
    if (result.status !== 0) {
      return fail('installer script exited with an error');
    }
    console.log('[synth-cli] Python synth CLI installed to ~/.synth-dist/bin/synth');
    console.log('[synth-cli] Ensure ~/.synth-dist/bin is on your PATH.');
  });
}).on('error', (err) => {
  fail(`installer download error: ${err.message}`);
});