#!/usr/bin/env node
// Thin wrapper around the installed synth CLI
// This is NOT a JavaScript implementation - it just forwards to the Python binary

const { spawn } = require('child_process');
const path = require('path');
const os = require('os');

// Find the installed synth binary (matches install.sh layout)
const candidates = [
  process.env.SYNTH_BIN, // explicit override
  path.join(os.homedir(), '.synth-dist', 'bin', 'synth'),
  path.join(os.homedir(), '.synth-dist', 'synth', '.venv', 'bin', 'synth'),
].filter(Boolean);

const synthBin = candidates.find((p) => {
  try {
    require('fs').accessSync(p, require('fs').constants.X_OK);
    return true;
  } catch {
    return false;
  }
});

if (!synthBin) {
  console.error('synth binary not found. Install it with:');
  console.error('  curl -fsSL https://raw.githubusercontent.com/Mercphobia/Synth/main/install.sh | sh');
  process.exit(1);
}

const synth = spawn(synthBin, process.argv.slice(2), {
  stdio: 'inherit',
});

synth.on('error', (err) => {
  console.error('Error running synth:', err.message);
  process.exit(1);
});

synth.on('exit', (code) => {
  process.exit(code || 0);
});
