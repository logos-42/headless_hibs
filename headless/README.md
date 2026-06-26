# headless v0.1.0

CLI tool and desktop Studio for **Hibs** — the twistor-inspired liquid neural network model.

## Installation

```bash
npm install -g headless
```

Or run directly:

```bash
npx headless --help
```

## Usage

```bash
# Show model info
headless info --ckpt models/twistor_quick.pt

# Interactive chat
headless chat --ckpt checkpoints/user_model_best.pt

# Start HTTP API server
headless serve --ckpt checkpoints/user_model_best.pt --port 8000

# Export model to mobile format
headless export --ckpt checkpoints/user_model_best.pt --format mobile

# Train a model
headless train --config configs/v16_6_50m.json

# Launch desktop Studio
headless studio
```

## Development

```bash
# CLI
cd headless
npm install
npm run build
node dist/cli/index.js --help

# Studio
cd headless/studio
npm install
npm run electron:dev
```

## Architecture

```
headless/          ← TypeScript (CLI + Studio UI)
  ├── src/         ← CLI source
  │   ├── index.ts        ← Commander entry, 5 commands
  │   ├── python.ts       ← Python subprocess bridge
  │   └── commands/       ← chat, serve, info, export, train
  ├── bridge/       ← Python helpers (info.py, train.py)
  └── studio/       ← Electron + React desktop app
      ├── electron/ ← Main process + PythonManager
      └── src/      ← React UI, 6 pages

hibs_lnn/          ← Python model (unchanged)
hibs_cli/          ← Python CLI backend (unchanged)
```
