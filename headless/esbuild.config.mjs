import esbuild from 'esbuild'

await esbuild.build({
  entryPoints: ['src/index.ts'],
  bundle: true,
  platform: 'node',
  target: 'node18',
  outfile: 'dist/cli/index.js',
  format: 'esm',
  banner: {
    js: '#!/usr/bin/env node',
  },
  external: [
    'commander',
    'chalk',
    'ora',
    'execa',
  ],
})
