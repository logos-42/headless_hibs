import { Command } from 'commander'
import { version } from './types.js'
import { cmdChat } from './commands/chat.js'
import { cmdServe } from './commands/serve.js'
import { cmdInfo } from './commands/info.js'
import { cmdExport } from './commands/export.js'
import { cmdTrain } from './commands/train.js'

const program = new Command()

program
  .name('headless')
  .description('Hibs model CLI — local, private, personal')
  .version(version())

program
  .command('chat')
  .description('Interactive chat with your model')
  .requiredOption('--ckpt <path>', 'checkpoint path')
  .option('--device <device>', 'cuda / cpu')
  .option('--max-tokens <n>', 'max new tokens', Number, 128)
  .option('--temperature <n>', 'sampling temperature', Number, 1.0)
  .option('--top-k <n>', 'top-k filtering', Number, 50)
  .option('--top-p <n>', 'nucleus (top-p) filtering', Number, 0.9)
  .action(async (opts) => {
    try {
      await cmdChat({
        ckpt: opts.ckpt,
        device: opts.device,
        maxTokens: opts.maxTokens,
        temperature: opts.temperature,
        topK: opts.topK,
        topP: opts.topP,
      })
      process.exit(0)
    } catch (err) {
      console.error(`Error: ${(err as Error).message}`)
      process.exit(1)
    }
  })

program
  .command('serve')
  .description('Start HTTP API server')
  .requiredOption('--ckpt <path>', 'checkpoint path')
  .option('--host <host>', 'bind address', '0.0.0.0')
  .option('--port <n>', 'HTTP port', Number, 8000)
  .option('--device <device>', 'cuda / cpu')
  .action(async (opts) => {
    try {
      await cmdServe({
        ckpt: opts.ckpt,
        host: opts.host,
        port: opts.port,
        device: opts.device,
      })
      process.exit(0)
    } catch (err) {
      console.error(`Error: ${(err as Error).message}`)
      process.exit(1)
    }
  })

program
  .command('info')
  .description('Show model info')
  .requiredOption('--ckpt <path>', 'checkpoint path')
  .option('--device <device>', 'cuda / cpu')
  .action(async (opts) => {
    try {
      await cmdInfo({ ckpt: opts.ckpt, device: opts.device })
    } catch (err) {
      console.error(`Error: ${(err as Error).message}`)
      process.exit(1)
    }
  })

program
  .command('export')
  .description('Export model to various formats')
  .requiredOption('--ckpt <path>', 'checkpoint path')
  .option('--format <fmt>', 'output format', 'all')
  .option('--output <dir>', 'output directory', 'exported')
  .action(async (opts) => {
    try {
      await cmdExport({
        ckpt: opts.ckpt,
        format: opts.format,
        output: opts.output,
      })
    } catch (err) {
      console.error(`Error: ${(err as Error).message}`)
      process.exit(1)
    }
  })

program
  .command('train')
  .description('Train a model')
  .option('--config <path>', 'training config path', 'configs/v16_6_50m.json')
  .action(async (opts) => {
    try {
      await cmdTrain({ config: opts.config })
    } catch (err) {
      console.error(`Error: ${(err as Error).message}`)
      process.exit(1)
    }
  })

program.parse()
