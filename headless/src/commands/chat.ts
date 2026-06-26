import { createInterface } from 'node:readline'
import chalk from 'chalk'
import { startPythonServer, stopPythonServer, generateViaHTTP } from '../python.js'
import { type GenerateOptions } from '../types.js'

export interface ChatArgs {
  ckpt: string
  device?: string
  maxTokens: number
  temperature: number
  topK: number
  topP: number
}

export async function cmdChat(args: ChatArgs): Promise<void> {
  const banner = [
    `\n${'='.repeat(60)}`,
    `Hibs 0.16 (headless v0.1.0) interactive chat`,
    `Model: ${args.ckpt}`,
    `${'='.repeat(60)}`,
    `Type 'exit' or 'quit' to exit, 'clear' to clear screen`,
    `${'='.repeat(60)}\n`,
  ].join('\n')

  console.log(banner)

  const { port, process: serverProc } = await startPythonServer(args.ckpt, args.device)

  const rl = createInterface({ input: process.stdin, output: process.stdout })
  let done = false

  const ask = (): Promise<void> => {
    return new Promise((resolve) => {
      rl.question(chalk.cyan('>>> '), async (input) => {
        if (done) { resolve(); return }

        const trimmed = input.trim()
        if (!trimmed) { resolve(); return }

        const lower = trimmed.toLowerCase()
        if (lower === 'exit' || lower === 'quit') {
          console.log(chalk.yellow('Goodbye!'))
          done = true
          rl.close()
          resolve()
          return
        }
        if (lower === 'clear') {
          console.clear()
          resolve()
          return
        }

        console.log()
        const opts: GenerateOptions = {
          maxNewTokens: args.maxTokens,
          temperature: args.temperature,
          topK: args.topK,
          topP: args.topP,
        }
        try {
          const result = await generateViaHTTP(port, trimmed, opts)
          console.log(chalk.green(result))
        } catch (err) {
          console.error(chalk.red(`Error: ${(err as Error).message}`))
        }
        console.log()
        resolve()
      })
    })
  }

  while (!done) {
    await ask()
  }

  rl.close()
  await stopPythonServer(serverProc)
}
