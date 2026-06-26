import chalk from 'chalk'
import { startPythonServer, stopPythonServer } from '../python.js'

export interface ServeArgs {
  ckpt: string
  host: string
  port: number
  device?: string
}

export async function cmdServe(args: ServeArgs): Promise<void> {
  console.log(chalk.cyan(`\nStarting API server: http://${args.host}:${args.port}`))
  console.log(chalk.gray('  POST /generate  - text generation'))
  console.log(chalk.gray('  POST /chat      - multi-turn chat'))
  console.log(chalk.gray('  GET  /info      - model info'))
  console.log(chalk.gray('  GET  /health    - health check\n'))

  const { process: serverProc } = await startPythonServer(
    args.ckpt,
    args.device,
    args.port,
  )

  console.log(chalk.green(`✓ Server ready at http://${args.host}:${args.port}`))
  console.log(chalk.gray('Press Ctrl+C to stop\n'))

  return new Promise((resolve) => {
    const cleanup = async () => {
      console.log(chalk.yellow('\nShutting down...'))
      await stopPythonServer(serverProc)
      resolve()
    }

    process.on('SIGINT', cleanup)
    process.on('SIGTERM', cleanup)

    serverProc.on('exit', (code) => {
      if (code !== 0) {
        console.error(chalk.red(`Server exited with code ${code}`))
      }
      resolve()
    })
  })
}
