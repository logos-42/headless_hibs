import chalk from 'chalk'
import { spawnExport } from '../python.js'

export interface ExportArgs {
  ckpt: string
  format: string
  output: string
}

export async function cmdExport(args: ExportArgs): Promise<void> {
  console.log(chalk.cyan(`Exporting model to ${args.format} format...`))

  const proc = spawnExport(args.ckpt, args.format, args.output)

  return new Promise((resolve, reject) => {
    proc.on('close', (code) => {
      if (code === 0) {
        console.log(chalk.green(`\n✓ Export complete → ${args.output}/`))
        resolve()
      } else {
        reject(new Error(`Export failed with code ${code}`))
      }
    })
    proc.on('error', reject)
  })
}
