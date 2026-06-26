import chalk from 'chalk'
import { spawnTrain } from '../python.js'

export interface TrainArgs {
  config: string
}

export async function cmdTrain(args: TrainArgs): Promise<void> {
  console.log(chalk.cyan(`Training with config: ${args.config}`))

  const proc = spawnTrain(args.config)

  return new Promise((resolve, reject) => {
    proc.on('close', (code) => {
      if (code === 0) {
        console.log(chalk.green('\n✓ Training complete'))
        resolve()
      } else {
        reject(new Error(`Training failed with code ${code}`))
      }
    })
    proc.on('error', reject)
  })
}
