import chalk from 'chalk'
import { getModelInfo } from '../python.js'

export interface InfoArgs {
  ckpt: string
  device?: string
}

export async function cmdInfo(args: InfoArgs): Promise<void> {
  const info = await getModelInfo(args.ckpt, args.device)

  console.log(chalk.cyan('\nModel Info:'))
  console.log(`  ${chalk.bold('path')}:      ${info.path}`)
  console.log(`  ${chalk.bold('format')}:    ${info.format}`)
  console.log(`  ${chalk.bold('device')}:    ${info.device}`)
  console.log(`  ${chalk.bold('size_mb')}:   ${info.sizeMb.toFixed(1)}`)
  if (info.vocabSize !== undefined) console.log(`  ${chalk.bold('vocab')}:     ${info.vocabSize}`)
  if (info.dModel !== undefined) console.log(`  ${chalk.bold('d_model')}:   ${info.dModel}`)
  if (info.nLayers !== undefined) console.log(`  ${chalk.bold('n_layers')}:  ${info.nLayers}`)
  console.log()
}
