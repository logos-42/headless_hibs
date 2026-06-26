export interface GenerateOptions {
  maxNewTokens?: number
  temperature?: number
  topK?: number
  topP?: number
}

export interface ModelInfo {
  path: string
  format: string
  device: string
  sizeMb: number
  vocabSize?: number
  dModel?: number
  nLayers?: number
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export interface TrainingProgress {
  progress: number
  message: string
}

export function version(): string {
  return '0.1.0'
}
