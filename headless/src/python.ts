import { spawn, type ChildProcess } from 'node:child_process'
import { createServer, type AddressInfo } from 'node:net'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { type GenerateOptions, type ModelInfo } from './types.js'

const __dirname = dirname(fileURLToPath(import.meta.url))
export const HEADLESS_ROOT = resolve(__dirname, '..', '..')
export const PROJECT_ROOT = resolve(HEADLESS_ROOT, '..')

function findPython(): string {
  return process.platform === 'win32' ? 'python' : 'python3'
}

function findFreePort(): Promise<number> {
  return new Promise((resolvePort, reject) => {
    const srv = createServer()
    srv.unref()
    srv.listen(0, '127.0.0.1', () => {
      const port = (srv.address() as AddressInfo).port
      srv.close(() => resolvePort(port))
    })
    srv.on('error', reject)
  })
}

function absPath(p: string): string {
  return resolve(PROJECT_ROOT, p)
}

export async function startPythonServer(
  ckpt: string,
  device?: string,
  preferredPort?: number,
): Promise<{ port: number; process: ChildProcess }> {
  const port = preferredPort ?? await findFreePort()

  const py = findPython()
  const args = [
    '-m', 'hibs_cli', 'serve',
    '--ckpt', absPath(ckpt),
    '--port', String(port),
    '--host', '127.0.0.1',
  ]
  if (device) args.push('--device', device)

  const proc = spawn(py, args, {
    cwd: PROJECT_ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  })

  proc.stderr?.on('data', (data: Buffer) => {
    process.stderr.write(data)
  })

  // 等待 health check 通过
  await waitForHealth(port, 60_000)

  return { port, process: proc }
}

export async function waitForHealth(port: number, timeoutMs = 30_000): Promise<void> {
  const start = Date.now()
  const baseUrl = `http://127.0.0.1:${port}`

  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(2000) })
      if (res.ok) return
    } catch { /* server not ready yet */ }
    await sleep(500)
  }
  throw new Error(`Python server did not start within ${timeoutMs}ms`)
}

export async function generateViaHTTP(
  port: number,
  prompt: string,
  opts?: GenerateOptions,
): Promise<string> {
  const res = await fetch(`http://127.0.0.1:${port}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt,
      max_new_tokens: opts?.maxNewTokens ?? 128,
      temperature: opts?.temperature ?? 1.0,
      top_k: opts?.topK ?? 50,
      top_p: opts?.topP ?? 0.9,
    }),
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`generate failed (${res.status}): ${text}`)
  }

  const data = await res.json() as { generated: string }
  return data.generated
}

export async function stopPythonServer(proc: ChildProcess): Promise<void> {
  if (proc.killed) return

  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      proc.kill('SIGKILL')
      resolve()
    }, 5000)

    proc.on('exit', () => {
      clearTimeout(timer)
      resolve()
    })

    if (process.platform === 'win32') {
      spawn('taskkill', ['/pid', String(proc.pid), '/f', '/t'], { windowsHide: true })
    } else {
      proc.kill('SIGTERM')
    }
  })
}

export function execPython(
  scriptArgs: string[],
  cwd: string = PROJECT_ROOT,
): Promise<string> {
  const py = findPython()
  return new Promise((resolveOut, reject) => {
    const proc = spawn(py, scriptArgs, {
      cwd,
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
    })
    let stdout = ''
    let stderr = ''
    proc.stdout?.on('data', (d: Buffer) => { stdout += d.toString() })
    proc.stderr?.on('data', (d: Buffer) => { stderr += d.toString() })
    proc.on('close', (code) => {
      if (code === 0) resolveOut(stdout.trim())
      else reject(new Error(`python exited ${code}: ${stderr}`))
    })
    proc.on('error', reject)
  })
}

export function execPythonStream(
  scriptArgs: string[],
  cwd: string,
  onLine: (line: string) => void,
): Promise<number> {
  const py = findPython()
  return new Promise((resolve, reject) => {
    const proc = spawn(py, scriptArgs, {
      cwd,
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
    })

    proc.stdout?.on('data', (data: Buffer) => {
      const lines = data.toString().split('\n').filter(Boolean)
      for (const line of lines) onLine(line)
    })

    let stderr = ''
    proc.stderr?.on('data', (d: Buffer) => { stderr += d.toString() })

    proc.on('close', (code) => {
      if (code === 0) resolve(code)
      else reject(new Error(`python exited ${code}: ${stderr}`))
    })
    proc.on('error', reject)
  })
}

export async function getModelInfo(ckpt: string, device?: string): Promise<ModelInfo> {
  const ckptAbs = absPath(ckpt)

  // inline Python script (replaces bridge/info.py)
  const code = `
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(p.parents[1]))
import torch
ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
info = {"path": str(p), "sizeMb": round(p.stat().st_size / 1e6, 1)}
s = p.suffix.lower()
info["format"] = "onnx" if s == ".onnx" else "mobile" if s == ".ptl" else "torchscript"
info["device"] = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else ("cuda" if torch.cuda.is_available() else "cpu")
if isinstance(ckpt, dict):
    cfg = ckpt.get("config") or ckpt.get("model_config", {}) or {}
    info["vocabSize"] = cfg.get("vocab_size")
    info["dModel"] = cfg.get("d_model")
    info["nLayers"] = cfg.get("n_layers")
print(json.dumps(info, ensure_ascii=False))
  `.trim()

  const py = findPython()
  return new Promise((resolveOut, reject) => {
    const proc = spawn(py, ['-c', code, ckptAbs, device ?? ''], {
      cwd: PROJECT_ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
    })
    let stdout = ''
    let stderr = ''
    proc.stdout?.on('data', (d: Buffer) => { stdout += d.toString() })
    proc.stderr?.on('data', (d: Buffer) => { stderr += d.toString() })
    proc.on('close', (code) => {
      if (code === 0) {
        try { resolveOut(JSON.parse(stdout.trim()) as ModelInfo) }
        catch { reject(new Error(`info: invalid JSON: ${stdout.slice(0, 200)}`)) }
      } else reject(new Error(`info: python exited ${code}: ${stderr}`))
    })
    proc.on('error', reject)
  })
}

export function spawnExport(ckpt: string, format: string, output: string): ChildProcess {
  const py = findPython()
  const script = resolve(PROJECT_ROOT, 'hibs_export', 'export_hibs_0_16.py')
  return spawn(py, [script, '--ckpt', absPath(ckpt), '--format', format, '--output', output], {
    cwd: PROJECT_ROOT,
    stdio: 'inherit',
    windowsHide: true,
  })
}

export function spawnTrain(config: string): ChildProcess {
  const py = findPython()
  const script = resolve(PROJECT_ROOT, 'scripts', 'train_hibs_0_16.py')
  return spawn(py, [script, '--config', config], {
    cwd: PROJECT_ROOT,
    stdio: 'inherit',
    windowsHide: true,
  })
}

function sleep(ms: number): Promise<void> {
  return new Promise(r => setTimeout(r, ms))
}
