/**
 * bundle-python.mjs
 * =================
 * 构建时脚本：下载便携 Python + 安装 pip 依赖
 *
 * 用法:
 *   node headless/scripts/bundle-python.mjs
 *
 * 输出:
 *   headless/studio/python-dist/   ← 带依赖的便携 Python
 */

import { createWriteStream, existsSync, mkdirSync, readdirSync, renameSync, rmSync, statSync, writeFileSync } from 'fs'
import { readFile } from 'fs/promises'
import { execSync, spawn } from 'child_process'
import { resolve, dirname, basename } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const ROOT = resolve(__dirname, '..')
const STUDIO_DIR = resolve(ROOT, 'studio')
const PYTHON_DIST = resolve(STUDIO_DIR, 'python-dist')
const REQS = resolve(STUDIO_DIR, 'python-requirements.txt')

const PLATFORM = process.platform
const ARCH = process.arch
const PVERSION = '3.12.8'
const PREL = '20250112'

function platformTag() {
  if (PLATFORM === 'win32')  return 'x86_64-pc-windows-msvc-shared'
  if (PLATFORM === 'darwin' && ARCH === 'arm64') return 'aarch64-apple-darwin'
  if (PLATFORM === 'darwin') return 'x86_64-apple-darwin'
  if (PLATFORM === 'linux')  return 'x86_64-unknown-linux-gnu'
  throw new Error(`不支持平台: ${PLATFORM} ${ARCH}`)
}

function pythonExe(dir) {
  return PLATFORM === 'win32'
    ? resolve(dir, 'python.exe')
    : resolve(dir, 'bin', 'python3')
}

function sh(cmd, opts = {}) {
  console.log(`  $ ${cmd}`)
  return execSync(cmd, { encoding: 'utf-8', stdio: 'pipe', ...opts })
}

async function download(url, dest) {
  console.log(`⬇ 下载 ${basename(url)} ...`)
  const resp = await fetch(url)
  if (!resp.ok) throw new Error(`下载失败: ${resp.status} ${resp.statusText}`)
  const fileStream = createWriteStream(dest)
  await new Promise((resolve, reject) => {
    resp.body.pipe(fileStream)
    resp.body.on('error', reject)
    fileStream.on('finish', resolve)
  })
}

function extractTarGz(src, dest) {
  console.log(`📦 解压到 ${dest} ...`)
  mkdirSync(dest, { recursive: true })
  if (PLATFORM === 'win32') {
    // Windows 10+ 自带 tar.exe
    sh(`tar -xf "${src}" -C "${dest}"`, { shell: true })
  } else {
    sh(`tar -xzf "${src}" -C "${dest}"`)
  }
}

function getDirSize(dir) {
  let total = 0
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = resolve(dir, e.name)
    if (e.isDirectory()) total += getDirSize(p)
    else total += statSync(p).size
  }
  return total
}

async function main() {
  if (existsSync(PYTHON_DIST)) {
    const entries = readdirSync(PYTHON_DIST).filter(e => !e.startsWith('.'))
    if (entries.length > 0) {
      console.log('⚠ python-dist 已存在，跳过捆绑。如需重新捆绑请删除。')
      return
    }
  }

  const tag = platformTag()
  const archive = `cpython-${PVERSION}+${PREL}-${tag}-install_only.tar.gz`
  const url = `https://github.com/indygreg/python-build-standalone/releases/download/${PREL}/${archive}`
  const cacheDir = resolve(ROOT, '.cache')
  const cachePath = resolve(cacheDir, archive)
  mkdirSync(cacheDir, { recursive: true })

  if (!existsSync(cachePath)) await download(url, cachePath)
  else console.log(`✔ 使用缓存 ${archive}`)

  extractTarGz(cachePath, PYTHON_DIST)

  // 扁平化目录结构（tarball 可能嵌套一层）
  let py = pythonExe(PYTHON_DIST)
  if (!existsSync(py)) {
    const entries = readdirSync(PYTHON_DIST).filter(e => !e.startsWith('.'))
    for (const e of entries) {
      const sub = resolve(PYTHON_DIST, e)
      if (statSync(sub).isDirectory() && existsSync(pythonExe(sub))) {
        for (const f of readdirSync(sub)) {
          renameSync(resolve(sub, f), resolve(PYTHON_DIST, f))
        }
        rmSync(sub, { recursive: true })
        break
      }
    }
  }

  py = pythonExe(PYTHON_DIST)
  if (!existsSync(py)) {
    console.error('❌ 解压后找不到 python 可执行文件')
    console.log('  内容:', readdirSync(PYTHON_DIST).slice(0, 20))
    process.exit(1)
  }
  console.log(`✔ 便携 Python: ${py}`)

  // 安装 pip
  console.log('📦 安装 pip ...')
  sh(`"${py}" -m ensurepip --upgrade`)

  // 安装依赖
  if (existsSync(REQS)) {
    const reqs = (await readFile(REQS, 'utf-8')).trim()
    console.log('📦 安装 pip 包:\n', reqs)
    const pip = PLATFORM === 'win32'
      ? resolve(dirname(py), 'Scripts', 'pip.exe')
      : resolve(dirname(py), 'bin', 'pip')
    sh(`"${pip}" install -r "${REQS}"`, { timeout: 600_000 })
  }

  // 验证
  console.log('🔍 验证安装...')
  try {
    const out = sh(`"${py}" -c "import torch, fastapi, uvicorn, numpy; print('验证通过')"`)
    console.log('  ', out.trim())
  } catch (e) {
    console.warn('  ⚠ 部分包导入失败:', e.message)
  }

  const size = (getDirSize(PYTHON_DIST) / 1e6).toFixed(0)
  console.log(`✔ 捆绑完成: ${PYTHON_DIST} (${size} MB)`)

  // 写入版本文件供 Electron 检测
  writeFileSync(
    resolve(PYTHON_DIST, '.bundle-info.json'),
    JSON.stringify({ version: PVERSION, tag, date: new Date().toISOString() }, null, 2)
  )
}

main().catch(err => {
  console.error('❌ 捆绑失败:', err)
  process.exit(1)
})
