// 测试用的替身：把 ssh / rsync 调用改在本机临时 HOME 里执行，不连网络。
import { spawn } from 'node:child_process';
import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { normalizeConfig } from '../src/config.mjs';

function runLocal(bin, args, { capture = false, cwd, env = process.env } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(bin, args, { cwd, env, stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '';
    child.stdout.on('data', (s) => { out += s; });
    child.stderr.resume();
    child.on('close', (code) => (code === 0 ? resolve(capture ? out.trim() : '') : reject(Object.assign(new Error(`${bin} 退出码 ${code}`), { code }))));
  });
}

/**
 * 替身执行器：remoteCall 发出的 ssh 在 home 下用 sh 执行（HOME=home），rsync 的 host:路径 换成 home 下的路径。
 * down：一律连不上；corruptPush：推单个文件时写入坏内容（模拟传输损坏）。calls 记录每次调用的程序与参数。
 */
export function localExec(home, { down = false, corruptPush = false } = {}) {
  const calls = [];
  const exec = async (bin, args, options = {}) => {
    calls.push([bin, ...args]);
    if (down) throw Object.assign(new Error(`${bin} 失败`), { code: 255 });
    if (bin === 'ssh') return runLocal('sh', ['-c', args[args.length - 1]], { ...options, cwd: home, env: { ...process.env, HOME: home } });
    if (bin === 'rsync') {
      const [src, dst] = args.slice(args.indexOf('--') + 1);
      const map = (p) => (!p.startsWith('/') && p.includes(':') ? join(home, p.slice(p.indexOf(':') + 1)) : p);
      if (corruptPush && dst.includes(':') && !src.endsWith('/')) { await writeFile(map(dst), '坏'); return ''; }
      return runLocal('rsync', ['-a', '--', map(src), map(dst)], options);
    }
    throw new Error(`替身不认识 ${bin}`);
  };
  exec.calls = calls;
  return exec;
}

/** 测试用配置：在示例配置的基础上覆盖字段，再走一遍 normalizeConfig。 */
export function testConfig(overrides = {}) {
  return normalizeConfig({ timeZone: 'Asia/Shanghai', ...overrides });
}

export const tmp = (prefix) => mkdtemp(join(tmpdir(), prefix));
