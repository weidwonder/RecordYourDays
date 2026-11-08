// 拼远程 shell 命令用的引用与路径校验。
export const quote = (s) => "'" + String(s).replaceAll("'", "'\\''") + "'";

/** 远程路径（~/ 或 / 开头的普通路径，不能越级）→ shell 片段；~/ 展开为远程的 $HOME。 */
export function homePath(p) {
  if (typeof p !== 'string' || !/^(~\/|\/)[a-zA-Z0-9_./-]+$/.test(p) || p.split('/').some((x) => x === '..' || x === '.'))
    throw new Error('远程路径必须是 ~/ 或 / 开头的普通路径，不能越级');
  return p.startsWith('~/') ? '"$HOME"/' + quote(p.slice(2)) : quote(p);
}
