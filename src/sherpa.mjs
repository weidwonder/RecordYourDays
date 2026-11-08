// sherpa-onnx-node 的唯一加载入口。
// 它是 CommonJS 包，ESM 的具名导入拿不到它的导出（得到 undefined，调用时才炸），必须取 default。
// 原生 addon 只在真正需要时加载，纯函数与单测不触碰它。
let sherpa;
export async function loadSherpa() {
  sherpa ??= (await import('sherpa-onnx-node')).default;
  return sherpa;
}
