// 每个进程使用一套线程参数；在创建任何模型实例前设置。
let numThreads = 4;
export function configureInference(threads) {
  if (!Number.isInteger(threads) || threads < 1 || threads > 36) throw new Error('numThreads 必须为 1–36 的整数');
  numThreads = threads;
}
export function inferenceOptions() { return { numThreads }; }
