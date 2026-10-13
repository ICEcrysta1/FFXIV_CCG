// 每个线程只加载一次上游模块，每份日志仍由 analyze 新建独立 Parser。
const {parentPort, workerData} = require('worker_threads');

function describeError(error) {
  return {name: error?.name || 'Error', message: String(error?.message ?? error),
    stack: typeof error?.stack === 'string' ? error.stack : undefined};
}

async function start() {
  const {bootstrap} = require('./bootstrap.cjs');
  bootstrap(workerData.analyzer_root, workerData.node_modules, {skipChdir: true});
  const {analyze} = require('./analyze.cjs');
  let busy = false;

  // 独立函数返回后，结果、Parser 和原始事件不再由当前调用栈持有。
  async function execute(request) {
    try {
      const analysis = await analyze({...workerData, ...request});
      parentPort.postMessage({type: 'result', id: request.id, analysis});
    } catch (error) {
      parentPort.postMessage({type: 'error', id: request.id, error: describeError(error)});
    }
  }

  parentPort.on('message', async request => {
    if (busy || request.type !== 'analyze') {
      parentPort.postMessage({type: 'fatal', error: describeError(new Error('分析线程收到重叠或无效请求'))});
      return;
    }
    busy = true;
    await execute(request);
    // 宿主必须等 idle 再复用，避免上份日志的大对象与下份日志同时存活。
    if (typeof global.gc === 'function') global.gc();
    busy = false;
    parentPort.postMessage({type: 'idle'});
  });
  parentPort.postMessage({type: 'ready'});
}

start().catch(error => {
  parentPort.postMessage({type: 'fatal', error: describeError(error)});
  process.exitCode = 1;
});
