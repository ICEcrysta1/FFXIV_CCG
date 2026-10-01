// Node 子进程通过标准输入和标准输出交换分析数据，正式合并和原子保存由 Python 管理。
const {Socket} = require('net');
const {Worker, isMainThread, parentPort} = require('worker_threads');

if (!isMainThread) {
  // 独立线程读请求并监视管道；分析主线程卡住时也能检测 Python 退出。
  const input = new Socket({fd: 0, readable: true, writable: false});
  const chunks = [];
  let sent = false;
  input.on('data', chunk => {
    if (!sent) {
      const newline = chunk.indexOf(10);
      chunks.push(newline < 0 ? chunk : chunk.subarray(0, newline));
      if (newline >= 0) {
        parentPort.postMessage(Buffer.concat(chunks).toString('utf8'));
        chunks.length = 0;
        sent = true;
      }
    }
  });
  const stop = () => {
    // Python 正常回收或被强制结束都会关闭全部写端，不依赖跨进程信号权限。
    process.kill(process.pid, 'SIGKILL');
  };
  input.on('end', stop);
  input.on('error', stop);
} else {
  const owner = new Worker(__filename);
  owner.on('error', error => {
    console.error(error.stack);
    process.exit(1);
  });
  owner.once('message', requestJson => (async () => {
    const {bootstrap} = require('./bootstrap.cjs');
    const request = JSON.parse(requestJson);
    bootstrap(request.analyzer_root, request.node_modules);
    const {analyze} = require('./analyze.cjs');
    const analysis = await analyze(request);
    process.stdout.write(JSON.stringify(analysis) + '\n', () => process.exit(0));
  })().catch(error => {
    console.error(error.stack);
    process.exit(1);
  }));
}
