// 主线程只处理有界 NDJSON 协议、截止时间和生命周期；分析在独立线程执行。
const path = require('path');
const readline = require('readline');
const {Worker} = require('worker_threads');

function describeError(error) {
  return {name: error?.name || 'Error', message: String(error?.message ?? error),
    stack: typeof error?.stack === 'string' ? error.stack : undefined};
}

function startPool() {
  let configuration = null;
  let capacity = 0;
  let shuttingDown = false;
  let stopped = false;
  const workers = new Set();
  const jobs = new Map();
  const queue = [];

  function send(message) {
    if (!stopped) process.stdout.write(JSON.stringify(message) + '\n');
  }

  function sendError(id, error) {
    send({type: 'error', id, error: describeError(error)});
  }

  function finishJob(job, response) {
    if (!jobs.delete(job.request.id)) return;
    clearTimeout(job.timer);
    send(response);
  }

  function retire(slot) {
    if (slot.state === 'retiring') return;
    slot.state = 'retiring';
    // terminate 能中断同步 parseEvents；不影响其他线程正在处理的请求。
    slot.worker.terminate().catch(error => console.error(error.stack || error));
  }

  function abortJob(job, name, message) {
    const error = new Error(message);
    error.name = name;
    finishJob(job, {type: 'error', id: job.request.id, error: describeError(error)});
    if (job.slot) retire(job.slot);
    else {
      const index = queue.indexOf(job);
      if (index >= 0) queue.splice(index, 1);
    }
    dispatch();
  }

  function checkShutdown() {
    if (!shuttingDown || jobs.size || workers.size && [...workers].some(slot =>
      slot.state !== 'idle' && slot.state !== 'retiring')) return;
    stopped = true;
    for (const slot of workers) retire(slot);
    // 等待已完成结果刷新到管道，随后退出会回收全部工作线程。
    process.stdout.write('', () => process.exit(0));
  }

  function createWorker(job) {
    const worker = new Worker(path.join(__dirname, 'pool_worker.cjs'), {
      workerData: configuration, stdout: true, stderr: true,
    });
    const slot = {worker, state: 'starting', job};
    job.slot = slot;
    workers.add(slot);
    // 上游偶发 console 输出不能污染机器协议。
    worker.stdout.pipe(process.stderr, {end: false});
    worker.stderr.pipe(process.stderr, {end: false});
    worker.on('message', message => {
      if (slot.state === 'retiring' || stopped) return;
      if (message.type === 'ready' && slot.state === 'starting') {
        slot.state = 'busy';
        worker.postMessage(slot.job.request);
      } else if ((message.type === 'result' || message.type === 'error') && slot.state === 'busy'
          && message.id === slot.job?.request.id) {
        finishJob(slot.job, message);
        slot.job = null;
        slot.state = 'finishing';
      } else if (message.type === 'idle' && slot.state === 'finishing') {
        slot.state = 'idle';
        const rss = process.memoryUsage().rss;
        if (rss > configuration.memory_budget_mb * 1024 * 1024) {
          // 预算是软上限：只回收空闲线程，正在分析的大日志允许短时超过预算。
          capacity = Math.max(1, Math.min(capacity - 1, workers.size - 1));
          console.error(`[action-quality-pool] memory rss_mb=${Math.ceil(rss / 1024 / 1024)}`
            + ` budget_mb=${configuration.memory_budget_mb} capacity=${capacity}; retire idle worker`);
          retire(slot);
        }
        dispatch();
      } else if (message.type === 'fatal') {
        if (slot.job) finishJob(slot.job, {type: 'error', id: slot.job.request.id, error: message.error});
        retire(slot);
      } else {
        if (slot.job) finishJob(slot.job, {type: 'error', id: slot.job.request.id,
          error: describeError(new Error('分析线程返回无效响应'))});
        retire(slot);
      }
    });
    worker.on('error', error => {
      if (slot.job) finishJob(slot.job, {type: 'error', id: slot.job.request.id, error: describeError(error)});
      retire(slot);
    });
    worker.on('exit', code => {
      workers.delete(slot);
      if (slot.job) finishJob(slot.job, {type: 'error', id: slot.job.request.id,
        error: describeError(new Error(`分析线程提前退出 (${code})`))});
      dispatch();
    });
  }

  function dispatch() {
    if (stopped || !configuration) return;
    while (queue.length) {
      const idle = [...workers].find(slot => slot.state === 'idle');
      if (idle) {
        const job = queue.shift();
        idle.job = job;
        idle.state = 'busy';
        job.slot = idle;
        idle.worker.postMessage(job.request);
      } else if (workers.size < capacity) {
        const job = queue.shift();
        try {
          createWorker(job);
        } catch (error) {
          finishJob(job, {type: 'error', id: job.request.id, error: describeError(error)});
        }
      } else break;
    }
    checkShutdown();
  }

  function initialize(request) {
    if (configuration || request.id !== 0) throw new Error('分析宿主只接受一次 id=0 的初始化');
    if (!Number.isSafeInteger(request.workers) || request.workers < 1) {
      throw new Error('分析线程数必须为有效正整数');
    }
    if (!Number.isFinite(request.timeout_seconds) || request.timeout_seconds <= 0
        || request.timeout_seconds * 1000 > 2147483647) throw new Error('分析超时必须为有效正数');
    const memoryBudget = request.memory_budget_mb ?? 2048;
    if (!Number.isFinite(memoryBudget) || memoryBudget <= 0) throw new Error('分析内存预算必须为有效正数');
    if (typeof request.analyzer_root !== 'string' || typeof request.node_modules !== 'string') {
      throw new Error('分析器和依赖目录必须为字符串');
    }
    const analyzerRoot = path.resolve(request.analyzer_root);
    const nodeModules = path.resolve(request.node_modules);
    // chdir 只在主线程初始化时执行一次，此后所有线程共享相同目录。
    process.chdir(analyzerRoot);
    configuration = {analyzer_root: analyzerRoot, node_modules: nodeModules,
      timeout_seconds: request.timeout_seconds, memory_budget_mb: memoryBudget, workers: request.workers};
    capacity = request.workers;
    send({id: 0, type: 'ready'});
  }

  function receive(line) {
    let request;
    try {
      request = JSON.parse(line);
      if (!request || typeof request !== 'object' || Array.isArray(request)) throw new Error('请求必须为 JSON 对象');
      if (request.type === 'shutdown') {
        shuttingDown = true;
        checkShutdown();
        return;
      }
      if (!Number.isSafeInteger(request.id) || request.id < 0) throw new Error('请求 id 必须为非负整数');
      if (request.type === 'init') {
        initialize(request);
        return;
      }
      if (!configuration) throw new Error('分析宿主尚未初始化');
      if (request.type === 'cancel') {
        const job = jobs.get(request.id);
        if (job) abortJob(job, 'AbortError', '分析请求已取消');
        return;
      }
      if (request.type !== 'analyze' || request.id === 0 || shuttingDown) throw new Error('分析宿主不接受该请求');
      if (jobs.has(request.id)) throw new Error('分析请求 id 重复');
      if (jobs.size >= configuration.workers) throw new Error('分析请求超过有界队列容量');
      if (typeof request.source !== 'string' || !Number.isSafeInteger(request.source_id) || request.source_id <= 0) {
        throw new Error('分析输入路径或玩家 id 无效');
      }
      const job = {request: {id: request.id, type: 'analyze', source: request.source, source_id: request.source_id},
        slot: null, timer: null};
      jobs.set(request.id, job);
      job.timer = setTimeout(() => abortJob(job, 'TimeoutError', '分析请求超过截止时间'),
        configuration.timeout_seconds * 1000);
      queue.push(job);
      dispatch();
    } catch (error) {
      // 无法确定请求身份时终止协议，避免客户端无限等待或误匹配响应。
      if (!Number.isSafeInteger(request?.id) || request.id < 0) {
        console.error(error.stack || error);
        process.exit(1);
      }
      sendError(request.id, error);
    }
  }

  const input = readline.createInterface({input: process.stdin, crlfDelay: Infinity, terminal: false});
  input.on('line', receive);
  // Python 正常退出或被强制结束都会关闭管道；主线程无需等待分析 CPU 循环。
  input.on('close', () => process.exit(0));
  process.stdin.on('error', () => process.exit(1));
  process.stdout.on('error', () => process.exit(1));
}

module.exports = {startPool};
