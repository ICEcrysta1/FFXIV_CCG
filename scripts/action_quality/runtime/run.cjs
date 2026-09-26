// Node 子进程只写系统临时分析结果，正式合并和原子保存由 Python 管理。
const fs = require('fs');
const {bootstrap} = require('./bootstrap.cjs');

(async () => {
  const request = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  bootstrap(request.analyzer_root, request.node_modules);
  const {analyze} = require('./analyze.cjs');
  const analysis = await analyze(request);
  fs.writeFileSync(process.argv[3], JSON.stringify(analysis) + '\n', 'utf8');
  process.exit(0);
})().catch(error => {
  console.error(error.stack);
  process.exit(1);
});
