// Node 子进程通过标准输入和标准输出交换分析数据，正式合并和原子保存由 Python 管理。
const fs = require('fs');
const {bootstrap} = require('./bootstrap.cjs');

(async () => {
  const request = JSON.parse(fs.readFileSync(0, 'utf8'));
  bootstrap(request.analyzer_root, request.node_modules);
  const {analyze} = require('./analyze.cjs');
  const analysis = await analyze(request);
  process.stdout.write(JSON.stringify(analysis) + '\n', () => process.exit(0));
})().catch(error => {
  console.error(error.stack);
  process.exit(1);
});
