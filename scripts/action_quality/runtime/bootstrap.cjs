// 仅补齐无界面运行需要的模块加载、DOM 和样式，不替换上游分析规则。
const fs = require('fs');
const path = require('path');
const Module = require('module');

function bootstrap(analyzerRoot, nodeModules) {
  if (!fs.existsSync(path.join(analyzerRoot, 'src/parser/AVAILABLE_MODULES.ts'))) {
    throw new Error('分析器源码缺失，请先运行 git submodule update --init');
  }
  const paths = [nodeModules, path.join(nodeModules, '.pnpm/node_modules'), path.join(__dirname, 'node_modules')];
  const resolve = name => require.resolve(name, {paths});
  // Lingui 从工作目录发现上游配置；只影响独立子进程。
  process.chdir(analyzerRoot);
  process.env.NODE_ENV = 'development';
  process.env.NODE_PATH = [path.join(analyzerRoot, 'src'), ...paths].join(path.delimiter);
  Module._initPaths();
  Symbol.metadata ??= Symbol.for('Symbol.metadata');
  let JSDOM;
  try {
    ({JSDOM} = require(resolve('jsdom')));
  } catch (error) {
    throw new Error('缺少桥接 DOM 依赖，请在 scripts/action_quality/runtime 运行 npm ci', {cause: error});
  }
  const dom = new JSDOM('<!doctype html><html><body></body></html>', {url: 'http://localhost/'});
  global.window = dom.window;
  global.document = dom.window.document;
  global.localStorage = dom.window.localStorage;
  Object.defineProperty(global, 'navigator', {value: dom.window.navigator, configurable: true});
  global.requestAnimationFrame = callback => setTimeout(callback, 0);
  global.cancelAnimationFrame = clearTimeout;
  window.requestAnimationFrame = global.requestAnimationFrame;
  window.cancelAnimationFrame = clearTimeout;
  for (const key of Object.getOwnPropertyNames(window)) {
    if (!(key in global)) Object.defineProperty(global, key, Object.getOwnPropertyDescriptor(window, key));
  }
  require.extensions['.css'] = (module, filename) => {
    const values = Object.fromEntries([...fs.readFileSync(filename, 'utf8').matchAll(/@value\s+(\w+):\s*([^;]+);/g)]
      .map(match => [match[1], match[2]]));
    module.exports = new Proxy(values, {get: (value, key) => key === '__esModule' ? false : value[key] ?? String(key)});
  };
  for (const extension of ['.png', '.jpg', '.jpeg', '.svg', '.gif', '.webp']) {
    require.extensions[extension] = (module, filename) => { module.exports = filename; };
  }
  const config = require(path.join(analyzerRoot, 'babel.config.js'))({caller: () => true, env: () => true});
  const resolvePlugin = plugin => {
    const [name, options] = Array.isArray(plugin) ? plugin : [plugin, undefined];
    const filename = name.startsWith('.') ? path.resolve(analyzerRoot, name) : resolve(name);
    return name === 'babel-plugin-macros'
      ? [filename, {resolvePath: (specifier, base) => require.resolve(specifier, {paths: [base, ...paths]})}]
      : options ? [filename, options] : filename;
  };
  config.presets = config.presets.map(resolvePlugin);
  for (const override of config.overrides) override.plugins = override.plugins.map(resolvePlugin);
  require(resolve('@babel/register'))({
    ...config, extensions: ['.js', '.jsx', '.ts', '.tsx'], configFile: false, babelrc: false, cache: false,
    ignore: [filename => filename.includes('node_modules')
      && !/[/\\](ky|d3-[^/\\]+|internmap|@compiled|@babel[/\\]runtime-corejs3)[/\\]/.test(filename)],
  });
}

module.exports = {bootstrap};
