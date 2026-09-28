// 保存消息标识、参数和组件描述；不执行网页组件，不保存函数或解析器实例。
function toJson(value, ancestors = new WeakSet()) {
  if (value == null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value === 'function' || typeof value === 'symbol') return undefined;
  if (ancestors.has(value)) throw new Error('分析输出包含循环引用');
  ancestors.add(value);
  try {
    if (Array.isArray(value)) return value.map(item => toJson(item, ancestors));
    if (value instanceof Map) return [...value].map(([key, item]) => [toJson(key, ancestors), toJson(item, ancestors)]);
    if (value instanceof Set) return [...value].map(item => toJson(item, ancestors));
    if (value.$$typeof) {
      return {component: typeof value.type === 'string' ? value.type : value.type?.displayName ?? value.type?.name ?? 'Fragment',
        props: toJson(value.props, ancestors)};
    }
    const output = {};
    for (const [key, item] of Object.entries(value)) {
      if (key === 'parser' || key === '_owner') continue;
      const serialized = toJson(item, ancestors);
      if (serialized !== undefined) output[key] = serialized;
    }
    return output;
  } finally {
    ancestors.delete(value);
  }
}

function observations(container, fields) {
  const output = {};
  for (const [handle, keys] of Object.entries(fields)) {
    const module = container[handle];
    if (!module) continue;
    output[handle] = Object.fromEntries(keys.filter(key => module[key] !== undefined)
      .map(key => [key, toJson(module[key])]));
  }
  return output;
}

module.exports = {toJson, observations};
