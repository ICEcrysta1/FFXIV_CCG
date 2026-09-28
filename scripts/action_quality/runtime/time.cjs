// 标注层唯一时间坐标：相对分析器本场战斗起点的毫秒；开怪前允许负值。
const {toJson} = require('./serialize.cjs');

// 只转换已确认的上游时刻字段，持续时间、间隔和资源数值不变。
const POINT_FIELDS = {
  timestamp: 'time_ms', start: 'start_ms', end: 'end_ms', stop: 'end_ms',
  startTime: 'start_ms', endTime: 'end_ms', fullElementTime: 'full_element_ms',
  lastApplied: 'last_applied_ms',
};
const POINT_ARRAY_FIELDS = {
  applicationTimestamps: 'application_times_ms', droppedEnoTimestamps: 'dropped_eno_times_ms',
};
const ZERO_IS_UNSET = new Set(['startTime', 'endTime', 'fullElementTime', 'lastApplied', 'stop']);

function createTimeline(origin) {
  if (!Number.isFinite(origin)) throw new Error('战斗时间原点必须是有限数值');
  const at = timestamp => {
    if (timestamp == null || !Number.isFinite(timestamp)) return null;
    return timestamp - origin;
  };
  function normalize(value) {
    if (Array.isArray(value)) return value.map(normalize);
    if (value == null || typeof value !== 'object') return value;
    const output = {};
    for (const [key, item] of Object.entries(value)) {
      const point = POINT_FIELDS[key];
      const array = POINT_ARRAY_FIELDS[key];
      const target = point ?? array ?? key;
      if (Object.hasOwn(output, target)) throw new Error(`重复的时间字段: ${target}`);
      output[target] = point
        ? ZERO_IS_UNSET.has(key) && item === 0 ? null : at(item)
        : array ? item.map(at) : normalize(item);
    }
    return output;
  }
  return {at, evidence: value => normalize(toJson(value))};
}

module.exports = {createTimeline};
