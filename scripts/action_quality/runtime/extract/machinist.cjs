// 机工使用自己的窗口与量谱证据，不读取黑魔字段，也不凭窗口存在判定错误。
const {observations, toJson} = require('../serialize.cjs');

function extractMachinist(parser) {
  const evidence = observations(parser.container, {
      wildfire: ['history'], hypercharge: ['history'], reassemble: ['history'],
      queen: ['summons'], drift: ['driftedWindows'],
    });
  const gauge = parser.container.gauge;
  if (gauge) {
    evidence.gauge = {};
    for (const resource of ['heat', 'battery']) {
      const counter = gauge[resource];
      if (counter) evidence.gauge[resource] = {
        value: counter.value, overcap: counter.overCap, history: toJson(counter.history),
      };
    }
  }
  return {actionLabels: [], windowLabels: [], cycleLabels: [], cycles: [], observations: evidence};
}

module.exports = {extractMachinist};
