// 通用规则输出；问题等级留在整场建议中，避免把聚合次数等级套到每个动作。
const {observations, toJson} = require('../serialize.cjs');

function extractCommon(parser, reference) {
  const container = parser.container;
  const actionLabels = (container.interrupts?.droppedCasts ?? []).map((entry, index) => ({
    id: `interrupt-${index}`, scope: 'cast_attempt', reason: 'interrupted_cast', severity: null,
    action: reference(entry.leadingEvent, 'begincast'), missed_time_ms: entry.missedTimeMS,
  }));
  const windowLabels = (container.weaving?.badWeaves ?? []).map((entry, index) => ({
    id: `weaving-${index}`, scope: 'weave_window', reason: 'incorrect_weaving', severity: null,
    leading_gcd: reference(entry.leadingGcdEvent), trailing_gcd: reference(entry.trailingGcdEvent),
    actions: entry.weaves.map(event => reference(event)), gcd_time_diff_ms: entry.gcdTimeDiff,
  }));
  // 连击问题的 breaker 往往是伤害结算，不冒充同时间戳的成功施法。
  for (const [index, issue] of (container.combos?.issues ?? []).entries()) {
    windowLabels.push({id: `combo-${index}`, scope: 'combo_window', reason: `combo_${issue.type}`,
      severity: null, evidence: toJson(issue)});
  }
  return {actionLabels, windowLabels, cycleLabels: [], cycles: [], observations: observations(container, {
    interrupts: ['droppedCasts', 'missedTimeMS', 'severity'], weaving: ['badWeaves', 'severity'],
    combos: ['issues'], aoeusages: ['badUsages', 'severity'], death: ['info'],
  })};
}

module.exports = {extractCommon};
