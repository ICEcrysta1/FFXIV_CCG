// 职业和副本均来自适配后的报告；没有职业模块时拒绝降级成仅通用规则。
function selectMeta(registry, pull, actor) {
  const jobMeta = registry.JOBS[actor.job];
  if (!jobMeta) throw new Error(`没有职业分析模块: ${actor.job}`);
  const encounterMeta = registry.BOSSES[pull.encounter.key];
  let meta = registry.CORE;
  if (encounterMeta) meta = meta.merge(encounterMeta);
  return {meta: meta.merge(jobMeta), encounterMeta};
}

module.exports = {selectMeta};
