// 使用上游职业/副本注册表动态路由；只导出结构化结果，不抓取网页。
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const {execFileSync} = require('child_process');
const {toJson} = require('./serialize.cjs');
const {createReferences} = require('./references.cjs');
const {extractCommon} = require('./extract/common.cjs');
const {selectMeta} = require('./routing.cjs');

async function analyze(request) {
  const load = name => require(path.join(request.analyzer_root, 'src', name));
  const {adaptReport} = load('reportSources/legacyFflogs/reportAdapter');
  const {adaptEvents} = load('reportSources/legacyFflogs/eventAdapter');
  const {resolveActorId} = load('reportSources/legacyFflogs/base');
  const {Parser} = load('parser/core/Parser');
  const {AVAILABLE_MODULES} = load('parser/AVAILABLE_MODULES');
  const {patchSupported, contentSupported, getPatch} = load('data/PATCHES');
  const {SEVERITY} = load('parser/core/modules/Suggestions');
  const content = fs.readFileSync(request.source);
  const raw = JSON.parse(content.toString('utf8'));
  const report = adaptReport(raw);
  const pull = report.pulls.find(candidate => candidate.id === String(raw.fight_id));
  if (!pull) throw new Error('选定 fight_id 不存在');
  const actor = pull.actors.find(candidate => candidate.id === String(request.source_id));
  if (!actor?.playerControlled) throw new Error('选定 source_id 不是受支持的玩家');
  const {meta, encounterMeta} = selectMeta(AVAILABLE_MODULES, pull, actor);
  const supported = meta.supportedPatches;
  if (!supported || !patchSupported(report.edition, supported.from, supported.to, pull.timestamp / 1000)) {
    throw new Error('该日志版本不在上游分析模块支持范围内');
  }
  const info = raw.events.find(event => event.type === 'combatantinfo' && event.sourceID === request.source_id);
  if (!contentSupported(info?.level, actor.job)) throw new Error('该日志角色等级不受上游支持');
  const fight = raw.fights.find(candidate => candidate.id === raw.fight_id);
  const events = adaptEvents(report, pull, structuredClone(raw.events), raw.start + fight.start_time);
  const parser = new Parser({report, pull, actor, meta});
  await parser.configure();
  const captured = [];
  const originalAdd = parser.container.suggestions.add.bind(parser.container.suggestions);
  parser.container.suggestions.add = suggestion => {
    const frames = new Error().stack.split('\n').map(frame => frame.replaceAll('\\', '/'));
    const locations = frames.filter(frame => frame.includes('/src/parser/')).slice(0, 5)
      .map(frame => frame.slice(frame.indexOf('/src/parser/') + 1));
    captured.push({suggestion, locations});
    originalAdd(suggestion);
  };
  parser.parseEvents({events});
  parser.generateResults();
  if (Object.keys(parser._moduleErrors).length) throw new Error('分析模块发生错误，拒绝保存不完整标签');
  const severityNames = {[SEVERITY.MAJOR]: 'major', [SEVERITY.MEDIUM]: 'medium', [SEVERITY.MINOR]: 'minor'};
  const fightLabels = captured.map(({suggestion, locations}, index) => ({
    id: `suggestion-${index}`, scope: 'fight', severity_code: Number.isFinite(suggestion.severity) ? suggestion.severity : null,
    severity: severityNames[suggestion.severity] ?? null,
    severity_kind: severityNames[suggestion.severity] ? 'standard' : suggestion.severity === SEVERITY.MORBID ? 'death' : 'ignored_or_special',
    visible: Number.isFinite(suggestion.severity) && suggestion.severity < SEVERITY.MEMES,
    content: toJson(suggestion.content), why: toJson(suggestion.why),
    value: suggestion.value ?? null, tiers: toJson(suggestion.tiers), origin_locations: locations,
  }));
  const reference = createReferences(raw, pull, resolveActorId);
  const common = extractCommon(parser, reference);
  const extractors = {
    BLACK_MAGE: () => require('./extract/black_mage.cjs').extractBlackMage(parser, reference, request.analyzer_root),
    MACHINIST: () => require('./extract/machinist.cjs').extractMachinist(parser),
  };
  const specific = extractors[actor.job]?.() ?? {actionLabels: [], windowLabels: [], cycleLabels: [], cycles: [], observations: {}};
  const commit = execFileSync('git', ['-C', request.analyzer_root, 'rev-parse', 'HEAD'], {encoding: 'utf8'}).trim();
  return {
    schema_version: 1, bridge_version: 1, status: 'annotated', training_ready: false,
    source: {sha256: crypto.createHash('sha256').update(content).digest('hex'), report_code: raw.report_code ?? raw.code, fight_id: raw.fight_id},
    engine: {commit, runtime: process.version, encounter_module: encounterMeta ? pull.encounter.key : null,
      job_module: actor.job, modules: Object.keys(parser.container)},
    actor: {id: actor.id, name: actor.name, job: actor.job},
    pull: {id: pull.id, timestamp: pull.timestamp, duration: pull.duration,
      patch: getPatch(report.edition, pull.timestamp / 1000)},
    module_errors: Object.keys(parser._moduleErrors),
    counts: {raw_events: raw.events.length, adapted_events: events.length,
      visible_suggestions: fightLabels.filter(label => label.visible).length},
    fight_labels: fightLabels, action_labels: [...common.actionLabels, ...specific.actionLabels],
    window_labels: [...common.windowLabels, ...specific.windowLabels], cycle_labels: specific.cycleLabels,
    cycles: specific.cycles, observations: {...common.observations, ...specific.observations},
    coverage: {common: true, job_evidence: Boolean(extractors[actor.job]), action_attribution: 'partial'},
    limitations: ['整场聚合严重程度未自动分配到动作或循环。',
      '逐规则动作归因与训练权重接入尚未完成；没有标签不代表动作正确。', '未计算 PPG，原始 ranking 保持不变。'],
  };
}

module.exports = {analyze};
