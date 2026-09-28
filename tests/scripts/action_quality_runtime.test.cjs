// 独立验证路由、事件匹配和序列化，不需要安装上游依赖。
const test = require('node:test');
const assert = require('node:assert/strict');
const {selectMeta} = require('../../scripts/action_quality/runtime/routing.cjs');
const {createReferences} = require('../../scripts/action_quality/runtime/references.cjs');
const {toJson} = require('../../scripts/action_quality/runtime/serialize.cjs');
const {extractMachinist} = require('../../scripts/action_quality/runtime/extract/machinist.cjs');
const {createTimeline} = require('../../scripts/action_quality/runtime/time.cjs');
const {extractBlackMage} = require('../../scripts/action_quality/runtime/extract/black_mage.cjs');
const {captureAttribution, attachActions} = require('../../scripts/action_quality/runtime/attribution.cjs');
const {readCommit} = require('../../scripts/action_quality/runtime/analyze.cjs');

test('Git 审计版本不可用时仍可继续分析', () => {
  assert.equal(readCommit('/nonexistent/action-quality-analyzer'), null);
});

test('AoE 伤害仅以唯一 packetID 关联施法，使用施法时间，不猜附近技能', () => {
  const raw = {start: 1000, events: [
    {type: 'cast', timestamp: 20, packetID: 70, sourceID: 1, targetID: 2, ability: {guid: 162}},
    {type: 'cast', timestamp: 30, packetID: 71, sourceID: 1, targetID: 2, ability: {guid: 162}},
    {type: 'cast', timestamp: 31, packetID: 71, sourceID: 1, targetID: 2, ability: {guid: 162}},
  ]};
  const ref = createReferences(raw, {timestamp: 1010, actors: [{id: '1'}, {id: '2'}]}, ({id}) => String(id));
  const event = {timestamp: 1080, sequence: 70, source: '1', action: 162};
  assert.deepEqual(ref(event, 'damage').raw_event_indices, [0]);
  assert.equal(ref(event, 'damage').time_ms, 10);
  assert.equal(ref(event, 'damage').event_type, 'cast');
  assert.equal(ref({...event, sequence: 71}, 'damage').match_status, 'unmatched');
  assert.equal(ref({...event, sequence: 72}, 'damage').match_status, 'unmatched');
  assert.equal(ref({...event, sequence: 72}, 'damage').target_id, 'unknown');
});

test('错误类别保留全部命中动作并沿用整场等级，不被循环优先级覆盖', () => {
  class ExtraF1Evaluator {
    constructor() {this.fire1Id = 141;}
    passesRule(window) {return !window.data.some(e => e.action.id === this.fire1Id);}
    suggest(windows) {
      const count = windows.filter(w => this.passesRule(w) === false).length;
      return {severity: count >= 2 ? 'major' : 'medium'};
    }
  }
  const ev = new ExtraF1Evaluator();
  const original = ev.passesRule;
  const parser = {actor: {id: '1'}, container: {RotationWatchdog: {evaluators: [ev]}}};
  const attribution = captureAttribution(parser);
  const event = timestamp => ({timestamp, action: {id: 141}, source: '1', target: '2'});
  const windows = [{data: [event(1010), event(1020), {...event(1030), action: {id: 152}}]}, {data: [event(1040)]}];
  const suggestion = ev.suggest(windows);
  // 重复生成建议不重复累积动作，也不改变上游判定方法。
  ev.suggest(windows);
  assert.equal(ev.passesRule, original);
  const labels = [{severity: suggestion.severity}, {severity: 'minor'}];
  attachActions(labels, [{suggestion}, {suggestion: {}}], attribution,
    e => ({time_ms: e.timestamp - 1000, action_id: e.action.id, match_status: 'exact'}));
  assert.equal(labels[0].severity, 'major');
  assert.deepEqual(labels[0].actions.map(a => a.time_ms), [10, 20, 40]);
  assert.equal(labels[0].severity_basis, 'fight_aggregate');
  assert.equal(labels[1].action_attribution, 'unavailable');
  assert.deepEqual(labels[1].actions, []);
});

test('追加观察保留 AoE 伤害证据和触发覆盖的 Triplecast，不重算等级', () => {
  const hooks = {};
  const module = key => ({addEventHook(type, callback) {hooks[key] = callback;}});
  const aoe = {...module('aoe'), badUsages: new Map()};
  const triplecast = {...module('triplecast'), overwrittenTriples: 0};
  const attribution = captureAttribution({actor: {id: '1'}, container: {aoeusages: aoe, triplecast}});
  aoe.badUsages.set(100, 1);
  hooks.aoe({type: 'damage', timestamp: 1500, source: '1', cause: {action: 100}});
  hooks.aoe({type: 'damage', timestamp: 1600, source: '1', cause: {action: 100}});
  triplecast.overwrittenTriples = 1;
  hooks.triplecast({type: 'action', timestamp: 2000, source: '1', action: 200});
  hooks.triplecast({type: 'action', timestamp: 2100, source: '1', action: 201});
  const captured = ['core.aoeusages.suggestion.content', 'blm.triplecast.suggestions.overwrote-triplecasts.content']
    .map(id => ({suggestion: {content: {props: {id}}}}));
  const labels = [{severity: 'major'}, {severity: 'minor'}];
  attachActions(labels, captured, attribution, (event, type) => ({time_ms: event.timestamp - 1000, action_id: event.action, type}));
  assert.equal(labels[0].actions.length, 1);
  assert.equal(labels[0].actions[0].event_type, 'damage');
  assert.equal(labels[1].actions.length, 1);
  assert.equal(labels[1].actions[0].action_id, 200);
});

test('根据输入选择黑魔或机工；未知副本仍使用通用加职业模块', () => {
  const build = names => ({names, merge(other) {return build([...names, ...other.names]);}});
  const registry = {CORE: build(['core']), JOBS: {BLACK_MAGE: build(['blm']), MACHINIST: build(['mch'])}, BOSSES: {FRU: build(['fru'])}};
  assert.deepEqual(selectMeta(registry, {encounter: {key: 'FRU'}}, {job: 'BLACK_MAGE'}).meta.names, ['core', 'fru', 'blm']);
  assert.deepEqual(selectMeta(registry, {encounter: {key: 'FRU'}}, {job: 'MACHINIST'}).meta.names, ['core', 'fru', 'mch']);
  assert.deepEqual(selectMeta(registry, {encounter: {}}, {job: 'MACHINIST'}).meta.names, ['core', 'mch']);
  assert.throws(() => selectMeta(registry, {encounter: {}}, {job: 'UNKNOWN'}));
});

test('匹配保留实例、未知目标、重复事件和失败施法类型', () => {
  const resolve = ({id = -1, instance = 1}) => (id === -1 ? 'unknown' : String(id)) + (instance > 1 ? `:${instance}` : '');
  const raw = {start: 1000, events: [
    {type: 'cast', timestamp: 20, sourceID: 2, targetID: 31, targetInstance: 2, ability: {guid: 100}},
    {type: 'begincast', timestamp: 30, sourceID: 2, ability: {guid: 101}},
    {type: 'cast', timestamp: 40, sourceID: 2, ability: {guid: 102}},
    {type: 'cast', timestamp: 40, sourceID: 2, ability: {guid: 102}},
  ]};
  const ref = createReferences(raw, {timestamp: 1010, actors: [{id: '2'}, {id: '31:2'}]}, resolve);
  assert.deepEqual(ref({timestamp: 1020, source: '2', target: '31:2', action: 100}).raw_event_indices, [0]);
  const first = ref({timestamp: 1020, source: '2', target: '31:2', action: 100});
  assert.equal(first.time_ms, 10);
  assert.equal(first.timestamp, undefined);
  assert.equal(first.relative_time_ms, undefined);
  assert.equal(ref({timestamp: 1030, source: '2', target: 'unknown', action: 101}).match_status, 'unmatched');
  assert.equal(ref({timestamp: 1030, source: '2', target: 'unknown', action: 101}, 'begincast').match_status, 'exact');
  assert.equal(ref({timestamp: 1040, source: '2', target: 'unknown', action: 102}).match_status, 'ambiguous');
});

test('统一时间原点，保留预读负值，持续时间和资源数值不平移', () => {
  const timeline = createTimeline(1000000);
  assert.equal(timeline.at(999000), -1000);
  const result = timeline.evidence({timestamp: 1060000, duration: 30000, clip: 6000,
    start: 999000, stop: 1060000, fullElementTime: 0,
    applicationTimestamps: [1000100, 1000200], droppedEnoTimestamps: [1000300],
    nested: new Map([['counter', {lastApplied: 1000400, value: 3}]])});
  assert.deepEqual(result, {time_ms: 60000, duration: 30000, clip: 6000,
    start_ms: -1000, end_ms: 60000, full_element_ms: null,
    application_times_ms: [100, 200], dropped_eno_times_ms: [300],
    nested: [['counter', {last_applied_ms: 400, value: 3}]]});
  assert.throws(() => createTimeline(NaN));
  assert.throws(() => timeline.evidence({end: 1000000, stop: 1000000}), /重复/);
});

test('黑魔资源错误、循环和循环元数据共用战斗原点，不使用报告原点', () => {
  const Module = require('module');
  const original = Module._load;
  const errorCode = {priority: 0};
  // 独立测试只替代上游常量导入，提取器本身真实运行。
  Module._load = function(name, ...args) {
    if (name.replaceAll('\\', '/').endsWith('/RotationWatchdog/WatchdogConstants')) return {ROTATION_ERRORS: {NO_ERROR: errorCode}};
    return original.call(this, name, ...args);
  };
  let result;
  try {
    result = extractBlackMage({pull: {timestamp: 1200000}, report: {timestamp: 1000000}, container: {
      RotationWatchdog: {history: {entries: [{start: 1200000, end: 1300000, data: []}]},
        metadataHistory: {entries: [{data: {errorCode, firePhaseMetadata: {startTime: 1201000, fullElementTime: 0}}}]}},
      gauge: {gaugeErrors: [{timestamp: 1260000, error: 'overcap'}]},
    }}, () => {}, '/fixture');
  } finally {Module._load = original;}
  assert.equal(result.windowLabels[0].time_ms, 60000);
  assert.equal(result.windowLabels[0].timestamp, undefined);
  assert.equal(result.cycles[0].start_ms, 0);
  assert.equal(result.cycles[0].end_ms, 100000);
  assert.equal(result.cycles[0].metadata.firePhaseMetadata.start_ms, 1000);
  assert.equal(result.cycles[0].metadata.firePhaseMetadata.full_element_ms, null);
});

test('消息参数、Map 和特殊严重程度保留为可序列化数据', () => {
  const value = {$$typeof: Symbol.for('react.element'), type: function Trans() {}, props: {id: 'reason', values: {count: 3}}};
  assert.equal(toJson(value).props.values.count, 3);
  assert.deepEqual(toJson(new Map([['a', 1]])), [['a', 1]]);
  assert.equal(toJson(Infinity), null);
  const circular = {}; circular.self = circular;
  assert.throws(() => toJson(circular), /循环引用/);
});

test('机工提取器不依赖黑魔循环或量谱字段，也不把普通窗口当错误', () => {
  const output = extractMachinist({container: {wildfire: {history: {entries: [{data: {stacks: 6}}]}}, reassemble: {history: {badUses: 1}}}});
  assert.equal(output.observations.wildfire.history.entries[0].data.stacks, 6);
  assert.equal(output.observations.reassemble.history.badUses, 1);
  assert.deepEqual(output.cycleLabels, []);
  const parser = {container: {gauge: {heat: {value: 10, overCap: 5, history: [{value: 10}]}}}};
  parser.container.gauge.heat._parser = parser;
  const evidence = extractMachinist(parser).observations.gauge.heat;
  assert.equal(evidence.value, 10);
  assert.equal(evidence.overcap, 5);
  assert.equal(evidence._parser, undefined);
});
