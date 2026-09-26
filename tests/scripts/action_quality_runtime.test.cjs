// 独立验证路由、事件匹配和序列化，不需要安装上游依赖。
const test = require('node:test');
const assert = require('node:assert/strict');
const {selectMeta} = require('../../scripts/action_quality/runtime/routing.cjs');
const {createReferences} = require('../../scripts/action_quality/runtime/references.cjs');
const {toJson} = require('../../scripts/action_quality/runtime/serialize.cjs');
const {extractMachinist} = require('../../scripts/action_quality/runtime/extract/machinist.cjs');

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
  const ref = createReferences(raw, {timestamp: 1000, actors: [{id: '2'}, {id: '31:2'}]}, resolve);
  assert.deepEqual(ref({timestamp: 1020, source: '2', target: '31:2', action: 100}).raw_event_indices, [0]);
  assert.equal(ref({timestamp: 1030, source: '2', target: 'unknown', action: 101}).match_status, 'unmatched');
  assert.equal(ref({timestamp: 1030, source: '2', target: 'unknown', action: 101}, 'begincast').match_status, 'exact');
  assert.equal(ref({timestamp: 1040, source: '2', target: 'unknown', action: 102}).match_status, 'ambiguous');
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
