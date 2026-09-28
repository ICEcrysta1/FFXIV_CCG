// 捕获上游实际规则判定；不使用循环最高优先级推测动作错误。
function captureAttribution(parser) {
  const bySuggestion = new WeakMap();
  const records = new Map();
  const remember = (key, events) => {
    if (!records.has(key)) records.set(key, new Map());
    for (const event of events) {
      const action = typeof event.action === 'object' ? event.action.id : event.action;
      records.get(key).set([event.timestamp, action, event.source, event.target].join('|'), event);
    }
  };
  const selectors = {
    ExtraF1Evaluator: (ev, window) => window.data.filter(e => e.action.id === ev.fire1Id),
    LukewarmF4Evaluator: (ev, window) => {
      const meta = ev.metadataHistory.entries.find(e => e.start === window.start).data.firePhaseMetadata;
      return window.data.filter(e => e.action.id === ev.fire4Action.id && e.timestamp >= meta.startTime && e.timestamp < meta.fullElementTime);
    },
    ColdF3Evaluator: (ev, window) => window.data.filter(e => e.action.id === ev.fire3Action.id &&
      ev.gauge.getGaugeState(e.timestamp - 1).umbralIce === 3 && ev.gauge.getGaugeState(e.timestamp - 1).umbralHearts === 3),
    ManafontTimingEvaluator: (ev, window) => [window.data.find(e => e.action.id === ev.manafontAction.id)],
    FirestarterUsageEvaluator: (ev, window) => window.data.slice(window.data.findIndex(e => e.action.id === ev.paradoxId))
      .filter(e => e.action.id === ev.fire3Id),
  };
  for (const ev of parser.container.RotationWatchdog?.evaluators ?? []) {
    const key = ev.constructor.name;
    if (!selectors[key] && key !== 'UptimeSoulsEvaluator') continue;
    const suggest = ev.suggest.bind(ev);
    ev.suggest = windows => {
      const original = ev.passesRule;
      if (selectors[key]) {
        ev.passesRule = function(window) {
          const result = original.call(this, window);
          if (result === false) remember(key, selectors[key](this, window));
          return result;
        };
      } else {
        remember(key, windows.flatMap(window => window.data.filter(e => e.action.id === ev.umbralSoulAction.id &&
          !ev.invulnerability.isActive({timestamp: e.timestamp, types: ['invulnerable']}))));
      }
      try {
        const suggestion = suggest(windows);
        if (suggestion) bySuggestion.set(suggestion, {reason: key, events: [...(records.get(key)?.values() ?? [])]});
        return suggestion;
      } finally {
        if (selectors[key]) ev.passesRule = original;
      }
    };
  }
  // 这些模块只保存计数，追加观察钩子记录本次处理后新增的错误。
  const aoe = parser.container.aoeusages;
  let aoeCount = 0;
  let triples = 0;
  aoe?.addEventHook('damage', event => {
    if (!aoe || event.source !== parser.actor.id) return;
    const total = [...aoe.badUsages.values()].reduce((a, b) => a + b, 0);
    if (total > aoeCount) remember('AoEUsages', [{...event, action: event.cause.action}]);
    aoeCount = total;
  });
  parser.container.triplecast?.addEventHook('action', event => {
    if (event.source !== parser.actor.id) return;
    const total = parser.container.triplecast?.overwrittenTriples ?? 0;
    if (total > triples) remember('Triplecast', [event]);
    triples = total;
  });
  return suggestion => {
    const direct = bySuggestion.get(suggestion);
    if (direct) return direct;
    const id = suggestion.content?.props?.id;
    const key = id === 'core.aoeusages.suggestion.content' ? 'AoEUsages' :
      id === 'blm.triplecast.suggestions.overwrote-triplecasts.content' ? 'Triplecast' : null;
    return key ? {reason: key, events: [...(records.get(key)?.values() ?? [])]} : null;
  };
}

function attachActions(labels, captured, attribution, reference) {
  labels.forEach((label, index) => {
    const evidence = attribution(captured[index].suggestion);
    label.actions = evidence ? evidence.events.map(event => {
      const ref = reference(event, event.type === 'damage' ? 'damage' : 'cast');
      return {...ref, event_type: ref.event_type ?? (event.type === 'damage' ? 'damage' : 'cast')};
    }) : [];
    label.action_attribution = evidence ? 'rule_events' : 'unavailable';
    label.reason = evidence?.reason ?? null;
    label.severity_basis = 'fight_aggregate';
  });
}

module.exports = {captureAttribution, attachActions};
