// 以类型、时间、技能和带实例号的双方 actor 精确匹配，禁止归因给最近动作。
function createReferences(raw, pull, resolveActorId) {
  const known = new Set(pull.actors.map(actor => actor.id));
  const index = new Map();
  const rawActor = (event, field) => {
    const id = resolveActorId({id: event[`${field}ID`], instance: event[`${field}Instance`], actor: event[field]});
    return known.has(id) ? id : 'unknown';
  };
  raw.events.forEach((event, position) => {
    const key = [event.type, event.timestamp, rawActor(event, 'source'), event.ability?.guid ?? event.abilityGameID,
      rawActor(event, 'target')].join('|');
    if (!index.has(key)) index.set(key, []);
    index.get(key).push(position);
  });
  return (event, type = 'cast') => {
    const timestamp = event.timestamp - raw.start;
    const action = typeof event.action === 'object' ? event.action.id : event.action;
    const positions = index.get([type, timestamp, event.source, action, event.target].join('|')) ?? [];
    return {timestamp, relative_time_ms: event.timestamp - pull.timestamp, action_id: action,
      source_id: String(event.source), target_id: String(event.target), raw_event_indices: positions,
      match_status: positions.length === 1 ? 'exact' : positions.length ? 'ambiguous' : 'unmatched'};
  };
}

module.exports = {createReferences};
