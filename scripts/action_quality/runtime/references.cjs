// 以类型、时间、技能和带实例号的双方 actor 精确匹配，禁止归因给最近动作。
const {createTimeline} = require('./time.cjs');

function createReferences(raw, pull, resolveActorId) {
  const timeline = createTimeline(pull.timestamp);
  const known = new Set(pull.actors.map(actor => actor.id));
  const index = new Map();
  const castPackets = new Map();
  const rawActor = (event, field) => {
    const id = resolveActorId({id: event[`${field}ID`], instance: event[`${field}Instance`], actor: event[field]});
    return known.has(id) ? id : 'unknown';
  };
  raw.events.forEach((event, position) => {
    const key = [event.type, event.timestamp, rawActor(event, 'source'), event.ability?.guid ?? event.abilityGameID,
      rawActor(event, 'target')].join('|');
    if (!index.has(key)) index.set(key, []);
    index.get(key).push(position);
    if (event.type === 'cast' && event.packetID != null) {
      const packetKey = [event.packetID, rawActor(event, 'source'), event.ability?.guid ?? event.abilityGameID].join('|');
      if (!castPackets.has(packetKey)) castPackets.set(packetKey, []);
      castPackets.get(packetKey).push(position);
    }
  });
  return (event, type = 'cast') => {
    const timestamp = event.timestamp - raw.start;
    const action = typeof event.action === 'object' ? event.action.id : event.action;
    const source = event.source ?? 'unknown';
    const target = event.target ?? 'unknown';
    // AoE 判定来自聚合伤害事件；只通过同一 packetID 关联施法，不猜最近动作。
    if (type === 'damage' && event.sequence != null) {
      const casts = castPackets.get([event.sequence, event.source, action].join('|')) ?? [];
      if (casts.length === 1) {
        const cast = raw.events[casts[0]];
        return {time_ms: timeline.at(raw.start + cast.timestamp), action_id: action,
          source_id: rawActor(cast, 'source'), target_id: rawActor(cast, 'target'),
          raw_event_indices: casts, match_status: 'exact', event_type: 'cast', association: 'packet_id'};
      }
    }
    const positions = index.get([type, timestamp, source, action, target].join('|')) ?? [];
    return {time_ms: timeline.at(event.timestamp), action_id: action,
      source_id: String(source), target_id: String(target), raw_event_indices: positions,
      match_status: positions.length === 1 ? 'exact' : positions.length ? 'ambiguous' : 'unmatched'};
  };
}

module.exports = {createReferences};
