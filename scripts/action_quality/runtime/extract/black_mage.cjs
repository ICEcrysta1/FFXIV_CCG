// 黑魔循环和职业证据提取；errorCode.priority 仅表示上游展示优先级。
const path = require('path');
const {observations, toJson} = require('../serialize.cjs');

function extractBlackMage(parser, reference, analyzerRoot) {
  const watchdog = parser.container.RotationWatchdog;
  const {ROTATION_ERRORS} = require(path.join(analyzerRoot, 'src/parser/jobs/blm/modules/RotationWatchdog/WatchdogConstants'));
  const cycles = watchdog.history.entries.map((window, index) => {
    const metadata = watchdog.metadataHistory.entries[index]?.data;
    return {id: `cycle-${index}`, start_ms: window.start - parser.pull.timestamp,
      end_ms: window.end - parser.pull.timestamp,
      error_code: Object.entries(ROTATION_ERRORS).find(([, code]) => code === metadata?.errorCode)?.[0] ?? null,
      metadata: toJson(metadata), actions: window.data.map(event => reference(event))};
  });
  const cycleLabels = cycles.filter(cycle => cycle.metadata?.errorCode.priority > 2).map(cycle => ({
    scope: 'cycle', cycle_id: cycle.id, reason: cycle.error_code, severity: null,
  }));
  const resourceEvents = (parser.container.gauge.gaugeErrors ?? []).map((event, index) => ({
    id: `gauge-${index}`, scope: 'resource_event', reason: 'gauge_error', severity: null,
    timestamp: event.timestamp - parser.report.timestamp, error_code: event.error,
  }));
  return {actionLabels: [], windowLabels: resourceEvents, cycleLabels, cycles,
    observations: observations(parser.container, {
      gauge: ['gaugeErrors', 'droppedEnoTimestamps'],
      procs: ['history', 'droppedProcs', 'overwrittenProcs', 'invulnUsages'],
      thunder: ['thunderCasts', 'totalThunderClip', 'clip'], dots: ['statusApplications'],
      swiftcast: ['history'], triplecast: ['history', 'overwrittenTriples'],
    })};
}

module.exports = {extractBlackMage};
