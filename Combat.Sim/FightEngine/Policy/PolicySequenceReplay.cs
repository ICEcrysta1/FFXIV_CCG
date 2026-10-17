// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using System.Text.Json;
using System.Text.Json.Serialization;
using Combat.Sim.Config;
using Combat.Sim.Facade;
using Combat.Sim.Models.Definitions;
using Combat.Sim.Models.Timeline;
using Combat.Sim.Outputs.TokenBuilders;

namespace Combat.Sim.Policy;

/// <summary>
/// 通用绝对时间策略回放：整批同步同刻事实，补充停止织入的等待，并返回逐请求增量观测。
/// JSON 只承担宿主数组传输，不读取日志格式、不生成训练标签、不合成场景 ETA。
/// </summary>
internal static class PolicySequenceReplay
{
    internal static Dictionary<string, object?> Run(
        JobSimulator simulator, PolicySession policy, string requestsJson, string factsJson,
        double fightDuration, string policyActionKey)
    {
        if (simulator.HistoryRetention.Limit is not null)
            throw new InvalidOperationException("policy sequence replay requires full history retention");
        ArgumentNullException.ThrowIfNull(requestsJson);
        ArgumentNullException.ThrowIfNull(factsJson);
        if (!double.IsFinite(fightDuration))
            throw new ArgumentOutOfRangeException(nameof(fightDuration));
        ArgumentException.ThrowIfNullOrWhiteSpace(policyActionKey);
        var requests = JsonSerializer.Deserialize<ReplayRequest[]>(requestsJson)
            ?? throw new ArgumentException("replay requests must be an array", nameof(requestsJson));
        if (requests.Length == 0)
            throw new ArgumentException("replay requests must not be empty", nameof(requestsJson));
        foreach (var request in requests)
        {
            if (request is null || !double.IsFinite(request.Timestamp) || string.IsNullOrWhiteSpace(request.SkillKey))
                throw new ArgumentException("replay requests require finite timestamps and skill keys", nameof(requestsJson));
            if (request.ActualCastSeconds is { } duration && (!double.IsFinite(duration) || duration < 0))
                throw new ArgumentOutOfRangeException(nameof(requestsJson), "actual cast duration must be finite and non-negative");
        }
        var suppliedFacts = JsonSerializer.Deserialize<ReplayFact[]>(factsJson)
            ?? throw new ArgumentException("replay facts must be an array", nameof(factsJson));
        var facts = new ExternalCombatEvent[suppliedFacts.Length];
        for (var index = 0; index < suppliedFacts.Length; index++)
        {
            var fact = suppliedFacts[index];
            if (fact is null || !double.IsFinite(fact.Timestamp) || fact.Timestamp < simulator.Time
                || (index > 0 && fact.Timestamp < facts[index - 1].Timestamp))
                throw new ArgumentException("replay facts must be ordered at or after the initial time", nameof(factsJson));
            facts[index] = new ExternalCombatEvent(
                fact.Timestamp, fact.EventKind, fact.Value, fact.TargetCount, fact.RemainingSeconds, fact.Mp);
        }

        var observations = new List<Dictionary<string, object?>>();
        var pendingMarkers = new List<Dictionary<string, object?>>();
        var factIndex = 0;
        var syncedTime = simulator.Time;
        var cursor = simulator.Time;
        long historyCursor = 0;
        var epsilon = SchemaConfigLoader.Instance.SceneEpsilon;
        for (var sourceIndex = 0; sourceIndex < requests.Length; sourceIndex++)
        {
            var request = requests[sourceIndex];
            if (cursor < request.Timestamp - epsilon)
            {
                var decisionTime = cursor;
                cursor = SyncAndAdvance(decisionTime);
                var gcdRemaining = simulator.GetStateWithoutHistory().GcdRemaining;
                if (gcdRemaining > epsilon)
                {
                    var windowEnd = decisionTime + gcdRemaining;
                    var nextSkill = simulator.Rules.ResolveSkill(request.SkillKey);
                    if (nextSkill.Kind != ActionKind.Ogcd || request.Timestamp >= windowEnd - epsilon)
                    {
                        var nextObservation = Math.Max(windowEnd, request.Timestamp);
                        var context = Observe(decisionTime, nextObservation);
                        policy.Record(simulator, decisionTime, policyActionKey, nextObservation);
                        AddFrame(sourceIndex, isWait: true, decisionTime, context, submission: null);
                    }
                }
            }

            SyncAndAdvance(request.Timestamp);
            var nextRequest = sourceIndex + 1 < requests.Length ? requests[sourceIndex + 1].Timestamp : fightDuration;
            var realContext = Observe(request.Timestamp, Math.Max(request.Timestamp, nextRequest));
            var submission = simulator.SubmitAction(request.Timestamp, request.SkillKey, request.ActualCastSeconds);
            AddFrame(sourceIndex, isWait: false, request.Timestamp, realContext, Submission(submission));
            // 拒绝帧仍返回给调用方，由其按自身标签、步号与错误契约处理；后续请求不再执行。
            if (!submission.Accepted) break;
            // 预测生效时刻只用于下一轮 wait 判断，不能提前推进并越过下一条日志请求。
            cursor = submission.EffectTimestamp ?? request.Timestamp;
        }
        return new Dictionary<string, object?> { ["observations"] = observations };

        double SyncAndAdvance(double timestamp)
        {
            if (timestamp < syncedTime)
                throw new InvalidOperationException("scene fact cursor cannot move backwards; reset the trajectory first");
            while (factIndex < facts.Length && facts[factIndex].Timestamp <= timestamp)
            {
                var batchEnd = factIndex + 1;
                while (batchEnd < facts.Length && facts[batchEnd].Timestamp == facts[factIndex].Timestamp) batchEnd++;
                var batch = facts[factIndex..batchEnd];
                var applied = simulator.ApplyExternalEvents(batch);
                if (!applied.Accepted)
                    throw new InvalidOperationException($"scene facts rejected at {batch[0].Timestamp:R}: {applied.Reason}");
                if (applied.Token is not null)
                {
                    var marker = DeathTokenBuilder.Build(applied.Timestamp, applied.HistoryCursor!.Value);
                    marker["step"] = observations.Count + 1;
                    pendingMarkers.Add(marker);
                }
                factIndex = batchEnd;
            }
            syncedTime = timestamp;
            simulator.AdvanceClockTo(timestamp);
            return simulator.Time;
        }

        Dictionary<string, object?> Observe(double timestamp, double nextObservation)
        {
            simulator.AdvanceClockTo(timestamp);
            var context = policy.BuildVectorDeltaContext(simulator, nextObservation, historyCursor);
            historyCursor = (long)context["history_cursor"]!;
            return context;
        }

        void AddFrame(int sourceIndex, bool isWait, double timestamp,
            Dictionary<string, object?> context, Dictionary<string, object?>? submission)
        {
            var frame = Frame(sourceIndex, isWait, timestamp, context, submission);
            if (pendingMarkers.Count > 0)
            {
                frame["history_markers"] = pendingMarkers.ToArray();
                pendingMarkers.Clear();
            }
            observations.Add(frame);
        }
    }

    private static Dictionary<string, object?> Frame(
        int sourceIndex, bool isWait, double timestamp, Dictionary<string, object?> context,
        Dictionary<string, object?>? submission) => new()
        {
            ["source_index"] = sourceIndex,
            ["is_wait"] = isWait,
            ["timestamp"] = timestamp,
            ["context"] = context,
            ["submission"] = submission,
        };

    private static Dictionary<string, object?> Submission(ActionSubmissionResult submission) => new()
    {
        ["accepted"] = submission.Accepted,
        ["queued"] = submission.Queued,
        ["reason"] = submission.Reason,
        ["accepted_timestamp"] = submission.AcceptedTimestamp,
        ["effect_timestamp"] = submission.EffectTimestamp,
    };

    private sealed record ReplayRequest
    {
        [JsonPropertyName("timestamp")] public required double Timestamp { get; init; }
        [JsonPropertyName("skill_key")] public required string SkillKey { get; init; }
        [JsonPropertyName("actual_cast_seconds")] public double? ActualCastSeconds { get; init; }
    }

    private sealed record ReplayFact
    {
        [JsonPropertyName("timestamp")] public required double Timestamp { get; init; }
        [JsonPropertyName("event_kind")] public required string EventKind { get; init; }
        [JsonPropertyName("value")] public bool? Value { get; init; }
        [JsonPropertyName("target_count")] public int? TargetCount { get; init; }
        [JsonPropertyName("remaining_seconds")] public double? RemainingSeconds { get; init; }
        [JsonPropertyName("mp")] public int? Mp { get; init; }
    }
}
