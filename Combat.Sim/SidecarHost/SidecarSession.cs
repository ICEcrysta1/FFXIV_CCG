using System.Diagnostics;
using System.Text.Json;
using Combat.Sim.Config;
using Combat.Sim.Facade;
using Combat.Sim.Models.Timeline;
using Combat.Sim.Policy;

namespace Combat.Sim.SidecarHost;

/// <summary>
/// Sidecar 的单会话协议处理器。动作命令只返回时序结果；完整模型维度只由 observe_at 返回。
/// </summary>
public sealed class SidecarSession
{
    private readonly string _projectRoot;
    private JobSimulator? _simulator;
    private PolicySession? _policy;

    public SidecarSession(string projectRoot)
    {
        _projectRoot = projectRoot;
    }

    public string? Handle(string line)
    {
        int? seq = null;
        try
        {
            using var document = JsonDocument.Parse(line);
            var root = document.RootElement;
            seq = root.GetProperty("seq").GetInt32();
            var op = root.GetProperty("op").GetString();
            return op switch
            {
                "init" => Reply(seq.Value, HandleInit(root)),
                "submit_action" => Reply(seq.Value, HandleSubmitAction(root)),
                "validate_at" => Reply(seq.Value, HandleValidateAt(root)),
                "advance_to" => Reply(seq.Value, HandleAdvanceTo(root)),
                "record_policy_action" => Reply(seq.Value, HandleRecordPolicyAction(root)),
                "apply_external_event" => Reply(seq.Value, HandleApplyExternalEvent(root)),
                "observe_at" => Reply(seq.Value, HandleObserveAt(root)),
                "close" => null,
                _ => throw new InvalidOperationException($"unknown op: {op}"),
            };
        }
        catch (Exception exception)
        {
            return JsonSerializer.Serialize(new Dictionary<string, object?>
            {
                ["seq"] = seq,
                ["ok"] = false,
                ["error"] = $"{exception.GetType().Name}: {exception.Message}",
            });
        }
    }

    private Dictionary<string, object?> HandleInit(JsonElement root)
    {
        var jobTag = root.GetProperty("job_tag").GetString()!;
        var actualBaseGcd = OptionalDouble(root, "actual_base_gcd");
        var fightRemaining = OptionalDouble(root, "fight_remaining");
        var maxHistory = root.TryGetProperty("max_history", out var historyElement)
            ? historyElement.GetInt32()
            : (int?)null;
        var initialTimestamp = OptionalDouble(root, "initial_timestamp") ?? 0.0;

        _simulator = JobSimulator.Create(
            _projectRoot,
            jobTag,
            actualBaseGcd,
            maxHistory,
            initialTimestamp,
            fightRemaining);
        _policy = PolicySession.Create(_projectRoot, _simulator);
        return new Dictionary<string, object?>
        {
            ["timestamp"] = _simulator.Time,
            ["sidecar_contract_version"] = SchemaConfigLoader.Instance.SidecarContractVersion,
        };
    }

    private Dictionary<string, object?> HandleSubmitAction(JsonElement root)
    {
        var simulator = RequireSimulator();
        var result = simulator.SubmitAction(
            root.GetProperty("timestamp").GetDouble(),
            root.GetProperty("action").GetString()!,
            OptionalDouble(root, "actual_cast_seconds"));
        return new Dictionary<string, object?>
        {
            ["accepted"] = result.Accepted,
            ["queued"] = result.Queued,
            ["reason"] = result.Reason,
            ["action_instance_id"] = result.ActionInstanceId?.ToString("D"),
            ["request_timestamp"] = result.RequestTimestamp,
            ["accepted_timestamp"] = result.AcceptedTimestamp,
            ["effect_timestamp"] = result.EffectTimestamp,
            ["next_scheduled_event_time"] = result.NextScheduledEventTime,
        };
    }

    /// <summary>
    /// 只读探测某个动作在指定时刻是否合法。它会把时钟推进到该时刻（不接受过去时刻），
    /// 并在先前请求已被排队时返回 action_queue_occupied；仅用于调用方的合法性断言，
    /// 不作为提交前置——校验与接受必须在 SubmitAction 内原子完成。
    /// </summary>
    private Dictionary<string, object?> HandleValidateAt(JsonElement root)
    {
        var simulator = RequireSimulator();
        var timestamp = root.GetProperty("timestamp").GetDouble();
        var validation = simulator.ValidateActionAt(
            timestamp,
            root.GetProperty("action").GetString()!);
        return new Dictionary<string, object?>
        {
            ["legal"] = validation.Ok,
            ["reason"] = validation.Reason,
            ["timestamp"] = simulator.Time,
            ["next_scheduled_event_time"] = simulator.GetNextScheduledEventTime(),
        };
    }

    private Dictionary<string, object?> HandleAdvanceTo(JsonElement root)
    {
        var simulator = RequireSimulator();
        var state = simulator.AdvanceTo(root.GetProperty("timestamp").GetDouble());
        return new Dictionary<string, object?>
        {
            ["timestamp"] = state.Time,
            ["next_scheduled_event_time"] = simulator.GetNextScheduledEventTime(),
        };
    }

    private Dictionary<string, object?> HandleRecordPolicyAction(JsonElement root)
    {
        var simulator = RequireSimulator();
        var timestamp = root.GetProperty("timestamp").GetDouble();
        var nextObservationTimestamp = root.GetProperty("next_observation_timestamp").GetDouble();
        var decision = RequirePolicy().Record(
            simulator,
            timestamp,
            root.GetProperty("action").GetString()!,
            nextObservationTimestamp);
        return new Dictionary<string, object?>
        {
            ["action"] = decision.Action.Key,
            ["timestamp"] = decision.Timestamp,
            ["next_observation_timestamp"] = nextObservationTimestamp,
            ["gcd_index"] = decision.GcdIndex,
        };
    }

    private Dictionary<string, object?> HandleApplyExternalEvent(JsonElement root)
    {
        var simulator = RequireSimulator();
        var eventKind = root.GetProperty("event_kind").GetString()!;
        var boolValue = root.TryGetProperty("value", out var valueElement)
            && valueElement.ValueKind is JsonValueKind.True or JsonValueKind.False
                ? valueElement.GetBoolean()
                : (bool?)null;
        var targetCount = root.TryGetProperty("target_count", out var targetCountElement)
            && targetCountElement.ValueKind == JsonValueKind.Number
                ? targetCountElement.GetInt32()
                : (int?)null;
        var remainingSeconds = OptionalDouble(root, "remaining_seconds");
        var result = simulator.ApplyExternalEvent(new ExternalCombatEvent(
            root.GetProperty("timestamp").GetDouble(),
            eventKind,
            boolValue,
            targetCount,
            remainingSeconds));
        return new Dictionary<string, object?>
        {
            ["accepted"] = result.Accepted,
            ["reason"] = result.Reason,
            ["event_kind"] = eventKind,
            ["timestamp"] = result.Timestamp,
            ["next_scheduled_event_time"] = simulator.GetNextScheduledEventTime(),
        };
    }

    private Dictionary<string, object?> HandleObserveAt(JsonElement root)
    {
        var simulator = RequireSimulator();
        var timestamp = root.GetProperty("timestamp").GetDouble();
        var format = root.TryGetProperty("format", out var formatElement)
            ? formatElement.GetString() ?? "vector"
            : "vector";
        if (format is not ("vector" or "seconds" or "gcd"))
        {
            throw new ArgumentException(
                $"unsupported observe format: {format}; supported=gcd, seconds, vector");
        }

        var nextObservationTimestamp = format == "vector"
            ? root.GetProperty("next_observation_timestamp").GetDouble()
            : timestamp;
        if (nextObservationTimestamp < timestamp)
        {
            throw new ArgumentOutOfRangeException(
                "next_observation_timestamp",
                nextObservationTimestamp,
                "next observation timestamp must not precede observation timestamp");
        }

        simulator.ObserveAt(timestamp);

        object context = format switch
        {
            "vector" => RequirePolicy().BuildVectorContext(simulator, nextObservationTimestamp),
            "seconds" or "gcd" => simulator.FormatState(format),
            _ => throw new UnreachableException(),
        };
        return new Dictionary<string, object?>
        {
            ["timestamp"] = simulator.Time,
            ["format"] = format,
            ["next_scheduled_event_time"] = simulator.GetNextScheduledEventTime(),
            ["context"] = context,
        };
    }

    private JobSimulator RequireSimulator() =>
        _simulator ?? throw new InvalidOperationException("sidecar session is not initialized");

    private PolicySession RequirePolicy() =>
        _policy ?? throw new InvalidOperationException("sidecar policy session is not initialized");

    private static double? OptionalDouble(JsonElement root, string key) =>
        root.TryGetProperty(key, out var element) && element.ValueKind == JsonValueKind.Number
            ? element.GetDouble()
            : null;

    private static string Reply(int seq, Dictionary<string, object?> body)
    {
        var payload = new Dictionary<string, object?>
        {
            ["seq"] = seq,
            ["ok"] = true,
        };
        foreach (var (key, value) in body)
        {
            payload[key] = value;
        }
        return JsonSerializer.Serialize(payload);
    }
}
