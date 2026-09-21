using System.Text.Json;
using Combat.Sim.Common;
using Combat.Sim.SidecarHost;

namespace SidecarHost.Tests;

public sealed class SidecarSessionTests
{
    [Fact]
    public void 动作提交只返回轻量时序结果()
    {
        var session = CreateInitializedSession();
        using var response = Parse(session.Handle(
            "{\"op\":\"submit_action\",\"seq\":2,\"timestamp\":0.0,\"action\":\"fire_iii\"}"));
        var root = response.RootElement;

        Assert.True(root.GetProperty("ok").GetBoolean());
        Assert.True(root.GetProperty("accepted").GetBoolean());
        Assert.False(root.GetProperty("queued").GetBoolean());
        Assert.Equal(0.0, root.GetProperty("request_timestamp").GetDouble());
        Assert.Equal(3.0, root.GetProperty("effect_timestamp").GetDouble(), 5);
        Assert.False(root.TryGetProperty("state", out _));
        Assert.False(root.TryGetProperty("context", out _));
    }

    [Fact]
    public void Vector观测按七个Canonical块返回完整25候选上下文()
    {
        var session = CreateInitializedSession();
        using var response = Parse(session.Handle(
            "{\"op\":\"observe_at\",\"seq\":2,\"timestamp\":0.0," +
            "\"next_observation_timestamp\":2.5,\"format\":\"vector\"}"));
        var root = response.RootElement;

        Assert.True(root.GetProperty("ok").GetBoolean());
        var context = root.GetProperty("context");
        Assert.Equal(new[]
        {
            "job_tag",
            "schema_version",
            "scene_context",
            "skill_history_context",
            "state_history_context",
            "candidate_skill_context",
            "candidate_state_context",
        }, context.EnumerateObject().Select(property => property.Name));
        Assert.Equal(25, context.GetProperty("candidate_skill_context").GetArrayLength());
        Assert.Equal(
            25,
            context.GetProperty("candidate_state_context").GetProperty("tokens").GetArrayLength());
        var candidateToken = context.GetProperty("candidate_state_context").GetProperty("tokens")[0];
        Assert.Equal(
            new[] { "player_state", "buff_state", "target_buff_state", "resource_state" },
            candidateToken.EnumerateObject().Select(property => property.Name));
        Assert.Equal(147,
            candidateToken.EnumerateObject().Sum(property => property.Value.GetArrayLength()));
        Assert.DoesNotContain("animation_lock", context.GetRawText());
    }

    [Fact]
    public void 初始化时间允许覆盖首个负时间预读()
    {
        var session = new SidecarSession(RepoRootLocator.Find());
        using var init = Parse(session.Handle(
            "{\"op\":\"init\",\"seq\":1,\"job_tag\":\"black_mage\",\"initial_timestamp\":-5.0}"));
        Assert.Equal(-5.0, init.RootElement.GetProperty("timestamp").GetDouble());

        using var observation = Parse(session.Handle(
            "{\"op\":\"observe_at\",\"seq\":2,\"timestamp\":-3.0," +
            "\"next_observation_timestamp\":0.0,\"format\":\"vector\"}"));
        Assert.True(observation.RootElement.GetProperty("ok").GetBoolean());
        Assert.Equal(-3.0, observation.RootElement.GetProperty("timestamp").GetDouble());
    }

    [Fact]
    public void Policy动作只写策略历史且不推进游戏状态()
    {
        var session = CreateInitializedSession();
        using var recorded = Parse(session.Handle(
            "{\"op\":\"record_policy_action\",\"seq\":2,\"timestamp\":0.0," +
            "\"action\":\"ogcd_wait\",\"next_observation_timestamp\":2.5}"));
        var recordedRoot = recorded.RootElement;

        Assert.True(recordedRoot.GetProperty("ok").GetBoolean());
        Assert.Equal("ogcd_wait", recordedRoot.GetProperty("action").GetString());
        Assert.Equal(0.0, recordedRoot.GetProperty("timestamp").GetDouble());
        Assert.False(recordedRoot.TryGetProperty("state", out _));
        Assert.False(recordedRoot.TryGetProperty("context", out _));

        using var observation = Parse(session.Handle(
            "{\"op\":\"observe_at\",\"seq\":3,\"timestamp\":0.0," +
            "\"next_observation_timestamp\":2.5,\"format\":\"vector\"}"));
        var observationRoot = observation.RootElement;
        Assert.Equal(0.0, observationRoot.GetProperty("timestamp").GetDouble());
        var history = observationRoot.GetProperty("context").GetProperty("skill_history_context");
        Assert.Single(history.EnumerateArray());
        Assert.Equal("ogcd_wait", history[0].GetProperty("skill_key").GetString());
    }

    [Fact]
    public void 外部事实使用绝对时间且非法载荷不会推进游标()
    {
        var session = CreateInitializedSession();
        using var targetCount = Parse(session.Handle(
            "{\"op\":\"apply_external_event\",\"seq\":2,\"timestamp\":1.0," +
            "\"event_kind\":\"target_count_changed\",\"target_count\":3}"));
        Assert.True(targetCount.RootElement.GetProperty("ok").GetBoolean());
        Assert.True(targetCount.RootElement.GetProperty("accepted").GetBoolean());
        Assert.False(targetCount.RootElement.TryGetProperty("state", out _));

        using var raidBuff = Parse(session.Handle(
            "{\"op\":\"apply_external_event\",\"seq\":3,\"timestamp\":1.0," +
            "\"event_kind\":\"raid_buff_window_changed\",\"value\":true," +
            "\"remaining_seconds\":10.0}"));
        Assert.True(raidBuff.RootElement.GetProperty("ok").GetBoolean());

        using var invalid = Parse(session.Handle(
            "{\"op\":\"apply_external_event\",\"seq\":4,\"timestamp\":2.0," +
            "\"event_kind\":\"target_count_changed\",\"target_count\":-1}"));
        Assert.False(invalid.RootElement.GetProperty("ok").GetBoolean());

        using var observation = Parse(session.Handle(
            "{\"op\":\"observe_at\",\"seq\":5,\"timestamp\":1.0,\"format\":\"seconds\"}"));
        Assert.True(observation.RootElement.GetProperty("ok").GetBoolean());
        Assert.Equal(1.0, observation.RootElement.GetProperty("timestamp").GetDouble());
    }

    [Fact]
    public void 观测响应带下一事件时刻供宿主调度()
    {
        var session = CreateInitializedSession();
        using var response = Parse(session.Handle(
            "{\"op\":\"observe_at\",\"seq\":2,\"timestamp\":0.0," +
            "\"next_observation_timestamp\":2.5,\"format\":\"vector\"}"));
        var root = response.RootElement;

        Assert.True(root.GetProperty("ok").GetBoolean());
        // 宿主 DecisionScheduler 需要该字段决定下一次调用时刻，不能只出现在动作/推进响应里。
        Assert.True(root.TryGetProperty("next_scheduled_event_time", out _));
    }

    [Fact]
    public void 只读探测返回时间锁原因且不提交动作()
    {
        var session = CreateInitializedSession();
        using var submitted = Parse(session.Handle(
            "{\"op\":\"submit_action\",\"seq\":2,\"timestamp\":0.0,\"action\":\"fire_iii\"}"));
        Assert.True(submitted.RootElement.GetProperty("accepted").GetBoolean());

        using var locked = Parse(session.Handle(
            "{\"op\":\"validate_at\",\"seq\":3,\"timestamp\":1.0,\"action\":\"fire_iii\"}"));
        var lockedRoot = locked.RootElement;
        Assert.True(lockedRoot.GetProperty("ok").GetBoolean());
        Assert.False(lockedRoot.GetProperty("legal").GetBoolean());
        Assert.False(string.IsNullOrEmpty(lockedRoot.GetProperty("reason").GetString()));

        // 探测不产生第二个动作实例：读到动作生效之后，历史里仍只有第一次提交的动作。
        using var observation = Parse(session.Handle(
            "{\"op\":\"observe_at\",\"seq\":4,\"timestamp\":4.0," +
            "\"next_observation_timestamp\":4.0,\"format\":\"vector\"}"));
        Assert.Single(observation.RootElement
            .GetProperty("context").GetProperty("skill_history_context").EnumerateArray());
    }

    private static SidecarSession CreateInitializedSession()
    {
        var session = new SidecarSession(RepoRootLocator.Find());
        using var response = Parse(session.Handle(
            "{\"op\":\"init\",\"seq\":1,\"job_tag\":\"black_mage\"}"));
        Assert.True(response.RootElement.GetProperty("ok").GetBoolean());
        Assert.False(response.RootElement.TryGetProperty("state", out _));
        Assert.False(response.RootElement.TryGetProperty("context", out _));
        return session;
    }

    private static JsonDocument Parse(string? response)
    {
        Assert.NotNull(response);
        return JsonDocument.Parse(response);
    }
}
