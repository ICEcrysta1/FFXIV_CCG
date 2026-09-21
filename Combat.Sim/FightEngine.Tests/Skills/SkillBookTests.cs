using Combat.Sim.Models.Definitions;
using Combat.Sim.Skills;
using FightEngine.Tests.Facade;
using FightEngine.Tests.System;
using Xunit;

namespace FightEngine.Tests.Skills;

public class SkillBookTests
{
    private static SkillDefinition Skill(string key, int gameId, ActionKind kind = ActionKind.Gcd, bool enabled = true) =>
        SystemTestKit.Skill(key, kind, gameId: gameId) with { Enabled = enabled };

    [Fact]
    public void 直接构造技能时集合字段默认为空集合()
    {
        var skill = new SkillDefinition("plain", 99, "普通技能", ActionKind.Gcd, "standard");

        Assert.Empty(skill.AppliesStatuses);
        Assert.Empty(skill.Tags);
    }

    [Fact]
    public void 按key与游戏id查询()
    {
        var book = new SkillBook(new[] { Skill("a", 1), Skill("b", 2) });
        Assert.Equal("a", book.Get("a").Key);
        Assert.Equal("b", book.Get(2).Key);
        Assert.True(book.Contains("a"));
        Assert.True(book.Contains(2));
        Assert.False(book.Contains("missing"));
        Assert.False(book.Contains(99));
    }

    [Fact]
    public void 未命中抛KeyNotFound()
    {
        var book = new SkillBook(new[] { Skill("a", 1) });
        Assert.Throws<KeyNotFoundException>(() => book.Get("missing"));
        Assert.Throws<KeyNotFoundException>(() => book.Get(99));
    }

    [Fact]
    public void 重复key或game_id抛错()
    {
        Assert.Throws<InvalidOperationException>(() => new SkillBook(new[] { Skill("a", 1), Skill("a", 2) }));
        Assert.Throws<InvalidOperationException>(() => new SkillBook(new[] { Skill("a", 1), Skill("b", 1) }));
    }

    [Fact]
    public void 启用技能按类型过滤()
    {
        var book = new SkillBook(new[]
        {
            Skill("gcd_a", 1, ActionKind.Gcd),
            Skill("ogcd_a", 2, ActionKind.Ogcd),
            Skill("gcd_disabled", 3, ActionKind.Gcd, enabled: false),
        });

        Assert.Equal(new[] { "gcd_a", "ogcd_a" }, book.EnabledSkills().Select(s => s.Key).ToArray());
        Assert.Equal(new[] { "gcd_a" }, book.EnabledSkills(ActionKind.Gcd).Select(s => s.Key).ToArray());
        Assert.Equal(new[] { "ogcd_a" }, book.EnabledSkills(ActionKind.Ogcd).Select(s => s.Key).ToArray());
        Assert.Equal(new[] { "gcd_a", "ogcd_a", "gcd_disabled" }, book.Keys().ToArray());
    }

    [Fact]
    public void 项目配置按系统加职业顺序合并()
    {
        var config = FacadeKit.BuildProjectConfig();
        var book = SkillBook.FromProjectConfig(config);

        var keys = book.Keys();
        Assert.Equal(new[] { "potion", "gcd_strike", "ogcd_punch", "cd_skill", "cast_skill" }, keys.ToArray());
    }
}
