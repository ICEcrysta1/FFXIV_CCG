"""policy 控制动作配置与词表测试。"""

from __future__ import annotations

from common.policy.data.policy_actions import load_policy_actions
from common.policy.data.skill_vocab import SkillVocab


def test_policy_action_zero_has_a_non_padding_vocab_token():
    actions = load_policy_actions()
    vocab = SkillVocab.build_from_job_tag("black_mage")

    assert [(action.key, action.raw_id) for action in actions] == [("ogcd_wait", 0)]
    assert vocab.lookup(0) > 0
    assert vocab.reverse_lookup(vocab.lookup(0)) == 0
