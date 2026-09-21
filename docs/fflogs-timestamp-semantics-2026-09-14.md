# FFLogs raw JSON 时间戳语义实测（2026-09-14）

> 目的：用真实数据固定 FFLogs 战斗日志里每类事件时间戳的真实含义，作为转换层
> （request time 还原、forced_move/slidecast 判定、DoT 处理）的语义依据。
>
> 数据来源：黑魔 raw 目录抽查 14 个文件交叉验证，主分析对象为
> `data/human/job/black_mage/raw/M5s/fflogs_rc3wfM7FTt4GAdYL_f1_Sol_Gaia.json`
> （1701 事件，497.8s）与 `FRU/fflogs_18RvraJXZgb97dmq_f33_Yris_Vhenn.json`
> （4700 事件，无团本加速机制对照）。技能标称读条取自 `config/skills/black_mage.yaml`。

## 1. 文件与顶层结构

- 每个 JSON = 单场战斗（`fights` 长度 1），按玩家导出，文件 1~2.7MB。
- 顶层 dict：`report_code / fight_id / source_id / player_name / player_dps / fetched_at / fights(1) / damage_table / events / event_count`。
  这是项目抓取器的自定义导出格式（含 `simulatedCrit`、`calculateddamage` 等合成字段），不是 FFLogs API 原始返回。
- `timestamp` 单位毫秒，**原点是整份 report 的起点，不是战斗起点**；战斗相对时间 = `timestamp - fights[0].start_time`。
- 事件已裁剪到战斗窗口 `[start_time, end_time]`：14 个抽查文件全部 `min(相对时间)=0`，**没有负时间戳事件**。
- 事件公共字段：`timestamp, type, sourceID, targetID, abilityGameID, fight, packetID`。
  cast 与 damage 都携带 `packetID`，可配对同一次动作实例。
- `begincast` 独有 `duration`（毫秒）；`damage` 独有 `amount / hitType / multiplier / unmitigatedAmount`。
- 合成事件 `calculateddamage / calculatedheal` 与对应 `cast` 同一时间戳、同金额（导出器在 cast 时刻做模拟暴击/直击结算）。

## 2. 三类时刻的真实语义（核心结论）

| 事件 | 真实语义 |
| --- | --- |
| `begincast` | 读条开始（≈按键）时刻；`duration` = 该玩家加速后的实际读条时长。仅硬读条技能有 |
| `cast` | 服务器结算动作的时刻：硬读条时比"条走完"早约 0.5~0.55s（滑步窗口）；瞬发时 ≈ 按键时刻。它是 raw 日志里最接近"生效"的时间点 |
| `damage` | 伤害真正落到目标上的时刻，比 `cast` 晚 0.5~1.3s，且按技能高度固定 |

### 2.1 实测证据

黑魔 7.x 标称读条：fire_iv / blizzard_iv / flare / flare_star = 2.0s，
fire_iii / blizzard_iii = 3.5s，despair / xenoglossy / paradox = 0s 瞬发。

| 技能 (gameID) | 标称 | begincast→cast | cast→damage |
| --- | --- | --- | --- |
| Fire IV (3577) | 2.0s | 1382ms（n=112，duration=1926） | 1158ms（n=139，min 1154 / max 1164，极稳） |
| Blizzard IV (3576) | 2.0s | 1382ms（duration=1926） | 1158ms（与 F4 完全同构） |
| Flare (162) | 2.0s | 1381ms | 1160ms |
| Fire III (152) | 3.5s | （M5S 开怪预读无 begincast） | 1292ms |
| Despair (16505) | 0s 瞬发 | 无 begincast（22 次 cast 0 次） | 490ms |
| Xenoglossy (16507) / Paradox (25797) | 0s 瞬发 | 无 begincast | 624ms |

统计口径均为 fight 相对时间的中位数。FRU 玩家 GCD 节奏 ≈2364ms（有急速装）；
M5S 出现 ~2100ms 加速期 / ~2400ms 常态两档，对应 `begincast.duration` 的 1662/1956 两档——
**duration 随玩家实际加速缩放，就是标称读条时长在游戏内的实际值**。

### 2.2 关键规律

- `cast − begincast = duration − 约 545ms`（FRU：1926−1384=542；M5S：531、501），
  即服务器在条结束前 ~0.55s 就结算动作——这就是滑步（slidecast）窗口的直接证据。
- `damage` 比 `cast` 的偏移按技能固定（F4/B4/Foul≈1158ms、F3≈1292ms、B3≈890ms、
  绝望≈490ms、异言/悖论≈624ms、高闪雷≈758ms）；FRU 665 对同 packetID 配对中
  0 例 damage 早于 cast。
- 瞬发（原生瞬发或三连/迅速瞬发化）没有 `begincast`，`cast` ≈ 按键时刻。
- 纯工具 oGCD（transpose、swiftcast、triplecast、ley_lines、manafont、amplifier、
  umbral_soul 等）只有 cast、无 damage。

## 3. 事件 type 清单

`damage, cast, heal, calculateddamage, calculatedheal, applybuff, refreshbuff,
removebuff, removebuffstack, applybuffstack, applydebuff, refreshdebuff,
removedebuff, absorbed, begincast, combatantinfo, tether, death, headmarker, dispel`

## 4. DoT 语义

- 施加技能与 DoT 是**不同 gameID**（如 High Thunder 36986 施加，DoT 为 1003871 / 1003872）；
  cast 仅 24 次而 damage tick 227 次。
- tick 间隔中位 2987ms ≈ 3.0s，与游戏 3 秒一跳一致。
- **首跳不立即结算**：cast → 首 tick 差值 0.8~3.7s（中位 1.9~2.5s），首跳落在施加后的下一个 3s 节拍上。
- tick 事件的 `packetID` = 施加 DoT 那次 cast 的 packetID（一跳 DoT 的全部 tick 都能配对到施加动作）。

## 5. 时间原点与 prepull

- 无负时间戳；事件全部裁剪到 `[fight.start_time, fight.end_time]`。
- prepull 只能间接识别：开怪后很快出现 `cast` 而没有对应 `begincast`
  （预读的 begincast 发生在开怪前、被窗口裁掉）。M5S 全量 195 个文件的首个玩家
  `cast` 均为 Fire III，战斗相对时刻为 0~1430ms，且 0/195 在文件内有此前对应的
  `begincast`；首个完整 `begincast` 要到后续战斗事件才出现。

## 6. 对转换层的含义

`scripts/convert_fflogs` 的 request time 判定顺序：

1. 有可按 duration 闭环配对的 `begincast` → `request = begincast`。这是日志直接记录的
   读条开始/按键时刻，不再为了让状态机效果事件贴合 `cast` 而反推几十毫秒偏移。
2. 瞬发动作（含瞬发化）→ `request = cast`。
3. 仅开怪首个硬读条例外：它的 `begincast` 被战斗窗口裁掉，按实测 GCD 缩放技能表
   读条时长，再用 `request = cast − 实际读条时长 + slidecast_window` 恢复预读请求。
4. 恢复后整场动作与场景事实统一平移，使最早请求为 0；同一份日志只做一次原点变换。

2026-09-15 按上述规则全量回放 M5S：102/195（52.3%）转换成功；此前对普通硬读条
使用 `cast − duration + 0.5` 的同版状态机结果为 99/195（50.8%）。直接使用
`begincast` 净增 3 场，但剩余失败仍以动作锁冲突为主，不能仅靠请求时刻来源解决。

由此：

- 状态机在 `request + max(duration − slidecast_window, 0)` 应用效果，完整读条锁仍在
  `request + duration` 结束。普通硬读条从真实 `begincast` 起算；日志 `cast` 与状态机效果事件
  的几十毫秒偏差由容量一动作队列吸收，不写入请求时间。
- `begincast.duration` 是"当时实际读条时长"的权威来源，可直接用于
  forced_move / instant_move（滑步）判定，不必依赖状态机查询或 Python 侧重算缩放公式。
- 盲区：三连/迅速瞬发化的技能没有 begincast；开怪预读的 begincast 被裁剪；
  偶发 begincast 延迟到达时 `duration` 反映的是剩余读条而非完整读条
  （M5S 出现 1 例 Fire III duration=1711ms 低于同档正常值），使用时应与同玩家同技能的其他样本核对。
