# 任务 L 续跑：审图回复（协调者，2026-11-10）

上一个 worker（dispatch `ctx_66fa10ede724`）已完成 §3 全部检查和 §4 试算图，因协调者未及时回复审图问题 `msg_c9f61262c42b` 而按 failed 结束。产出在 `work/side_L_wifimix_20261110/`（preflight.json、protected_before.json、fig_L_pilot_2.4GHz.png、pilot_duty.csv、REPORT.md 等）。**本文是对该问题的回复，你从 §4 的全量生成继续。** 原任务书 `work/codex_tasks/side_TASK_L_2.4G无人机叠加WiFi训练_20261110.md` 其余内容全部照旧。

## 1. 审图结论：X 范围沿用 [−5, 15] dB，不改

协调者看过 fig_L_pilot_2.4GHz.png：
- 11 个型号中 9 个在 −5 dB 时突发清楚可见（Flip、air 3s、avata、avata2、mini 4k、mini4 pro、mini5 pro、neo、neo2）。
- mavic 4：−5 dB 时只有窄突发可见，宽带竖条要到 5 dB 以上才明显。
- atom se：该原片本身几乎没有信号（突发占空比 0.003），所以任何 X 下都看不出无人机。这是原片的问题，与 X 范围无关，收窄范围解决不了。
- air 3s 的占空比为 0 是指标问题（宽带信号铺满频带，中位数被抬高），图上明显可见。不用修这个指标，在 REPORT 里写一句说明即可。
- bgA + 背景：−5/0 dB 时像原 v3 背景，15 dB 时像原 bgA，符合预期。

## 2. 续跑要求

1. **不要重做 §3**。沿用上一轮的 `protected_before.json` 作为保护目录基线，交付前再算一次 sha256，与它比较。preflight 结果直接引用上一轮的。
2. 开始全量生成前再查一次 `nvidia-smi`。若有其他 python 计算进程在用 GPU，用 `ask` 问协调者。其余情况一律不要停下来问，自己决定并写进 REPORT。
3. 之后按原任务书 §4（全量生成）→ §5（冒烟 + 6 次训练）→ §6（评估）→ §7（交付）执行。
4. **新增一项评估**（写进 §6 第 2 项）：H 的 `2.4GHz_mix_data_drone` 叠加检验，按 data 无人机**型号**分别给出每个 X 的检出率（J / L-a / L-b 并列，3 种子均值）。重点看 atom se 和 mavic 4。
5. **REPORT 加一段说明**（只描述）：功率比按整片平均功率计算，整片功率里包含接收机 A 的底噪。所以 X 高时叠加样本里的 WiFi 会被接收机 A 底噪盖住（试算图 15 dB 列可见）。这与 `mix_test.py` 的定义一致，本任务不改。
6. 原任务书要求重写的 REPORT.md 直接覆盖上一轮的版本，但在开头保留一行“续跑自 ctx_66fa10ede724，§3/§4 试算沿用上一轮”。

完成时发一次 `worker_done`（`--report-path work/side_L_wifimix_20261110/REPORT.md`）。不要发送 heartbeat，不要调用 DSH / dsh-cli / mcp__dsh。
