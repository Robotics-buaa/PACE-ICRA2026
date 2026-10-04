# Unitree G1 table-tennis task with a simulated wrist joint

本修改基于 PACE-ICRA2026，将乒乓球任务适配到 Isaac Lab 2.1.0 自带的 Unitree G1，并增加一个可控的仿真右腕自由度。保留原 T1 任务和原 G1 任务，新增任务使用独立的实验目录。

## 修改动机与最终行为

初版 G1 策略能够接近或碰到球，但在此前一次 10 次来球评估中未成功回球。检查确认模型的 `right_palm_joint` 是固定关节。本修改增加一个控制拍面方向的旋转自由度，并完成对应资产、执行器、动作、观测及复位接口的接入。

腕部版本完整训练后的单次 headless 评估统计为：

| 指标 | 结果 |
|---|---:|
| 来球数 | 400 |
| Success | 390/400，97.5% |
| Hits | 397/400，99.25% |

统计沿用项目现有的 Success/Hits 判定逻辑。Hits 不能作为独立接触传感器验证的替代。上述结果来自当前仿真配置下的一次评估，尚未开展多随机种子评估或实机测试；前后两次评估的样本量也不相同。

## 关键修改

- 增加 G1 机器人与固定球拍资产，使用名称匹配刚体和关节，适配接触传感器、奖励引用及复位事件。
- 根据 G1 默认姿态测量球拍相对基座的位置，校准任务目标与无效球预测的回退位置。
- 修复 G1 环境在关闭观测噪声时的历史缓冲初始化。
- 在独立 USD 资产中将 `right_palm_joint` 从 FixedJoint 改为 RevoluteJoint；保留原连接锚点与零位姿态。
- 旋转轴为零位时手掌局部 Z，限位 ±90°；球拍继续固定到手掌。
- 增加独立隐式 PD 执行器：stiffness=20、damping=1、effort_limit_sim=8 Nm、velocity_limit_sim=6 rad/s、armature=0.001。这些是仿真参数，不是实机规格。
- 策略动作由 23 增加到 24，总运动关节由 37 增加到 38；14 个手指关节继续保持默认位置。
- Actor 五帧输入由 435 增加到 450，输出由 23 增加到 24；Critic 五帧输入由 520 增加到 535。Predictor 输入/输出保持 15/3。
- 腕部实际关节状态和位置目标随指定环境复位。
- 保留 Isaac Lab 2.1.0 兼容修改：外力/力矩由世界坐标转换到物体坐标、移除不支持的 `is_global` 参数，以及使用 `quat_rotate_inverse`。

## 文件职责

| 路径 | 作用 |
|---|---|
| `legged_lab/assets/unitree/g1_tt.py` | 基础 G1 球拍资产配置及球拍几何常量 |
| `legged_lab/assets/unitree/G1_TT/` | 基础 G1 球拍 USD 与几何检查记录 |
| `legged_lab/assets/unitree/g1_tt_wrist.py` | 腕部版本资产配置及独立执行器 |
| `legged_lab/assets/unitree/G1_TT_WRIST/` | 引用基础 G1 的腕部 USD 与关节转换报告 |
| `legged_lab/envs/g1_tt/` | 基础 G1 任务配置、机器人接口和目标几何适配 |
| `legged_lab/envs/g1_tt_wrist/` | 在基础 G1 上增加腕部动作、观测与复位接口 |
| `legged_lab/envs/__init__.py` | 注册基础 G1 和腕部 G1 的训练、评估任务 |
| `legged_lab/physics/aerodynamics.py` | Isaac Lab 2.1.0 外力接口兼容修改 |
| `legged_lab/mdp/rewards.py` | 四元数 API 兼容修改 |
| `tools/migration_archive/` | 若采用归档整理方案，保存迁移、修复与诊断工具；日常训练和评估不依赖这些工具 |
| `check_g1_wrist_smoke.py`、`check_g1_wrist_full.py` | checkpoint 验收工具 |

包目录中的 `__init__.py` 一并提交。腕部任务继承基础 G1 环境，腕部 USD 引用基础 G1 USD，因此两套 G1 资产和环境代码均需保留。

## 已验证环境

- 宿主机 Ubuntu 20.04；现有 Docker 容器内 Ubuntu 22.04.5。
- Isaac Sim 4.5.0、Isaac Lab 2.1.0。
- NVIDIA RTX 4090 D 24 GB，驱动 575.57.08。
- 训练随机种子 42。

以下命令在 PACE 项目根目录执行，假设 IsaacLab 与本项目位于同一级目录。容器以 root 运行时设置 `OMNI_KIT_ALLOW_ROOT=1`。

```bash
export OMNI_KIT_ALLOW_ROOT=1

../IsaacLab/isaaclab.sh -p -m legged_lab.scripts.train \
  --task=g1_tt_wrist --num_envs=4096 --headless --predictor \
  --max_iterations=10000 \
  --experiment_name=g1_table_tennis_wrist \
  --run_name=g1_wrist_pred_4096_full \
  --logger=tensorboard --seed=42
```

评估前，将 `RUN_NAME` 设置为完整训练的实际目录名：

```bash
RUN_NAME="YYYY-MM-DD_HH-MM-SS_g1_wrist_pred_4096_full"

../IsaacLab/isaaclab.sh -p -m legged_lab.scripts.eval \
  --task=g1_tt_wrist_eval --num_envs=1 --headless \
  --load_run="$RUN_NAME" --checkpoint=model_9999.pt --predictor
```

模型目录为 `logs/g1_table_tennis_wrist/$RUN_NAME/model_9999.pt`。GUI 评估使用已配置的显示环境，并去掉 `--headless`。

## 验证记录

- 独立机械检查：±0.8 rad 目标跟踪通过，拍面法线随腕部转动，球拍保持固定连接；38 个运动关节、独立执行器与 ±90° 限位检查通过。
- 任务接口检查：2 个环境、24 个带噪声控制步、Actor/Critic/Predictor 前向计算、腕部动作目标及指定环境复位通过。正式任务保持重力与自由基座。
- 短训：64 个环境、30 次迭代、46,080 个时间步；最终 checkpoint iter=29，策略与 predictor 权重为有限值。策略优化器最大 step=600，predictor 优化器最大 step=152。
- 完整训练：4096 个环境、10000 次迭代、983,040,000 个时间步，耗时约 8 小时 36 分钟，最后 Mean reward=32.21。
- 完整训练最后一轮奖励：reward_future_pass_net=0.7472、reward_table_success=1.0463。这些是训练奖励，不是评估成功率。
- 完整模型评估：当前配置下 Success=390/400，Hits=397/400。

## 模型兼容性与范围

本次腕部训练使用新初始化的策略和 predictor，没有直接续训旧 23 维 G1 策略。旧模型输入、输出宽度与新任务不同；复用策略需要按每个历史帧的字段布局迁移输入权重，并扩展动作输出及相关状态。

新增腕部是对当前仿真模型的扩展，不代表原始 G1 资产或真实机器人具有相同硬件自由度。当前结果不构成对“缺少腕部是此前失败唯一原因”的证明。

保留 PACE-ICRA2026 原作者说明、原项目链接与现有许可证文件；注明基础 G1 资产来自 Isaac Lab。项目原有训练与评估入口继续复用。

## 提交前检查

- 提交最终生效的源码、包入口、本地 USD、必要记录和本说明，而不只提交修改脚本。
- 检查 `G1_TT_WRIST.usda` 是否包含 `/workspace/projects/PACE-ICRA2026/...` 绝对资产引用。原生成脚本会写入绝对路径；发布前应改为相对于该 USD 文件的 `../G1_TT/G1_TT.usda`，并验证资产加载。生成脚本若保留，也应相应处理导出引用。
- 检查基础 G1 USD 的上游资产引用是否可以在已声明环境中解析。部分早期迁移脚本固定了项目路径；若保留，注明其用途和路径假设。
- 不提交 `.bak`、`__pycache__`、临时日志、PID 文件、个人 SSH 配置或完整训练输出目录。最终模型若另行发布，记录实际下载地址、目录名和放置位置。
- 记录本次实际源码版本、完整训练目录名和评估命令，以便把 400 次来球结果与对应模型关联。
