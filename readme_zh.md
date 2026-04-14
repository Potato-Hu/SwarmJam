# SwarmJam

一个面向多无人机协同对抗的研究型仿真与强化学习框架。

---

# 1. 项目简介

SwarmJam 聚焦于如下问题：

> 在感知不完备和数据关联不确定的条件下，多无人机如何协同压制敌方关键节点？

本项目结合：

- 敌方集群仿真（ground truth）
- 全局与局部感知建模
- 简化 JPDA 的关键节点关联方法
- 基于 MAPPO 的多智能体强化学习
- 干扰与信道模型（含莱斯衰落）

实现一个**从感知 → 关联 → 决策 → 干扰 → 奖励闭环**的完整系统。

---

# 2. 核心思想

系统核心流程如下：

```
真实状态 → 全局感知（带噪） → 局部感知
                ↓
          关键节点关联（简化 JPDA）
                ↓
           关键目标估计
                ↓
             MAPPO
                ↓
        无人机机动与干扰行为
                ↓
            干扰效果计算
                ↓
                奖励
```

---

# 3. 研究目标

本项目的目标是：

- 在存在 clutter（非关键节点）情况下识别关键节点
- 在观测噪声与缺失条件下保持鲁棒性
- 实现多无人机协同压制关键目标
- 控制对非关键节点与友方的副作用

---

# 4. 项目结构

```
SwarmJam/
├── config/           # 参数配置
├── src/
│   ├── env/          # 强化学习环境
│   ├── simulation/   # 真值生成
│   ├── sensing/      # 感知模型
│   ├── association/  # 关键节点关联（核心模块）
│   ├── interference/ # 干扰与信道模型
│   ├── rl/           # MAPPO
│   ├── runner/       # 训练与评估入口
│   └── visualization/# 可视化
├── outputs/          # 训练结果
└── docs/             # 文档
```
SwarmJam/
├── README.md                         # 项目说明
├── requirements.txt                  # 依赖
├── .gitignore
│
├── config/                           # 所有超参与场景配置
│   ├── default.yaml                  # 默认全局配置（agent 数、key/non-key 比例、噪声协方差、训练 epoch 等）
│   ├── env.yaml                      # 环境与动力学参数（地图范围、dt、速度上限、感知半径等）
│   ├── mappo.yaml                    # MAPPO 专用训练配置
│   ├── jammer.yaml                   # 干扰功率、友方干扰阈值、莱斯信道参数等
│   ├── association.yaml              # 关键节点关联模块配置（softmax 温度、阈值、历史长度等）
│   └── scenarios/                    # 不同实验场景（clutter 密度、key 数量）
│       ├── scenario_base.yaml        # 基础场景
│       ├── scenario_high_clutter.yaml# 高 clutter（non-key 多）消融场景
│       ├── scenario_high_noise.yaml  # 高雷达噪声场景
│       ├── scenario_sparse_key.yaml  # key 节点更分散的场景
│       └── scenario_dense_enemy.yaml # 敌方节点整体更密集的场景
│
├── src/
│   ├── __init__.py
│   │
│   ├── env/                          # 多无人机对抗环境
│   │   ├── __init__.py
│   │   ├── swarm_env.py              # 主环境：step/reset/reward/obs 拼接
│   │   ├── world.py                  # 世界状态容器，统一维护敌我节点
│   │   ├── dynamics.py               # 运动学更新模型
│   │   └── spaces.py                 # 状态、动作空间定义
│   │
│   ├── entities/                     # 环境实体
│   │   ├── __init__.py
│   │   ├── friendly_uav.py           # 我方干扰无人机
│   │   ├── enemy_node.py             # 敌方节点（key / non-key）
│   │   └── swarm_group.py            # 敌我群体初始化与管理
│   │
│   ├── simulation/                   # 仿真真值与场景生成
│   │   ├── __init__.py
│   │   ├── initializer.py            # 初始位置/速度/类型采样
│   │   ├── ground_truth.py           # 敌方节点真值状态生成与更新
│   │   └── trajectory_generator.py   # 敌方轨迹生成
│   │
│   ├── sensing/                      # 感知模块
│   │   ├── __init__.py
│   │   ├── global_sensor.py          # 外部全局感知：带噪 key 节点观测
│   │   ├── local_sensor.py           # agent 局部感知：范围内局部观测
│   │   ├── noise.py                  # 噪声模型、协方差采样
│   │   └── observation_builder.py    # 组织 global_obs / local_obs / agent_obs
│   │
│   ├── association/                  # 关键节点关联模块
│   │   ├── __init__.py
│   │   ├── scorer.py                 # 计算 S_kj 关联得分
│   │   ├── soft_association.py       # softmax + threshold 简化 JPDA
│   │   ├── history_buffer.py         # 历史观测缓存
│   │   ├── key_target_builder.py     # 生成 key_targets
│   │   └── confidence_filter.py      # 置信度筛选与缺失处理
│   │
│   ├── interference/                 # 干扰与信道模型
│   │   ├── __init__.py
│   │   ├── channel.py                # 路损 + 莱斯衰落
│   │   ├── interference_eval.py      # 计算 I_k(t), 发射功率约束(配置),友方干扰
│   │   └── utility.py                # key gain / non-key penalty 等效能计算
│   │
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── policy/                   # 策略网络
│   │   │   ├── __init__.py
│   │   │   ├── actor.py
│   │   │   ├── critic.py
│   │   │   └── feature_encoder.py    # 编码 self state + key targets
│   │   ├── mappo.py              # MAPPO 主实现
│   │   ├── buffer.py    # 轨迹缓存 / rollout buffer
│   │   └── reward.py     # r = key_gain - nonkey_penalty - friendly_penalty
│   │
│   ├── runner/                       # 训练与评估调度
│   │   ├── __init__.py
│   │   ├── train.py                  # 训练入口
│   │   └── evaluate.py               # 测试入口
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config_loader.py          # YAML 配置读取
│   │   ├── logger.py                 # 日志工具
│   │   ├── seed.py                   # 随机种子
│   │   ├── checkpoint.py             # 模型保存/加载
│   │   └── metrics.py                # 指标统计
│   │
│   └── visualization/                # 可视化
│       ├── __init__.py
│       ├── plot_scene.py             # 场景快照
│       ├── plot_training.py          # 训练曲线
│       ├── animate_episode.py        # 回合动画
│       └── plot_association.py       # 关键节点关联结果可视化
│
│
│
├── outputs/                          # 训练输出
│   ├── logs/
│   ├── checkpoints/
│   ├── figures/
│   └── videos/
│
│
└── docs/                             # 文档
    ├── system_model.md               # 系统模型
    └── problem_formulation.md        # 问题建模

---

# 5. 系统模块说明

## 5.1 Simulation（仿真真值）

- 生成敌方节点（key / non-key）
- 提供真实轨迹

## 5.2 Sensing（感知）

- 全局感知：带噪关键节点观测
- 局部感知：无人机附近观测（无标签）

## 5.3 Association（关键模块）

- 基于距离的 soft association
- 使用 softmax 近似 JPDA
- 输出关键目标估计

## 5.4 Interference（干扰建模）

- 路损模型
- 莱斯衰落
- 多无人机叠加干扰

## 5.5 RL（MAPPO）

- 集中式 critic
- 分布式 actor
- 多智能体协同优化

---

# 6. 强化学习建模

## 状态

- 全局状态（用于 critic）
- 局部观测（用于 actor）

## 动作

- 无人机速度控制（可扩展功率控制）

## 奖励函数

```
r = key_gain - nonkey_penalty - friendly_penalty
```

- key_gain：关键节点压制效果
- nonkey_penalty：误干扰
- friendly_penalty：友方干扰

---

# 7. 快速开始

## 安装依赖

```bash
pip install -r requirements.txt
```

## 训练

```bash
python -m src.runner.train --scenario scenario_base
```

## 测试

```bash
python -m src.runner.evaluate
```

---

# 8. 实验设计

支持多种场景：

- 基础场景
- 高 clutter 场景
- 高噪声场景

可用于：

- 消融实验（association 模块）
- 鲁棒性分析
- 协同策略评估

---

# 9. 项目特点

- 端到端闭环建模
- 明确区分 key / non-key
- 结合数据关联与强化学习
- 结构清晰，易于扩展

---

# 10. 适用方向

- 多无人机协同控制
- 对抗博弈
- 目标跟踪与数据关联
- 强化学习在通信/电子对抗中的应用

---

# 11. 后续扩展方向

- 引入更复杂的信道模型
- 引入通信约束
- 更真实的传感器模型
- 多类型干扰策略

---

# 12. License

仅用于学术研究用途。

---

# 13. 联系方式

如需交流，请通过学术渠道联系。

