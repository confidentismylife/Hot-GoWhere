# v2.0 架构设计：多模态感知 + 扩散模型轨迹生成

> Status: 设计阶段 | Target GPU: H800 (80GB) | 基于 v1.0-final

---

## 一、v1.0 → v2.0 变更总览

```
v1.0 (当前)                        v2.0 (目标)

感知: 数值→NL文本                   感知: 数值→NL文本 + VLM画面理解
决策: Qwen-3B-AWQ                  决策: Qwen-7B-FP16 (质量升级)
执行: 社会力模型(1995)              执行: 扩散模型轨迹生成(2024 SOTA)
GPU:  4090 24GB                    GPU:  H800 80GB
```

| 模块 | v1.0 | v2.0 | 理由 |
|------|------|------|------|
| LLM | Qwen-3B-AWQ | Qwen-7B-FP16 | H800 80GB 不差这点显存 |
| 感知补充 | 无 | Qwen-VL-7B | CCTV画面语义理解 |
| 轨迹生成 | Social Force Model | Diffusion Policy (轻量U-Net) | 轨迹更自然, 可多模态采样 |
| 训练数据 | 无 | v1.0仿真生成5000条+ETH/UCY | 先预训练再微调 |

---

## 二、多模态感知层设计

### 2.1 双通道感知

```
CCTV画面 (640×480 RGB)
    │
    ├──→ 通道A: YOLOv9 检测 (快速, 结构化)
    │        输出: person_boxes[], crowd_density_map
    │
    └──→ 通道B: Qwen-VL-7B (慢速, 语义)
             Prompt: "描述这个画面的烟雾分布、人群异常行为、
                     可见的障碍物、地面状况。限100字。"
             输出: "西南角浓烟弥漫,能见度约5米。东出口有约30人
                   排队,移动缓慢。地面有水迹反光,可能有消防喷淋
                   启动。柱子后方蹲着一个人,判断可能受伤。"

融合策略:
  通道A数据 → 结构化数值 (已有)
  通道B数据 → 自然语言描述 (新增)
  ↓
  NL Converter 合并:
  """
  [环境状态-传感器]
  烟雾浓度45%, 温度52°C...
  
  [环境状态-视觉理解]
  西南角浓烟弥漫, 能见度约5米...
  
  [个人状态] ...
  """
```

### 2.2 VLM 调用策略

不是每帧都调用 VLM（太慢太贵），而是：

```
触发条件 (满足任一即调用):
  - 每30 tick (3秒) 定时采样
  - 烟雾浓度变化 >20%
  - 新着火点出现
  - Agent密度某区域突变

每次调用:
  - 输入: 当前关键帧 (640×480)
  - 模型: Qwen-VL-7B-FP16
  - 耗时: ~1.5s (H800)
  - 输出: 100字场景描述
  - 缓存: 结果共享给所有Agent (同一画面,不需每人都调)
```

### 2.3 显存预算 (H800 80GB)

```
模型驻留显存:
  Qwen-7B-FP16 (决策LLM)      14 GB
  Qwen-VL-7B-FP16 (VLM)       16 GB
  扩散模型 U-Net (轨迹生成)     2 GB
  KV Cache (7B, batch 128)    20 GB
  Python开销 + Agent数据        8 GB
  ─────────────────────────────────
  合计                         60 GB / 80 GB
  剩余安全余量                  20 GB ✓
```

---

## 三、扩散模型轨迹生成层设计

### 3.1 模型选型

不从头训一个扩散模型。用预训练的轻量行人轨迹扩散模型做backbone，在你的仿真数据上微调。

| 候选 Backbone | Params | 推理速度 | 显存 | 来源 |
|---------------|--------|:---:|------|------|
| MID (Gu 2022) | 15M | ~50ms | 0.5GB | 条件扩散轨迹预测 |
| LED (Mao 2023) | 22M | ~80ms | 0.8GB | 隐空间扩散 |
| MotionDiffuser (Jiang 2023) | 30M | ~120ms | 1.2GB | 多智能体联合扩散 |

推荐 **MID** ——最轻量，条件注入机制成熟，和LLM的条件接口好对接。

### 3.2 训练流水线

```
Phase 1: 预训练 (ETH/UCY 数据集)
  - 几万条真实行人轨迹
  - 学习基础行为: 直行、转弯、避让、跟随
  - 与LLM无关, 纯视觉-轨迹映射

Phase 2: 微调 (你的仿真数据)
  - v1.0 仿真生成 5000-10000 条火灾疏散轨迹
  - 每条包含: 起点, 出口, 障碍物, LLM决策文本, 生成的轨迹
  - 微调扩散模型适应:
      - 烟雾减速行为 (正常数据里没有)
      - 弯腰爬行低姿态
      - 多人拥堵排队
      - 恐慌奔跑

Phase 3: 部署
  - 微调后的模型替代社会力模型
  - 每3秒生成一次轨迹 (31步)
  - 物理tick只做插值播放
```

### 3.3 推理接口

```python
# execution/diffusion_policy.py (新增)

class DiffusionTrajectoryGenerator:
    """条件扩散模型: LLM意图 + 场景 → 平滑轨迹"""

    def generate(self,
                 start: np.ndarray,          # [2] 起点
                 target: np.ndarray,         # [2] 出口
                 llm_decision: str,          # LLM的reasoning文本
                 scene_map: np.ndarray,      # [H,W,3] 障碍物+烟雾图
                 others_trajs: np.ndarray,   # [K,31,2] 其他人的预测轨迹
                 num_steps: int = 31,
                 ) -> np.ndarray:            # [31,2] 生成的轨迹
        """
        1. 条件编码: llm_decision → BERT → [256]
                    scene_map → CNN → [256]
                    others_trajs → Social Encoder → [256]
                    concat → [768]

        2. 扩散去噪: 纯噪声[31,2] → 100步DDIM → 平滑轨迹

        3. 输出: 31步坐标序列
        """
```

### 3.4 为什么扩散模型比社会力模型好（论文论据）

| 对比维度 | 社会力模型 | 扩散模型 | 论文价值 |
|----------|-----------|----------|----------|
| 轨迹平滑度 | 生硬 | 自然 | 可视化对比图 |
| 提前规划 | 无 | 全局 | 拥堵预判实验 |
| 社交规范 | 硬编码参数 | 数据学习 | 避让成功率对比 |
| 多样性 | 0 | 高 | 多次采样轨迹方差 |
| 烟雾场景 | 不支持 | 微调后支持 | 烟雾减速曲线 |
| 计算成本 | ~5ms | ~80ms | 速度/质量 tradeoff 图 |

---

## 四、v2.0 文件结构

```
single_gpu_evacuation/
│
├── main.py
├── config/default.yaml              # 新增 vlm 和 diffusion 配置节
│
├── perception/
│   ├── environment.py               # 不变
│   ├── nl_converter.py              # 修改: 接收 VLM 文本输入
│   └── vlm_perceiver.py             # 新增: Qwen-VL 推理封装
│
├── decision/
│   ├── agent_state.py               # 新增字段: future_trajectory
│   ├── cognitive_engine.py          # 修改: batch_size ↑, model ↑
│   ├── knowledge_base.py            # 不变
│   └── prompt_manager.py            # 修改: Prompt 新增 VLM 描述段
│
├── execution/
│   ├── batched_physics.py           # 保留 (baseline对比用)
│   ├── diffusion_policy.py          # 新增: 扩散模型轨迹生成
│   ├── diffusion_trainer.py         # 新增: 微调训练脚本
│   └── orchestrator.py              # 修改: 支持 physics/diffusion 切换
│
├── group_intel/
│   └── propagation.py               # 不变
│
├── visualization/
│   ├── renderer.py
│   └── headless_renderer.py
│
└── config/
    ├── default.yaml                 # v1.0 兼容 (4090)
    └── h800_v2.yaml                 # v2.0 H800 配置
```

---

## 五、实施计划

```
Week 1: VLM 感知接入
  - Qwen-VL-7B 部署 + Prompt 调试
  - NL Converter 扩展 (合并 VLM 输出)
  - 显存验证 (7B LLM + 7B VLM 共存)

Week 2: 扩散模型集成
  - MID backbone 加载 + 推理接口
  - 条件编码器 (LLM文本 → embedding)
  - 与社会力模型 A/B 切换

Week 3: 训练数据生成 + 微调
  - v1.0 跑 5000 条轨迹导出
  - 在 ETH/UCY 预训练权重上微调
  - 烟雾场景特化微调

Week 4: 全链路联调 + 实验
  - VLM + 7B LLM + 扩散模型 端到端
  - 消融实验: w/ VLM vs w/o VLM
  - 消融实验: Social Force vs Diffusion
  - 论文图表 + 可视化视频
```

---

## 六、预期实验结果

```
假设: 多模态 + 扩散模型 > 纯数值 + 社会力模型

实验1: VLM 感知的效果
  w/o VLM: 疏散率 42.8%, 死亡率 20.6%
  w/  VLM: 疏散率 52%+ (VLM能发现YOLO漏掉的危险)
  论证: VLM提供了"地面有水""有人被柱子挡住"等YOLO检测不到的信息

实验2: 扩散模型的效果
  Social Force: 轨迹平滑度 2.8/5 (人工评分)
  Diffusion:   轨迹平滑度 4.2/5
  论证: 扩散模型从真实数据学习,轨迹更像真人

实验3: 组合效果
  Baseline(v1.0):      疏散率 42.8%
  +7B LLM:             疏散率 48% (更好的决策)
  +7B LLM + VLM:       疏散率 54% (更全面的感知)
  +7B LLM + VLM + Diff: 疏散率 56% + 轨迹更自然
```
