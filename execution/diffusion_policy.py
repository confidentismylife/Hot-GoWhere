"""扩散模型轨迹生成器 — MID backbone, 替代社会力模型.

Conditional Diffusion Model for pedestrian trajectory generation.
Input:  start + target + LLM decision text + scene occupancy map
Output: 31-step smooth trajectory

开发时(4090/5090): d_model=256, num_layers=6, ~0.5GB
实验时(H800):    d_model=512, num_layers=8, ~1.2GB

References:
  - DDPM (Ho et al., 2020): 扩散模型基础
  - DDIM (Song et al., 2021): 加速采样
  - MID (Gu et al., 2022): 条件扩散轨迹预测
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


# ================================================================
# 时间嵌入
# ================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Transformer 风格的位置编码, 用于扩散时间步."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] → embedding: [B, dim]"""
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# ================================================================
# 场景编码器 (轻量CNN)
# ================================================================

class SceneMapEncoder(nn.Module):
    """3通道场景占用图 → 特征向量.

    Input:  [B, 3, 64, 64]
      ch0 = 障碍物占用 (0/1)
      ch1 = 烟雾密度   (0~1)
      ch2 = 出口位置   (0/1)
    Output: [B, d_model]
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 32×32
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16×16
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 8×8
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 4×4
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, d_model]


# ================================================================
# 扩散轨迹 Transformer
# ================================================================

class DiffusionTrajectoryModel(nn.Module):
    """MID-style 条件扩散轨迹生成器.

    Architecture:
      Input → TrajProj → + TimeEmb → Transformer × N → OutputHead
                ↑                        ↑
           Start+Target约束          Cross-Attn ← Condition
    """

    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 num_timesteps: int = 1000,
                 num_inference_steps: int = 100,
                 ):
        super().__init__()

        self.d_model = d_model
        self.num_timesteps = num_timesteps
        self.num_inference_steps = num_inference_steps

        # 轨迹投影
        self.traj_proj = nn.Linear(2, d_model)

        # 时间嵌入
        self.time_embed = SinusoidalTimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

        # 条件编码 (这里只做投影, 实际编码由外部完成)
        self.cond_proj = nn.Linear(d_model * 2, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出头
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

        # 噪声调度
        self.register_buffer(
            'betas', self._linear_beta_schedule(num_timesteps)
        )
        self.register_buffer(
            'alphas', 1.0 - self.betas
        )
        self.register_buffer(
            'alphas_cumprod', torch.cumprod(self.alphas, dim=0)
        )

    def _linear_beta_schedule(self, T: int) -> torch.Tensor:
        """线性调度: β₁=0.0001 → β_T=0.02"""
        return torch.linspace(0.0001, 0.02, T)

    # ============================================================
    # 前向 (训练)
    # ============================================================

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor,
                fixed_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """预测噪声.

        Args:
            x_t:  [B, L, 2] 加噪轨迹
            t:    [B] 扩散时间步
            cond: [B, d_model*2] 条件特征 (文本+场景)
            fixed_mask: [B, L, 2] 固定点mask (起点+终点)

        Returns:
            epsilon_pred: [B, L, 2] 预测的噪声
        """
        B, L, _ = x_t.shape

        # 轨迹编码
        h = self.traj_proj(x_t)  # [B, L, d_model]

        # 时间编码
        t_emb = self.time_embed(t)     # [B, d_model]
        t_feat = self.time_mlp(t_emb)  # [B, d_model]
        h = h + t_feat.unsqueeze(1)

        # 条件注入 (通过 Add, 简化版Cross-Attn)
        cond_feat = self.cond_proj(cond)  # [B, d_model]
        h = h + cond_feat.unsqueeze(1)

        # Transformer
        h = self.transformer(h)  # [B, L, d_model]

        # 输出
        eps = self.out_proj(h)  # [B, L, 2]

        # 固定点约束: 起点和终点噪声为0
        if fixed_mask is not None:
            eps = eps * (1.0 - fixed_mask)

        return eps

    # ============================================================
    # 训练 loss
    # ============================================================

    def training_loss(self, x_0: torch.Tensor, cond: torch.Tensor,
                      fixed_mask: Optional[torch.Tensor] = None
                      ) -> torch.Tensor:
        """计算扩散训练 loss.

        Args:
            x_0:  [B, L, 2] 真实轨迹
            cond: [B, d_model*2] 条件特征

        Returns:
            loss: scalar MSE loss
        """
        B = x_0.shape[0]

        # 采样时间步
        t = torch.randint(0, self.num_timesteps, (B,), device=x_0.device)

        # 采样噪声
        epsilon = torch.randn_like(x_0)

        # 前向加噪
        alpha_cumprod_t = self.alphas_cumprod[t]  # [B]
        alpha_cumprod_t = alpha_cumprod_t.view(B, 1, 1)
        x_t = torch.sqrt(alpha_cumprod_t) * x_0 + \
              torch.sqrt(1.0 - alpha_cumprod_t) * epsilon

        # 预测噪声
        epsilon_pred = self.forward(x_t, t, cond, fixed_mask)

        # MSE loss
        loss = nn.functional.mse_loss(epsilon_pred, epsilon)
        return loss

    # ============================================================
    # 推理 (DDIM 采样)
    # ============================================================

    @torch.no_grad()
    def generate(self, cond: torch.Tensor,
                 start: torch.Tensor, target: torch.Tensor,
                 num_steps: int = 31,
                 num_inference_steps: Optional[int] = None,
                 ) -> torch.Tensor:
        """从噪声生成轨迹 (DDIM 采样).

        Args:
            cond:     [B, d_model*2] 条件特征
            start:    [B, 2] 起点坐标
            target:   [B, 2] 目标坐标
            num_steps: int 轨迹长度
            num_inference_steps: int DDIM采样步数 (默认100, 越小越快)

        Returns:
            trajectory: [B, num_steps, 2]
        """
        B = cond.shape[0]
        num_inf = num_inference_steps or self.num_inference_steps

        # 固定点 mask
        fixed_mask = torch.zeros(B, num_steps, 2, device=cond.device)
        fixed_mask[:, 0, :] = 1.0
        fixed_mask[:, -1, :] = 1.0

        # 初始噪声
        x_t = torch.randn(B, num_steps, 2, device=cond.device)
        x_t[:, 0, :] = start
        x_t[:, -1, :] = target

        # DDIM 时间步 (均匀间隔)
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_inf, dtype=torch.long, device=cond.device
        )

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            t_batch = torch.full((B,), t, device=cond.device, dtype=torch.long)

            # 预测噪声
            eps_pred = self.forward(x_t, t_batch, cond, fixed_mask)

            # DDIM 更新
            alpha_t = self.alphas_cumprod[t]
            alpha_next = self.alphas_cumprod[t_next]

            # x0 预测
            x0_pred = (x_t - torch.sqrt(1.0 - alpha_t) * eps_pred) / \
                      torch.sqrt(alpha_t)

            # 方向指向 x0
            dir_xt = torch.sqrt(1.0 - alpha_next) * eps_pred

            # 更新
            x_t = torch.sqrt(alpha_next) * x0_pred + dir_xt

            # 固定点约束
            x_t = x_t * (1.0 - fixed_mask) + \
                  torch.cat([start.unsqueeze(1),
                             torch.zeros(B, num_steps - 2, 2, device=cond.device),
                             target.unsqueeze(1)], dim=1) * fixed_mask

        return x_t  # [B, num_steps, 2]


# ================================================================
# 便捷包装类 — 对接 orchestrator
# ================================================================

class DiffusionPolicy:
    """对接 SimulationOrchestrator 的扩散模型包装器.

    Usage:
        policy = DiffusionPolicy(config)
        policy.initialize()

        for agent in agents:
            if agent.needs_new_traj():
                traj = policy.generate_one(
                    agent.position, agent.dynamic.target_exit,
                    agent.dynamic.reasoning_text,
                    build_scene_map(agent, env)
                )
                agent.future_trajectory = traj
    """

    def __init__(self, config: dict):
        cfg = config.get("diffusion", {})

        self.d_model = cfg.get("d_model", 256)
        self.nhead = cfg.get("nhead", 8)
        self.num_layers = cfg.get("num_layers", 6)
        self.checkpoint = cfg.get("checkpoint", None)
        self.num_inference_steps = cfg.get("num_inference_steps", 100)

        self.model: Optional[DiffusionTrajectoryModel] = None
        self.text_encoder = None
        self._initialized = False

    def initialize(self):
        """初始化模型. 如果提供了checkpoint则加载权重."""
        if self._initialized:
            return

        self.model = DiffusionTrajectoryModel(
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            num_inference_steps=self.num_inference_steps,
        )

        if self.checkpoint:
            print(f"[Diffusion] Loading checkpoint: {self.checkpoint}")
            state = torch.load(self.checkpoint, map_location="cpu")
            self.model.load_state_dict(state)
        else:
            print("[Diffusion] No checkpoint provided. "
                  "Using random weights (trajectories will be noise).")

        self.model.eval()
        self.model.cuda()

        # 文本编码器
        try:
            from transformers import AutoModel, AutoTokenizer
            self.text_encoder = AutoModel.from_pretrained(
                "BAAI/bge-base-zh-v1.5"
            ).cuda()
            self.tokenizer = AutoTokenizer.from_pretrained(
                "BAAI/bge-base-zh-v1.5"
            )
            self._text_dim = 768
        except Exception:
            print("[Diffusion] BGE not available, using random text embeddings.")
            self.text_encoder = None
            self.tokenizer = None
            self._text_dim = 768

        self._initialized = True
        print(f"[Diffusion] Model ready. "
              f"d_model={self.d_model}, layers={self.num_layers}")

    def encode_text(self, text: str) -> torch.Tensor:
        """LLM决策文本 → 特征向量."""
        if self.text_encoder is None or not text:
            return torch.zeros(1, self._text_dim).cuda()

        tokens = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        ).to(self.model.device)

        with torch.no_grad():
            output = self.text_encoder(**tokens)
            # CLS token 或 mean pooling
            emb = output.last_hidden_state[:, 0, :]  # [1, 768]
        return emb

    def encode_scene(self, scene_map: np.ndarray) -> torch.Tensor:
        """场景占用图 → 特征向量."""
        if scene_map.ndim == 3:
            scene_map = scene_map[np.newaxis, ...]
        x = torch.tensor(scene_map, dtype=torch.float32).cuda()
        return self.model.scene_encoder(x)  # 使用模型内置编码器

    @torch.no_grad()
    def generate_one(self, start: np.ndarray, target: np.ndarray,
                     llm_reasoning: str, scene_map: np.ndarray,
                     num_steps: int = 31) -> np.ndarray:
        """为一个Agent生成轨迹.

        Args:
            start:   [2] 起点
            target:  [2] 出口
            llm_reasoning: str LLM决策文本
            scene_map: [64, 64, 3] 场景占用图
            num_steps: int 轨迹步数

        Returns:
            [num_steps, 2] numpy 轨迹
        """
        if self.model is None:
            # Fallback: 线性插值
            return np.linspace(start, target, num_steps)

        self.model.eval()

        # 编码条件
        txt_feat = self.encode_text(llm_reasoning)  # [1, 768]
        scn_feat = self.encode_scene(scene_map)      # [1, d_model]
        cond = torch.cat([txt_feat, scn_feat], dim=-1)  # [1, 768+d_model]

        # 对齐维度
        if cond.shape[-1] != self.d_model * 2:
            cond = torch.nn.functional.pad(
                cond, (0, self.d_model * 2 - cond.shape[-1])
            )

        start_t = torch.tensor(start, dtype=torch.float32).unsqueeze(0).cuda()
        target_t = torch.tensor(target, dtype=torch.float32).unsqueeze(0).cuda()

        # 生成
        traj = self.model.generate(
            cond=cond,
            start=start_t,
            target=target_t,
            num_steps=num_steps,
        )

        return traj[0].cpu().numpy()  # [31, 2]

    @torch.no_grad()
    def generate_batch(self, starts, targets, reasonings, scene_maps,
                       num_steps=31) -> np.ndarray:
        """批量生成 (更高效)."""
        B = len(starts)

        # 编码所有条件
        txt_feats = torch.cat([
            self.encode_text(r) for r in reasonings
        ], dim=0)  # [B, 768]

        scn_feats = torch.cat([
            self.encode_scene(m) for m in scene_maps
        ], dim=0)  # [B, d_model]

        cond = torch.cat([txt_feats, scn_feats], dim=-1)
        if cond.shape[-1] != self.d_model * 2:
            cond = torch.nn.functional.pad(
                cond, (0, self.d_model * 2 - cond.shape[-1])
            )

        start_t = torch.tensor(np.array(starts), dtype=torch.float32).cuda()
        target_t = torch.tensor(np.array(targets), dtype=torch.float32).cuda()

        trajs = self.model.generate(
            cond=cond, start=start_t, target=target_t, num_steps=num_steps
        )

        return trajs.cpu().numpy()  # [B, 31, 2]

    def shutdown(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.text_encoder is not None:
            del self.text_encoder
            self.text_encoder = None
        torch.cuda.empty_cache()


# ================================================================
# 辅助: 构建场景占用图
# ================================================================

def build_scene_map(agent_position, env_snapshot, resolution=64):
    """从环境快照构建 [H, W, 3] 场景占用图.

    Channel 0: 障碍物 (0/1)
    Channel 1: 烟雾 (0~1)
    Channel 2: 出口 (0/1)
    """
    H, W = resolution, resolution
    scene = np.zeros((H, W, 3), dtype=np.float32)

    # Channel 0: 障碍物
    for obs in env_snapshot.obstacles:
        cx, cy = obs["center"]
        r = obs["radius"]
        # 粗糙栅格化
        for i in range(H):
            for j in range(W):
                wx = j / W * env_snapshot.width
                wy = i / H * env_snapshot.height
                if (wx - cx)**2 + (wy - cy)**2 < r**2:
                    scene[i, j, 0] = 1.0

    # Channel 1: 烟雾 (从grid降采样)
    grid = env_snapshot.grid
    for i in range(H):
        for j in range(W):
            gr = int(i / H * grid.shape[0])
            gc = int(j / W * grid.shape[1])
            scene[i, j, 1] = grid[gr, gc, 0]

    # Channel 2: 出口
    for ex, ey in env_snapshot.exits:
        xi = int(ex / env_snapshot.width * W)
        yi = int(ey / env_snapshot.height * H)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = xi + dx, yi + dy
                if 0 <= nx < W and 0 <= ny < H:
                    scene[ny, nx, 2] = 1.0

    return scene
