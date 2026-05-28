"""扩散模型训练脚本 — 预训练(ETH/UCY) + 微调(疏散仿真数据).

Usage:
    # 生成训练数据 (用v1.0仿真)
    python execution/diffusion_trainer.py --mode generate --num_trajectories 10000

    # 预训练 (ETH/UCY)
    python execution/diffusion_trainer.py --mode pretrain --epochs 200

    # 微调 (疏散数据)
    python execution/diffusion_trainer.py --mode finetune --epochs 100

    # 评测
    python execution/diffusion_trainer.py --mode eval --checkpoint path/to/model.pt
"""

import os
import sys
import math
import argparse
import json
import time
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from execution.diffusion_policy import (
    DiffusionTrajectoryModel, SceneMapEncoder, build_scene_map
)
from execution.orchestrator import SimulationOrchestrator


# ================================================================
# 数据集
# ================================================================

class TrajectoryDataset(Dataset):
    """疏散轨迹数据集.

    每条样本:
      - start:        [2] 起点
      - target:       [2] 出口
      - llm_decision: str LLM决策文本
      - scene_map:    [3, 64, 64] 场景占用图
      - trajectory:   [L, 2] 真实轨迹 (用于监督)
    """

    def __init__(self, data_dir: str, max_samples: Optional[int] = None):
        self.data = []
        self.data_dir = data_dir

        # 加载 JSON Lines
        jsonl_path = os.path.join(data_dir, "trajectories.jsonl")
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.data.append(json.loads(line))
                    if max_samples and len(self.data) >= max_samples:
                        break

        print(f"[Dataset] Loaded {len(self.data)} samples from {data_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        return {
            "start":    torch.tensor(record["start"], dtype=torch.float32),
            "target":   torch.tensor(record["target"], dtype=torch.float32),
            "cond_txt": record.get("llm_decision", ""),
            "cond_map": torch.tensor(record["scene_map"], dtype=torch.float32),
            "traj":     torch.tensor(record["trajectory"], dtype=torch.float32),
        }


class TrajectoryBatcher:
    """自定义 batch 整理, 处理变长文本."""

    def __init__(self, text_encoder, device="cuda"):
        self.text_encoder = text_encoder
        self.device = device

    def __call__(self, samples: List[dict]) -> dict:
        B = len(samples)

        starts   = torch.stack([s["start"] for s in samples])
        targets  = torch.stack([s["target"] for s in samples])
        cond_map = torch.stack([s["cond_map"] for s in samples])
        trajs    = torch.stack([s["traj"] for s in samples])

        # 编码文本
        texts = [s["cond_txt"] for s in samples]
        with torch.no_grad():
            txt_feats = self._encode_texts(texts)  # [B, 768]

        # 场景编码 (用内置编码器)
        scn_feats = self._encode_scenes(cond_map)  # [B, d_model]

        # 拼接条件
        cond = torch.cat([txt_feats, scn_feats], dim=-1)

        # 固定点mask (起点+终点)
        L = trajs.shape[1]
        fixed_mask = torch.zeros(B, L, 2)
        fixed_mask[:, 0, :] = 1.0
        fixed_mask[:, -1, :] = 1.0

        return {
            "x_0": trajs.to(self.device),
            "cond": cond.to(self.device),
            "fixed_mask": fixed_mask.to(self.device),
        }

    def _encode_texts(self, texts):
        if self.text_encoder is None:
            return torch.zeros(len(texts), 768)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-zh-v1.5")
        tokens = tokenizer(texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=256)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            out = self.text_encoder(**tokens)
        return out.last_hidden_state[:, 0, :]

    def _encode_scenes(self, cond_map):
        # 轻量降采样
        B = cond_map.shape[0]
        pooled = nn.functional.adaptive_avg_pool2d(
            cond_map.permute(0, 3, 1, 2), (8, 8)
        )
        return pooled.reshape(B, -1).to(self.device)


# ================================================================
# 训练器
# ================================================================

class DiffusionTrainer:
    """扩散模型训练器."""

    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型
        self.model = DiffusionTrajectoryModel(
            d_model=config.get("d_model", 256),
            nhead=config.get("nhead", 8),
            num_layers=config.get("num_layers", 6),
            num_inference_steps=config.get("num_inference_steps", 100),
        ).to(self.device)

        self.scene_encoder = SceneMapEncoder(
            d_model=config.get("d_model", 256)
        ).to(self.device)

        # 文本编码器
        try:
            from transformers import AutoModel
            self.text_encoder = AutoModel.from_pretrained(
                "BAAI/bge-base-zh-v1.5"
            ).to(self.device)
            self.text_encoder.eval()
        except Exception:
            self.text_encoder = None

        # 优化器
        self.optimizer = AdamW(
            list(self.model.parameters()) + list(self.scene_encoder.parameters()),
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 1e-5),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("epochs", 100),
        )

        self.batcher = TrajectoryBatcher(self.text_encoder, self.device)

    def train_epoch(self, dataloader, epoch: int) -> dict:
        self.model.train()
        self.scene_encoder.train()

        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            # batch 已经在 batcher 中处理好了
            loss = self.model.training_loss(
                batch["x_0"], batch["cond"], batch["fixed_mask"]
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        self.scheduler.step()
        avg_loss = total_loss / max(1, n_batches)

        return {"epoch": epoch, "loss": avg_loss, "lr": self.scheduler.get_last_lr()[0]}

    @torch.no_grad()
    def evaluate(self, dataloader) -> dict:
        """评估: ADE (Average Displacement Error) 和 FDE (Final Displacement Error)."""
        self.model.eval()
        self.scene_encoder.eval()

        ade_sum = 0.0
        fde_sum = 0.0
        n = 0

        for batch in dataloader:
            # 用 inference 模式生成轨迹
            B = batch["x_0"].shape[0]
            L = batch["x_0"].shape[1]

            starts  = batch["x_0"][:, 0, :]
            targets = batch["x_0"][:, -1, :]

            pred = self.model.generate(
                cond=batch["cond"],
                start=starts,
                target=targets,
                num_steps=L,
            )  # [B, L, 2]

            gt = batch["x_0"]  # [B, L, 2]

            # ADE: 所有点平均误差
            ade = torch.sqrt(((pred - gt) ** 2).sum(dim=-1)).mean()
            ade_sum += ade.item() * B

            # FDE: 终点误差
            fde = torch.sqrt(((pred[:, -1] - gt[:, -1]) ** 2).sum(dim=-1)).mean()
            fde_sum += fde.item() * B

            n += B

        return {"ADE": ade_sum / n, "FDE": fde_sum / n, "samples": n}

    def save_checkpoint(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "scene_encoder": self.scene_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }, path)
        print(f"[Trainer] Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.scene_encoder.load_state_dict(ckpt["scene_encoder"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
        print(f"[Trainer] Checkpoint loaded: {path}")


# ================================================================
# 训练数据生成 (用 v1.0 仿真)
# ================================================================

def generate_training_data(config_path: str, output_dir: str,
                           num_trajectories: int = 10000):
    """运行 v1.0 仿真, 导出轨迹作为训练数据."""
    print(f"[DataGen] Generating {num_trajectories} trajectories...")
    os.makedirs(output_dir, exist_ok=True)

    orch = SimulationOrchestrator(config_path)
    orch.num_agents = 50  # 少量Agent, 多次跑
    total_generated = 0
    run_id = 0

    jsonl_path = os.path.join(output_dir, "trajectories.jsonl")
    with open(jsonl_path, 'w', encoding='utf-8') as f_out:

        while total_generated < num_trajectories:
            orch.tick = 0
            orch.sim_time = 0.0
            orch.generate_agents()

            # 跑一次短仿真 (60秒)
            while orch.tick < 600 and (total_generated < num_trajectories):
                orch.disaster.step(orch.dt)
                orch.spatial_grid.rebuild(orch.agents)
                orch.physics.step_all(orch.agents, orch.dt)

                # 每30 tick 采样一次轨迹片段
                if orch.tick % 30 == 0:
                    for agent in orch.agents:
                        if not agent.dynamic.alive or agent.dynamic.evacuated:
                            continue
                        # 取最近3秒的轨迹
                        if hasattr(agent, 'position_history') and \
                           len(agent.position_history) >= 31:
                            record = {
                                "start": agent.position_history[-31].tolist(),
                                "target": agent.dynamic.target_exit.tolist()
                                          if agent.dynamic.target_exit
                                          is not None else [50, 30],
                                "llm_decision": agent.dynamic.reasoning_text or "",
                                "scene_map": build_scene_map(
                                    agent.position,
                                    orch.disaster.snapshot(
                                        orch.tick, orch.sim_time,
                                        orch.exits, orch.obstacles
                                    )
                                ).tolist(),
                                "trajectory": agent.position_history[-31:],
                            }
                            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                            total_generated += 1
                            if total_generated >= num_trajectories:
                                break

                orch.tick += 1
                orch.sim_time += orch.dt

            run_id += 1
            print(f"  Run {run_id}: {total_generated}/{num_trajectories} samples")

    print(f"[DataGen] Done. {total_generated} trajectories → {jsonl_path}")


# ================================================================
# 主入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="扩散模型训练")
    parser.add_argument("--mode", choices=["generate", "pretrain", "finetune", "eval"],
                       default="pretrain")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--data_dir", default="./data/training_trajs")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_trajectories", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)

    args = parser.parse_args()

    config = {
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
    }

    if args.mode == "generate":
        generate_training_data(args.config, args.data_dir, args.num_trajectories)

    elif args.mode == "pretrain":
        dataset = TrajectoryDataset(args.data_dir, max_samples=None)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=TrajectoryBatcher(None), num_workers=4
        )
        trainer = DiffusionTrainer(config)
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)

        for epoch in range(1, args.epochs + 1):
            metrics = trainer.train_epoch(dataloader, epoch)
            print(f"  Epoch {epoch:4d}: loss={metrics['loss']:.4f}  "
                  f"lr={metrics['lr']:.2e}")

            if epoch % 20 == 0:
                trainer.save_checkpoint(
                    f"./checkpoints/diffusion_epoch{epoch:04d}.pt"
                )

        trainer.save_checkpoint("./checkpoints/diffusion_final.pt")

    elif args.mode == "finetune":
        # 加载预训练权重 → 微调
        dataset = TrajectoryDataset(args.data_dir, max_samples=None)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size // 2, shuffle=True,
            collate_fn=TrajectoryBatcher(None), num_workers=2
        )
        trainer = DiffusionTrainer(config)
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        else:
            print("[Finetune] Warning: no pretrained checkpoint, "
                  "training from scratch.")

        for epoch in range(1, args.epochs + 1):
            metrics = trainer.train_epoch(dataloader, epoch)
            print(f"  Epoch {epoch:4d}: loss={metrics['loss']:.4f}")

        trainer.save_checkpoint("./checkpoints/diffusion_finetuned.pt")

    elif args.mode == "eval":
        dataset = TrajectoryDataset(args.data_dir, max_samples=500)
        dataloader = DataLoader(
            dataset, batch_size=32, shuffle=False,
            collate_fn=TrajectoryBatcher(None)
        )
        trainer = DiffusionTrainer(config)
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)

        results = trainer.evaluate(dataloader)
        print(f"\n  Evaluation Results:")
        print(f"  ADE: {results['ADE']:.3f} m  (average displacement error)")
        print(f"  FDE: {results['FDE']:.3f} m  (final displacement error)")
        print(f"  Samples: {results['samples']}")


if __name__ == "__main__":
    main()
