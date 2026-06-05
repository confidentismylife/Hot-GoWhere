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
    DiffusionTrajectoryModel, build_scene_map
)


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
    """自定义 batch 整理, 处理变长文本.

    v2.0: 匹配 DiffusionTrajectoryModel 新 API — 分别传 txt_feat + scene_map.
    """

    def __init__(self, text_encoder, device="cuda"):
        self.text_encoder = text_encoder
        self.device = device

        # 缓存 tokenizer, 避免每次 batch 都从磁盘加载
        if text_encoder is not None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-zh-v1.5")
        else:
            self.tokenizer = None

    def __call__(self, samples: List[dict]) -> dict:
        B = len(samples)

        starts   = torch.stack([s["start"] for s in samples])
        targets  = torch.stack([s["target"] for s in samples])
        trajs    = torch.stack([s["traj"] for s in samples])

        # 场景占用图: list of [H,W,3] → [B,3,H,W]
        cond_maps = torch.stack([s["cond_map"] for s in samples])
        if cond_maps.ndim == 4 and cond_maps.shape[-1] == 3:
            cond_maps = cond_maps.permute(0, 3, 1, 2)  # [B,H,W,3] → [B,3,H,W]

        # 编码文本
        texts = [s["cond_txt"] or "向前移动" for s in samples]
        with torch.no_grad():
            txt_feats = self._encode_texts(texts)  # [B, 768]

        # 固定点mask (起点+终点)
        L = trajs.shape[1]
        fixed_mask = torch.zeros(B, L, 2)
        fixed_mask[:, 0, :] = 1.0
        fixed_mask[:, -1, :] = 1.0

        return {
            "x_0": trajs.to(self.device),
            "txt_feat": txt_feats.to(self.device),
            "scene_map": cond_maps.to(self.device),
            "fixed_mask": fixed_mask.to(self.device),
        }

    def _encode_texts(self, texts):
        if self.text_encoder is None or self.tokenizer is None:
            return torch.zeros(len(texts), 768)
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True,
                              truncation=True, max_length=256)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            out = self.text_encoder(**tokens)
        return out.last_hidden_state[:, 0, :]


# ================================================================
# 训练器
# ================================================================

class DiffusionTrainer:
    """扩散模型训练器."""

    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型 (内置 scene_encoder + 条件融合)
        self.model = DiffusionTrajectoryModel(
            d_model=config.get("d_model", 256),
            nhead=config.get("nhead", 8),
            num_layers=config.get("num_layers", 6),
            num_inference_steps=config.get("num_inference_steps", 100),
            text_dim=768,
        ).to(self.device)

        # 文本编码器 (frozen)
        try:
            from transformers import AutoModel
            self.text_encoder = AutoModel.from_pretrained(
                "BAAI/bge-base-zh-v1.5"
            ).to(self.device)
            self.text_encoder.eval()
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        except Exception:
            self.text_encoder = None

        # 优化器 (只训练扩散模型)
        self.optimizer = AdamW(
            self.model.parameters(),
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

        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            loss = self.model.training_loss(
                batch["x_0"], batch["txt_feat"],
                batch["scene_map"], batch["fixed_mask"]
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

        ade_sum = 0.0
        fde_sum = 0.0
        n = 0

        for batch in dataloader:
            B = batch["x_0"].shape[0]
            L = batch["x_0"].shape[1]

            starts  = batch["x_0"][:, 0, :]
            targets = batch["x_0"][:, -1, :]

            pred = self.model.generate(
                txt_feat=batch["txt_feat"],
                scene_map=batch["scene_map"],
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
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }, path)
        print(f"[Trainer] Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
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

    from perception.environment import DisasterSimulator, EnvironmentSnapshot
    from execution.batched_physics import BatchedPhysics

    jsonl_path = os.path.join(output_dir, "trajectories.jsonl")
    total_generated = 0
    run_id = 0

    # 使用 SimulationOrchestrator 的简化版本
    from decision.agent_state import Agent, AgentProfile, AgentDynamic, Speed
    import random

    with open(jsonl_path, 'w', encoding='utf-8') as f_out:
        while total_generated < num_trajectories:
            run_id += 1
            random.seed(run_id)
            np.random.seed(run_id)

            import yaml
            with open(config_path, 'r', encoding='utf-8') as f_cfg:
                cfg = yaml.safe_load(f_cfg)

            env_cfg = cfg["environment"]

            disaster = DisasterSimulator(
                width=env_cfg["width"], height=env_cfg["height"],
                disaster_type=env_cfg["disaster"],
                origin=tuple(env_cfg["disaster_origin"]),
                spread_rate=env_cfg["disaster_spread_rate"],
                resolution=0.5,
            )

            exits = [tuple(e) for e in env_cfg["exit_positions"]]
            obstacles = env_cfg.get("obstacles", [])

            physics = BatchedPhysics(
                width=env_cfg["width"], height=env_cfg["height"],
                obstacles=obstacles,
            )

            agents = []
            for i in range(50):
                x = random.uniform(5, env_cfg["width"] - 5)
                y = random.uniform(5, env_cfg["height"] - 5)
                chosen_exit = random.choice(exits)
                profile = AgentProfile(
                    max_speed=random.uniform(0.8, 2.0),
                    risk_aversion=random.uniform(0.2, 0.9),
                    conformity=random.uniform(0.1, 0.9),
                )
                dynamic = AgentDynamic(
                    position=np.array([x, y], dtype=np.float64),
                    stamina=random.uniform(60, 100),
                    target_exit=np.array(chosen_exit, dtype=np.float64),
                    speed_choice=Speed.WALK,
                    has_new_info=True,
                )
                agents.append(Agent(profile=profile, dynamic=dynamic))

            # 位置历史跟踪
            pos_history = {a.id: [] for a in agents}

            dt = 0.1
            for tick in range(600):
                disaster.step(dt)
                snapshot = disaster.snapshot(
                    tick, tick * dt, exits, obstacles
                )
                physics.step_all(agents, dt)

                # 跟踪位置
                for a in agents:
                    if a.dynamic.alive and not a.dynamic.evacuated:
                        pos_history[a.id].append(a.position.copy())

                # 每30 tick 采样
                if tick % 30 == 0 and tick > 30:
                    for a in agents:
                        history = pos_history[a.id]
                        if len(history) < 31:
                            continue
                        recent = np.array(history[-31:])
                        record = {
                            "start": recent[0].tolist(),
                            "target": (a.dynamic.target_exit.tolist()
                                       if a.dynamic.target_exit is not None
                                       else [50.0, 30.0]),
                            "llm_decision": a.dynamic.reasoning_text or "",
                            "scene_map": build_scene_map(
                                a.position, snapshot
                            ).tolist(),
                            "trajectory": recent.tolist(),
                        }
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_generated += 1
                        if total_generated >= num_trajectories:
                            break

                if total_generated >= num_trajectories:
                    break

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
        trainer = DiffusionTrainer(config)
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)

        dataset = TrajectoryDataset(args.data_dir, max_samples=None)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=trainer.batcher, num_workers=0
        )

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
        trainer = DiffusionTrainer(config)
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        else:
            print("[Finetune] Warning: no pretrained checkpoint, "
                  "training from scratch.")

        dataset = TrajectoryDataset(args.data_dir, max_samples=None)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size // 2, shuffle=True,
            collate_fn=trainer.batcher, num_workers=0
        )

        for epoch in range(1, args.epochs + 1):
            metrics = trainer.train_epoch(dataloader, epoch)
            print(f"  Epoch {epoch:4d}: loss={metrics['loss']:.4f}")

        trainer.save_checkpoint("./checkpoints/diffusion_finetuned.pt")

    elif args.mode == "eval":
        trainer = DiffusionTrainer(config)
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)

        dataset = TrajectoryDataset(args.data_dir, max_samples=500)
        dataloader = DataLoader(
            dataset, batch_size=32, shuffle=False,
            collate_fn=trainer.batcher, num_workers=0
        )

        results = trainer.evaluate(dataloader)
        print(f"\n  Evaluation Results:")
        print(f"  ADE: {results['ADE']:.3f} m  (average displacement error)")
        print(f"  FDE: {results['FDE']:.3f} m  (final displacement error)")
        print(f"  Samples: {results['samples']}")


if __name__ == "__main__":
    main()
