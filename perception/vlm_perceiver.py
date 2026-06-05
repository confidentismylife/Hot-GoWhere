"""VLM 视觉感知器 — Qwen-VL 画面语义理解.

设计要点:
1. 不每帧调用—隔30 tick或环境剧变时触发
2. 结果全局缓存—同一帧的所有Agent共享
3. 失败容错—VLM不可用时静默降级,不影响主流程
4. 显存友好—INT4量化, 仅5GB

开发时(4090/5090): Qwen2.5-VL-7B-INT4
实验时(H800):    Qwen2.5-VL-7B-FP16
"""

import time
import numpy as np
from typing import Optional


VLM_SYSTEM_PROMPT = """你是一个地铁站监控系统的视觉分析模块。
请仔细观察画面,以客观、精确的语言描述以下内容:

1. 烟雾分布: 位置、浓度、扩散方向
2. 火焰情况: 位置、大小、是否在蔓延
3. 人群状态: 分布、密度、移动方向、是否有异常行为(摔倒/推挤/逆行)
4. 环境变化: 积水、掉落物、应急指示灯、建筑损坏
5. 出口可见性: 各出口是否可见、是否被遮挡或封锁

规则:
- 只描述你实际看到的,不要推断或假设
- 用具体方位词(东北角/左侧/距镜头约10米处)而非模糊词汇
- 如果某个类别没有异常,写"无异常"即可
- 总字数不超过150字"""

VLM_USER_PROMPT = "分析当前监控画面,按类别描述所见情况。"


class VLMPerceiver:
    """Qwen-VL 视觉感知器.

    Usage:
        vlm = VLMPerceiver("Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
        vlm.initialize()

        for tick in range(total_ticks):
            frame = get_cctv_frame()  # (480, 640, 3) numpy
            desc = vlm.perceive(frame, tick, env_snapshot)
            if desc:
                agents_share_vlm_output(desc)
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
                 call_interval: int = 30):
        self.model_name = model_name
        self.call_interval = call_interval

        # 延迟加载
        self.model = None
        self.processor = None
        self._initialized = False

        # 缓存
        self.last_description: str = ""
        self.last_call_tick: int = -999999
        self._call_count: int = 0

        # 统计
        self.total_calls: int = 0
        self.total_time: float = 0.0

    # ================================================================
    # 初始化
    # ================================================================

    def initialize(self):
        """加载 VLM 模型到显存."""
        if self._initialized:
            return

        print(f"[VLM] Loading {self.model_name} ...")
        t0 = time.time()

        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )

            elapsed = time.time() - t0
            self._initialized = True
            print(f"[VLM] Loaded in {elapsed:.1f}s. Ready.")

        except ImportError:
            print("[VLM] Qwen-VL dependencies not installed. "
                  "VLM perception disabled. "
                  "(pip install qwen-vl-utils transformers>=4.45)")
            self.model = None
            self.processor = None
        except Exception as e:
            print(f"[VLM] Failed to load model: {e}. "
                  "Running without visual perception.")
            self.model = None
            self.processor = None

    # ================================================================
    # 主接口
    # ================================================================

    def perceive(self, frame: np.ndarray, tick: int,
                 env_snapshot=None) -> str:
        """调用 VLM 分析画面. 返回场景描述文本.

        Args:
            frame: (H, W, 3) numpy array, RGB格式
            tick:  当前模拟tick
            env_snapshot: 用于判断是否需要重新调用

        Returns:
            场景描述文本. 如果不需要重新调用,返回缓存.
            如果VLM未加载或出错,返回空字符串.
        """
        # 模型未加载 → 静默降级
        if self.model is None:
            return ""

        # 判断是否需要重新调用
        if not self._should_call(tick, env_snapshot):
            return self.last_description

        # 调用 VLM
        try:
            result = self._call_vlm(frame)
            self.last_description = result
            self.last_call_tick = tick
            self.total_calls += 1
            return result

        except Exception as e:
            print(f"[VLM] Inference error (tick {tick}): {e}")
            return self.last_description  # 降级到缓存

    # ================================================================
    # 内部方法
    # ================================================================

    def _should_call(self, tick: int, env_snapshot=None) -> bool:
        """判断是否需要重新调用 VLM (当前仅定时触发)."""
        return tick - self.last_call_tick >= self.call_interval

    def _call_vlm(self, frame: np.ndarray) -> str:
        """执行一次 VLM 推理."""
        t0 = time.time()

        from PIL import Image

        # numpy → PIL
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            frame = (frame * 255).astype(np.uint8)
        image = Image.fromarray(frame).convert("RGB")

        # 构建消息
        messages = [
            {"role": "system", "content": VLM_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": VLM_USER_PROMPT},
            ]},
        ]

        # 推理
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs, max_new_tokens=150, temperature=0.2, do_sample=False
        )
        # 只取生成的部分 (去掉输入)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        elapsed = (time.time() - t0) * 1000
        self.total_time += elapsed
        self._call_count += 1

        return response

    # ================================================================
    # 统计
    # ================================================================

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "avg_time_ms": self.total_time / max(1, self.total_calls),
            "last_description": self.last_description[:80] + "...",
        }

    def shutdown(self):
        """释放 GPU 显存."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            import torch
            torch.cuda.empty_cache()
            print("[VLM] Shutdown complete.")


class MockVLMPerceiver:
    """VLM mock for development/testing — generates NL from env state.

    Same interface as VLMPerceiver but uses numeric env data to produce
    realistic Chinese scene descriptions. No GPU required.

    Usage:
        vlm = MockVLMPerceiver(call_interval=30)
        desc = vlm.perceive(None, tick, env_snapshot)
    """

    def __init__(self, call_interval: int = 30):
        self.call_interval = call_interval
        self.last_description: str = ""
        self.last_call_tick: int = -999999
        self.total_calls: int = 0
        self.total_time: float = 0.0

    def initialize(self):
        print("[MockVLM] Using synthetic scene descriptions (no GPU).")

    def perceive(self, frame, tick: int, env_snapshot=None) -> str:
        if tick - self.last_call_tick < self.call_interval:
            return self.last_description

        if env_snapshot is None:
            return self.last_description or "监控画面暂无数据。"

        import time
        t0 = time.time()

        desc = self._synthesize(env_snapshot)
        self.last_description = desc
        self.last_call_tick = tick
        self.total_calls += 1
        self.total_time += (time.time() - t0) * 1000
        return desc

    def _synthesize(self, env) -> str:
        grid = env.grid
        smoke_field = grid[:, :, 0]
        fire_field = grid[:, :, 3]

        smoke_max = float(smoke_field.max())
        smoke_mean = float(smoke_field.mean())
        fire_pixels = int((fire_field > 0.5).sum())

        # Smoke description
        if smoke_max < 0.1:
            smoke_desc = "现场无明显烟雾，视野清晰。"
        elif smoke_max < 0.3:
            smoke_desc = "轻微烟雾出现在局部区域，能见度良好。"
        elif smoke_max < 0.6:
            region = self._smoke_region(smoke_field, env)
            smoke_desc = f"烟雾浓度中等，{region}方向烟雾较明显，建议低姿前进。"
        else:
            region = self._smoke_region(smoke_field, env)
            smoke_desc = f"浓烟弥漫，{region}区域能见度严重下降，部分区域伸手不见五指。"

        # Fire description
        if fire_pixels == 0:
            fire_desc = "未见明火。"
        elif fire_pixels < 20:
            fire_desc = "局部出现明火，火焰范围较小。"
        elif fire_pixels < 80:
            fire_desc = "火势正在扩大，火焰高度约1-2米，伴有大量黑烟。"
        else:
            fire_desc = "大面积燃烧，火势猛烈，结构存在坍塌风险。"

        # Exit status
        exit_parts = []
        for i, ep in enumerate(env.exits):
            e_smoke = env.smoke_at(np.array(ep, dtype=np.float64))
            if e_smoke < 0.3:
                exit_parts.append(f"出口{i+1}畅通")
            elif e_smoke < 0.6:
                exit_parts.append(f"出口{i+1}有烟雾但可通行")
            else:
                exit_parts.append(f"出口{i+1}被浓烟封锁")
        exit_desc = "；".join(exit_parts)

        # Structural
        struct_field = grid[:, :, 2]
        struct_min = float(struct_field.min())
        if struct_min > 0.8:
            struct_desc = "建筑结构完好。"
        elif struct_min > 0.5:
            struct_desc = "部分墙体出现裂缝，天花板有碎片掉落。"
        else:
            struct_desc = "建筑结构严重受损，部分区域可能坍塌。"

        return (
            f"烟雾分布: {smoke_desc} "
            f"火情: {fire_desc} "
            f"出口状态: {exit_desc}。"
            f"建筑: {struct_desc}"
        )

    @staticmethod
    def _smoke_region(smoke_field, env) -> str:
        """Determine which region has heaviest smoke."""
        h, w = smoke_field.shape
        quadrants = {
            "东北": smoke_field[:h//2, w//2:].mean(),
            "西北": smoke_field[:h//2, :w//2].mean(),
            "东南": smoke_field[h//2:, w//2:].mean(),
            "西南": smoke_field[h//2:, :w//2].mean(),
        }
        return max(quadrants, key=quadrants.get)

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "avg_time_ms": self.total_time / max(1, self.total_calls),
            "last_description": self.last_description[:80] + "...",
            "mode": "mock",
        }

    def shutdown(self):
        print("[MockVLM] Shutdown.")
