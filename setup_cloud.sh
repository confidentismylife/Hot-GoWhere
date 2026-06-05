#!/bin/bash
# ================================================================
# setup_cloud.sh — GPU 云平台一键部署脚本
# ================================================================
# 适用平台: AutoDL, 恒源云, Vast.ai, 矩池云, Colab(付费GPU)
#
# Usage:
#   bash setup_cloud.sh              # 安装依赖 + 下载模型 + 启动Web
#   bash setup_cloud.sh --install    # 仅安装依赖
#   bash setup_cloud.sh --start      # 仅启动服务
#   bash setup_cloud.sh --port 9090  # 指定端口
# ================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 配置 ----
PORT=8080
NUM_AGENTS=800
FRAME_INTERVAL=10
CONFIG_FILE="config/default.yaml"
INSTALL_ONLY=false
START_ONLY=false

# ---- 解析参数 ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --install) INSTALL_ONLY=true; shift ;;
        --start)   START_ONLY=true; shift ;;
        --port)    PORT="$2"; shift 2 ;;
        --agents)  NUM_AGENTS="$2"; shift 2 ;;
        --config)  CONFIG_FILE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ================================================================
# 0. 平台检测 & 基本信息
# ================================================================

echo "================================================================"
echo "  🔥 LLM Evacuation Simulation — GPU Cloud Setup"
echo "================================================================"
echo "  Platform: $(uname -a)"
echo "  Python:   $(python3 --version 2>/dev/null || python --version)"
echo "  CUDA:     $(nvcc --version 2>/dev/null | grep release || echo 'checking...')"
echo "  GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

# 检测 VRAM
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
echo "  GPU Memory: ${VRAM}MB"
if [ "$VRAM" -lt 8000 ] 2>/dev/null; then
    echo "  ⚠ WARNING: VRAM < 8GB. 仿真可能失败, 请用更小的模型."
    echo "  → 建议: 修改 config/default.yaml:"
    echo '    llm.model: "Qwen/Qwen2.5-1.5B-Instruct"  # 更小的模型'
fi
echo ""

# ================================================================
# 1. 安装系统依赖
# ================================================================

if [ "$START_ONLY" = false ]; then

echo "[1/5] 安装系统依赖..."

# ffmpeg (用于生成MP4视频)
if ! command -v ffmpeg &>/dev/null; then
    echo "  → 安装 ffmpeg..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg
    elif command -v yum &>/dev/null; then
        sudo yum install -y ffmpeg
    elif command -v conda &>/dev/null; then
        conda install -y -c conda-forge ffmpeg
    else
        echo "  ⚠ 无法自动安装 ffmpeg, MP4 下载功能将不可用."
    fi
else
    echo "  ✓ ffmpeg 已安装"
fi

# ================================================================
# 2. 安装 Python 依赖
# ================================================================

echo "[2/5] 安装 Python 依赖..."

# 升级 pip
python3 -m pip install --upgrade pip -q 2>/dev/null || python -m pip install --upgrade pip -q

# 安装核心依赖 (分批, 避免大包超时)
echo "  → 核心依赖 (numpy, pyyaml, torch)..."
python3 -m pip install numpy>=1.26.0 pyyaml>=6.0 tqdm>=4.66.0 orjson>=3.10.0 -q 2>/dev/null ||
python -m pip install numpy>=1.26.0 pyyaml>=6.0 tqdm>=4.66.0 orjson>=3.10.0

# PyTorch (如果还没装)
python3 -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')" 2>/dev/null || {
    echo "  → 安装 PyTorch (CUDA 12.1)..."
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 ||
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
}

# LLM / VLM 推理
echo "  → LLM 推理 (vllm, transformers)..."
python3 -m pip install vllm>=0.6.0 transformers>=4.44.0 autoawq>=0.2.0 -q 2>/dev/null ||
python -m pip install vllm>=0.6.0 transformers>=4.44.0 autoawq>=0.2.0

# 知识库
echo "  → 知识库 (chromadb, embeddings)..."
python3 -m pip install chromadb>=0.5.0 sentence-transformers>=3.0.0 -q 2>/dev/null ||
python -m pip install chromadb>=0.5.0 sentence-transformers>=3.0.0

# 可视化
echo "  → 可视化 (matplotlib, pygame, flask)..."
python3 -m pip install matplotlib>=3.9.0 pygame>=2.6.0 flask>=3.0.0 Pillow imageio -q 2>/dev/null ||
python -m pip install matplotlib>=3.9.0 pygame>=2.6.0 flask>=3.0.0 Pillow imageio

# YOLO (人员检测)
echo "  → YOLO 检测..."
python3 -m pip install ultralytics>=8.3.0 -q 2>/dev/null ||
python -m pip install ultralytics>=8.3.0

# VLM
echo "  → VLM 工具..."
python3 -m pip install qwen-vl-utils>=0.0.8 -q 2>/dev/null ||
python -m pip install qwen-vl-utils>=0.0.8

echo "  ✓ Python 依赖安装完成"

# ================================================================
# 3. 下载模型文件
# ================================================================

echo "[3/5] 下载模型文件..."

# YOLO 模型 (自动下载, ~6MB)
echo "  → YOLOv8n (人员检测, ~6MB)..."
python3 -c "
from ultralytics import YOLO
import os
model_path = os.path.expanduser('~/.cache/ultralytics/')
os.makedirs(model_path, exist_ok=True)
try:
    m = YOLO('yolov8n.pt')
    print('  ✓ YOLOv8n 已就绪')
except Exception as e:
    print(f'  ⚠ YOLO 下载失败: {e}')
" 2>/dev/null || echo "  ⚠ YOLO 模型将在首次运行时自动下载"

# Embedding 模型 (自动通过 sentence-transformers 下载)
echo "  → Embedding 模型 (bge-small-zh, ~100MB)..."
python3 -c "
from sentence_transformers import SentenceTransformer
try:
    m = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    print('  ✓ Embedding 模型已就绪')
except Exception as e:
    print(f'  ⚠ Embedding 下载失败 (首次运行时会自动下载): {e}')
" 2>/dev/null || echo "  ⚠ Embedding 模型将在首次运行时自动下载"

# LLM 模型提示 (需要 HuggingFace token 或国内镜像)
echo ""
echo "  ⚠ LLM/VLM 模型较大 (2-7GB), 不会自动下载."
echo "  → Qwen2.5-3B-Instruct-AWQ (LLM, ~2GB)"
echo "  → Qwen2.5-VL-7B-Instruct-AWQ (VLM, ~4GB, 可选)"
echo ""
echo "  下载方式:"
echo "    A) HuggingFace 直接下载:"
echo "       python3 -c \"from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct-AWQ')\""
echo ""
echo "    B) 国内镜像 (modelscope):"
echo "       python3 -c \"from modelscope import snapshot_download; snapshot_download('qwen/Qwen2.5-3B-Instruct-AWQ')\""

# ================================================================
# 4. 创建必要目录
# ================================================================

echo "[4/5] 创建目录..."
mkdir -p data/disaster_kb
mkdir -p checkpoints
mkdir -p logs
echo "  ✓ 目录已就绪"

# ================================================================
# 5. 平台特定配置提示
# ================================================================

echo "[5/5] 平台信息..."

# AutoDL
if [ -d "/root/autodl-tmp" ] || grep -q "AutoDL" /etc/motd 2>/dev/null; then
    echo "  🖥 检测到 AutoDL 平台"
    echo "  → 数据盘: /root/autodl-tmp (建议把模型放这里)"
    echo "  → 端口转发: JupyterLab 控制台 → 端口转发 → 添加 ${PORT}"
    echo "  → 或 SSH: ssh -L ${PORT}:localhost:${PORT} root@<instance-ip> -p <ssh-port>"
fi

# 恒源云
if [ -d "/hy-tmp" ] || grep -q "恒源云" /etc/motd 2>/dev/null; then
    echo "  🖥 检测到 恒源云 平台"
    echo "  → 数据盘: /hy-tmp (建议把模型放这里)"
    echo "  → SSH 端口转发: ssh -L ${PORT}:localhost:${PORT} root@<ip> -p <port>"
fi

# Vast.ai
if [ -f "/.vast" ] 2>/dev/null; then
    echo "  🖥 检测到 Vast.ai 平台"
    echo "  → 在实例设置中开放端口 ${PORT}"
    echo "  → 访问: http://<instance-ip>:${PORT}"
fi

echo ""
fi  # end of START_ONLY guard

# ================================================================
# 启动 Web 服务
# ================================================================

if [ "$INSTALL_ONLY" = true ]; then
    echo "  ✅ 安装完成! 运行以下命令启动:"
    echo "     bash setup_cloud.sh --start --port ${PORT}"
    exit 0
fi

echo "================================================================"
echo "  🚀 启动 Web 可视化仿真"
echo "================================================================"
echo "  Config:       ${CONFIG_FILE}"
echo "  Agents:       ${NUM_AGENTS}"
echo "  Port:         ${PORT}"
echo "  Frame Interval: ${FRAME_INTERVAL}"
echo ""
echo "  本地访问: http://localhost:${PORT}"
echo "  退出:     按 Ctrl+C"
echo "================================================================"
echo ""

# 启动 (用 python3 或 python)
if command -v python3 &>/dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

$PYTHON main.py \
    --config "${CONFIG_FILE}" \
    --web \
    --port "${PORT}" \
    --agents "${NUM_AGENTS}" \
    --frame-interval "${FRAME_INTERVAL}" \
    2>&1 | tee "logs/simulation_$(date +%Y%m%d_%H%M%S).log"
