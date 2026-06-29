#!/bin/bash
# ================================================================
# LLM 疏散仿真 — 7B 模型验证 (5090 一键脚本)
# 用法: bash setup_and_run.sh
# ================================================================

echo "=============================================="
echo "  环境检查"
echo "=============================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Python: $(python --version)"

echo ""
echo "=============================================="
echo "  安装依赖"
echo "=============================================="
pip install numpy numba pyyaml chromadb sentence-transformers \
    pygame matplotlib Pillow ultralytics flask tqdm orjson \
    qwen-vl-utils

echo ""
echo "=============================================="
echo "  下载 7B 模型 (~4GB)"
echo "=============================================="
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-AWQ

echo ""
echo "=============================================="
echo "  验证文件完整性"
echo "=============================================="
for f in config/mall_floorplan.yaml perception/environment.py perception/nl_converter.py decision/prompt_manager.py decision/cognitive_engine.py execution/orchestrator.py; do
    if [ -f "$f" ]; then
        echo "  [OK] $f"
    else
        echo "  [MISSING] $f — 请先上传文件!"
        exit 1
    fi
done

echo ""
echo "=============================================="
echo "  显示关键配置"
echo "=============================================="
python -c "
import yaml
with open('config/mall_floorplan.yaml') as f:
    cfg = yaml.safe_load(f)
print(f'  模型: {cfg[\"llm\"][\"model\"]}')
print(f'  GPU内存: {cfg[\"llm\"][\"gpu_memory_utilization\"]}')
print(f'  max_tokens: {cfg[\"llm\"][\"max_tokens\"]}')
print(f'  出口数: {len(cfg[\"environment\"][\"exit_positions\"])}')
print(f'  fire_spread: {cfg[\"environment\"][\"disaster_spread_rate\"]}')
"

echo ""
echo "=============================================="
echo "  开始验证 (5 runs x 3 conditions, ~1-2h)"
echo "=============================================="

python -m experiments.real_llm_validation \
    --config config/mall_floorplan.yaml \
    --n_runs 20 --agents 100 --duration 180

echo ""
echo "=============================================="
echo "  验证完成 — 汇总结果"
echo "=============================================="
python -c "
import json
with open('data/experiments/real_llm_validation.json') as f:
    d = json.load(f)
print(f\"{'Condition':<15s} {'Evac':>8s}  {'Casualty':>8s}\")
print('-'*35)
for r in d['results']:
    print(f\"{r['condition']:<15s} {r['evac_rate']:>7.1%}  {r['casualty_rate']:>7.1%}\")
"