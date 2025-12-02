#!/bin/bash
# Mol2Vec一键设置脚本
# 自动安装mol2vec库、准备预训练模型并运行预计算

set -e  # 遇到错误立即退出

echo "=========================================="
echo "  Mol2Vec自动设置脚本"
echo "=========================================="
echo ""

# 检查当前目录
if [[ ! -f "precompute_mol2vec.py" ]]; then
    echo "错误: 请在项目根目录运行此脚本"
    echo "当前目录: $(pwd)"
    exit 1
fi

# 步骤1: 安装mol2vec库
echo "==> 步骤1/4: 安装mol2vec库"
if [[ -f "mol2vec-master.zip" ]]; then
    echo "  - 发现 mol2vec-master.zip，正在解压..."
    unzip -q mol2vec-master.zip
    echo "  ✓ 解压完成"
elif [[ -d "mol2vec-master" ]]; then
    echo "  - mol2vec-master 目录已存在"
else
    echo "  错误: 找不到 mol2vec-master.zip"
    echo "  请确保已下载 mol2vec-master.zip 到项目根目录"
    exit 1
fi

echo "  - 安装mol2vec库..."
cd mol2vec-master
pip install -e . -q
cd ..
echo "  ✓ mol2vec库安装完成"
echo ""

# 步骤2: 验证安装
echo "==> 步骤2/4: 验证mol2vec安装"
python -c "from mol2vec.features import mol2alt_sentence; print('  ✓ mol2vec库可用')" || {
    echo "  错误: mol2vec库安装失败"
    exit 1
}
echo ""

# 步骤3: 准备预训练模型
echo "==> 步骤3/4: 准备预训练模型"
mkdir -p mol2vec_pretrained

if [[ -f "mol2vec-master/examples/models/model_300dim.pkl" ]]; then
    echo "  - 复制预训练模型..."
    cp mol2vec-master/examples/models/model_300dim.pkl mol2vec_pretrained/
    echo "  ✓ 模型文件已复制到 mol2vec_pretrained/"

    # 显示文件大小
    MODEL_SIZE=$(ls -lh mol2vec_pretrained/model_300dim.pkl | awk '{print $5}')
    echo "  - 模型文件大小: $MODEL_SIZE"
else
    echo "  错误: 找不到预训练模型文件"
    echo "  期望位置: mol2vec-master/examples/models/model_300dim.pkl"
    exit 1
fi
echo ""

# 验证模型可加载
echo "  - 验证模型文件..."
python -c "
import pickle
with open('./mol2vec_pretrained/model_300dim.pkl', 'rb') as f:
    model = pickle.load(f)
print(f'  ✓ 模型加载成功 (维度: {model.wv.vector_size}, 词汇量: {len(model.wv):,})')
" || {
    echo "  错误: 模型文件损坏或无法加载"
    exit 1
}
echo ""

# 步骤4: 运行预计算
echo "==> 步骤4/4: 预计算Mol2Vec嵌入"
echo "  这可能需要5-10分钟，请耐心等待..."
echo ""

# 检查数据文件是否存在
if [[ ! -f "./datasets/moses2.csv" ]]; then
    echo "  错误: 找不到数据文件 ./datasets/moses2.csv"
    exit 1
fi

python precompute_mol2vec.py \
    --data_path ./datasets/moses2.csv \
    --model_path ./mol2vec_pretrained/model_300dim.pkl \
    --output_path ./mol2vec_pretrained/moses2_mol2vec_300d.npy \
    --radius 1

echo ""
echo "=========================================="
echo "  ✅ Mol2Vec设置完成！"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  - mol2vec库: mol2vec-master/"
echo "  - 预训练模型: mol2vec_pretrained/model_300dim.pkl"
echo "  - 预计算嵌入: mol2vec_pretrained/moses2_mol2vec_300d.npy"
echo ""

# 验证嵌入文件
if [[ -f "./mol2vec_pretrained/moses2_mol2vec_300d.npy" ]]; then
    EMB_SIZE=$(ls -lh mol2vec_pretrained/moses2_mol2vec_300d.npy | awk '{print $5}')
    echo "嵌入文件信息:"
    python -c "
import numpy as np
emb = np.load('./mol2vec_pretrained/moses2_mol2vec_300d.npy')
print(f'  - 文件大小: $EMB_SIZE')
print(f'  - 数据形状: {emb.shape}')
print(f'  - 数据类型: {emb.dtype}')
print(f'  - 非零嵌入: {(emb.sum(axis=1) != 0).sum():,}')
    "
    echo ""
fi

echo "下一步:"
echo "  1. 检查配置文件: diffumol/config_mol2vec.json"
echo "  2. 开始训练: python train.py --config diffumol/config_mol2vec.json"
echo ""
echo "详细文档见: QUICK_START.md"
echo "=========================================="
