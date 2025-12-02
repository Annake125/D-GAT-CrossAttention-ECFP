# Mol2Vec快速设置指南 🚀

你已经下载了 `mol2vec-master.zip`，里面包含了所有需要的文件！

---

## 📦 步骤1: 安装mol2vec库

### 在服务器上运行：

```bash
cd /home/user/D-GAT-CrossAttention-ECFP/

# 1. 解压你下载的文件
unzip mol2vec-master.zip

# 2. 安装mol2vec库
cd mol2vec-master
pip install -e .
cd ..

# 3. 验证安装
python -c "from mol2vec.features import mol2alt_sentence; print('✓ mol2vec安装成功')"
```

---

## 📂 步骤2: 准备预训练模型

**好消息**：模型已经在你下载的文件里了！

```bash
cd /home/user/D-GAT-CrossAttention-ECFP/

# 创建目录
mkdir -p mol2vec_pretrained

# 复制预训练模型（在 examples/models 目录下）
cp mol2vec-master/examples/models/model_300dim.pkl mol2vec_pretrained/

# 验证模型
ls -lh mol2vec_pretrained/model_300dim.pkl
```

**预期输出**:
```
-rw-r--r-- 1 user user 73M model_300dim.pkl
```

### 模型信息

根据README，这个预训练模型：
- ✅ **训练数据**: 2000万个化合物（来自ZINC数据库）
- ✅ **半径**: radius=1（与我们的实现匹配）
- ✅ **维度**: 300维
- ✅ **窗口大小**: 10
- ✅ **架构**: Skip-gram
- ✅ **UNK处理**: 替换出现<4次的标识符

---

## 🚀 步骤3: 运行预计算

```bash
cd /home/user/D-GAT-CrossAttention-ECFP/

python precompute_mol2vec.py \
    --data_path ./datasets/moses2.csv \
    --model_path ./mol2vec_pretrained/model_300dim.pkl \
    --output_path ./mol2vec_pretrained/moses2_mol2vec_300d.npy \
    --radius 1
```

### 预期输出

```
✓ 检测到mol2vec库

Mol2Vec分子嵌入预计算工具
============================================================
输入数据: ./datasets/moses2.csv
模型文件: ./mol2vec_pretrained/model_300dim.pkl
片段半径: 1
实现方式: 官方mol2vec库

加载Mol2Vec模型: ./mol2vec_pretrained/model_300dim.pkl
  格式: Pickle
  嵌入维度: 300
  词汇表大小: ~13,000 个片段

开始计算 1,936,962 个分子的Mol2Vec嵌入...
计算Mol2Vec: 100%|██████████| 1936962/1936962 [05:30<00:00]

计算完成!
  有效分子: 1,936,962 / 1,936,962 (100.00%)
  非零嵌入: 1,936,962
  平均L2范数: ~2.5

✅ Mol2Vec嵌入预计算完成!
输出文件: ./mol2vec_pretrained/moses2_mol2vec_300d.npy
数据形状: (1936962, 300)
文件大小: ~2.2 GB
```

---

## ⚙️ 步骤4: 配置训练

编辑 `diffumol/config_mol2vec.json`（已经创建好了）:

```json
{
  "use_graph": true,
  "graph_embed_path": "./hg_embed.pt",

  "use_fingerprint": true,
  "fingerprint_path": "./moses2_ecfp4_2048.npy",

  "use_mol2vec": true,
  "mol2vec_dim": 300,
  "mol2vec_path": "./mol2vec_pretrained/moses2_mol2vec_300d.npy",

  "checkpoint_path": "./weight_mol2vec"
}
```

---

## 🎯 步骤5: 开始训练

```bash
cd /home/user/D-GAT-CrossAttention-ECFP/

# 使用门控融合配置
python train.py --config diffumol/config_mol2vec.json
```

### 训练日志示例

```
### Loading molecular fingerprints from ./moses2_ecfp4_2048.npy
### Loaded fingerprints with shape (1936962, 2048)

### Loading Mol2Vec embeddings from ./mol2vec_pretrained/moses2_mol2vec_300d.npy
### Loaded Mol2Vec embeddings with shape (1936962, 300)

### [Gated Fusion] ECFP + Mol2Vec enabled
### ECFP dim: 2048, Mol2Vec dim: 300

### Creating DIFFUMOL:
[Info] 使用门控融合: ECFP(2048) + Mol2Vec(300)

### The parameter count is 85,234,567
### Training...
```

---

## ✅ 快速检查清单

- [ ] 已解压 `mol2vec-master.zip`
- [ ] 已安装mol2vec库: `pip install -e mol2vec-master/`
- [ ] 已复制 `model_300dim.pkl` 到 `mol2vec_pretrained/`
- [ ] 已运行 `precompute_mol2vec.py`
- [ ] 生成的嵌入文件存在: `ls mol2vec_pretrained/moses2_mol2vec_300d.npy`
- [ ] 嵌入形状正确: `(1936962, 300)`
- [ ] 配置文件路径已更新
- [ ] 准备开始训练！

---

## 🔍 验证脚本

### 验证mol2vec安装
```bash
python -c "
from mol2vec.features import mol2alt_sentence
from rdkit import Chem
mol = Chem.MolFromSmiles('CCO')
sentence = mol2alt_sentence(mol, radius=1)
print(f'✓ mol2vec工作正常')
print(f'  示例句子: {sentence[:5]}...')
"
```

### 验证模型加载
```bash
python -c "
import pickle
with open('./mol2vec_pretrained/model_300dim.pkl', 'rb') as f:
    model = pickle.load(f)
print(f'✓ 模型加载成功')
print(f'  维度: {model.wv.vector_size}')
print(f'  词汇量: {len(model.wv):,}')
"
```

### 验证嵌入文件
```bash
python -c "
import numpy as np
emb = np.load('./mol2vec_pretrained/moses2_mol2vec_300d.npy')
print(f'✓ 嵌入文件正确')
print(f'  形状: {emb.shape}')
print(f'  类型: {emb.dtype}')
print(f'  范围: [{emb.min():.3f}, {emb.max():.3f}]')
"
```

---

## 🛠️ 故障排查

### 问题1: "No module named 'mol2vec'"

```bash
# 检查是否正确安装
pip list | grep mol2vec

# 如果没有，重新安装
cd /home/user/D-GAT-CrossAttention-ECFP/mol2vec-master
pip install -e .
```

### 问题2: 模型文件找不到

```bash
# 检查文件位置
find /home/user/D-GAT-CrossAttention-ECFP -name "model_300dim.pkl"

# 应该在
# /home/user/D-GAT-CrossAttention-ECFP/mol2vec-master/examples/models/model_300dim.pkl

# 复制到正确位置
cp mol2vec-master/examples/models/model_300dim.pkl mol2vec_pretrained/
```

### 问题3: 预计算运行慢

```bash
# 检查是否使用了官方实现
python precompute_mol2vec.py --help

# 输出应该显示:
# ✓ 检测到mol2vec库
# 实现方式: 官方mol2vec库
```

---

## 📊 预期时间

基于2000万分子的训练经验：

| 步骤 | 预期时间 | CPU核数 |
|------|---------|---------|
| 安装mol2vec | 1-2分钟 | - |
| 预计算嵌入 | 5-10分钟 | 4核 |
| 训练模型 | 取决于配置 | GPU |

---

## 🎯 一键运行脚本

创建 `setup_mol2vec.sh`:

```bash
#!/bin/bash
set -e

echo "==> 步骤1: 安装mol2vec库"
cd /home/user/D-GAT-CrossAttention-ECFP/
unzip -q mol2vec-master.zip
cd mol2vec-master
pip install -e . -q
cd ..

echo "==> 步骤2: 准备预训练模型"
mkdir -p mol2vec_pretrained
cp mol2vec-master/examples/models/model_300dim.pkl mol2vec_pretrained/

echo "==> 步骤3: 验证安装"
python -c "from mol2vec.features import mol2alt_sentence; print('✓ mol2vec安装成功')"

echo "==> 步骤4: 运行预计算"
python precompute_mol2vec.py \
    --data_path ./datasets/moses2.csv \
    --model_path ./mol2vec_pretrained/model_300dim.pkl \
    --output_path ./mol2vec_pretrained/moses2_mol2vec_300d.npy \
    --radius 1

echo ""
echo "✅ 设置完成！现在可以运行训练："
echo "   python train.py --config diffumol/config_mol2vec.json"
```

运行：
```bash
chmod +x setup_mol2vec.sh
./setup_mol2vec.sh
```

---

## 📚 参考

- **论文**: Jaeger et al., "Mol2vec: Unsupervised Machine Learning Approach with Chemical Intuition" (2018)
- **链接**: https://pubs.acs.org/doi/10.1021/acs.jcim.7b00616
- **代码**: https://github.com/samoturk/mol2vec

---

**准备就绪？开始训练！** 🚀

```bash
cd /home/user/D-GAT-CrossAttention-ECFP/
python train.py --config diffumol/config_mol2vec.json
```
