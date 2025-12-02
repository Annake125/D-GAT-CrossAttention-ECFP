# Mol2Vec设置指南

## 📦 准备工作

### 步骤1: 安装mol2vec库（推荐）

**方式1: 从GitHub安装（推荐）**

```bash
cd /home/user/D-GAT-CrossAttention-ECFP/

# 方式1a: 直接pip安装
pip install git+https://github.com/samoturk/mol2vec

# 方式1b: 手动克隆后安装（如果服务器无外网）
# 在本地机器:
git clone https://github.com/samoturk/mol2vec.git
tar -czf mol2vec.tar.gz mol2vec/
scp mol2vec.tar.gz your_server:/home/user/D-GAT-CrossAttention-ECFP/

# 在服务器上:
cd /home/user/D-GAT-CrossAttention-ECFP/
tar -xzf mol2vec.tar.gz
cd mol2vec
pip install -e .
cd ..
```

**方式2: 不安装（使用备用实现）**

如果无法安装mol2vec库，脚本会自动使用备用实现：
- ✓ 功能完整,可正常运行
- ⚠️  精度可能略低于官方实现
- ✓ 不需要额外安装

### 步骤2: 下载预训练模型

**在本地机器上**（如果服务器无外网）:

```bash
# 下载预训练模型
wget http://hdl.handle.net/11499/112/model_300dim.pkl

# 或者从mol2vec仓库获取
# 查看: https://github.com/samoturk/mol2vec#pre-trained-models

# 上传到服务器
scp model_300dim.pkl your_server:/home/user/D-GAT-CrossAttention-ECFP/mol2vec_pretrained/
```

**在服务器上**:

```bash
cd /home/user/D-GAT-CrossAttention-ECFP/
mkdir -p mol2vec_pretrained
# 将model_300dim.pkl放到这个目录
```

---

## 🚀 运行预计算

### 基本用法

```bash
cd /home/user/D-GAT-CrossAttention-ECFP/

python precompute_mol2vec.py \
    --data_path ./datasets/moses2.csv \
    --model_path ./mol2vec_pretrained/model_300dim.pkl
```

### 完整参数

```bash
python precompute_mol2vec.py \
    --data_path ./datasets/moses2.csv \
    --model_path ./mol2vec_pretrained/model_300dim.pkl \
    --output_path ./mol2vec_pretrained/moses2_mol2vec_300d.npy \
    --radius 1
```

---

## 📊 预期输出

### 检测到mol2vec库

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
  词汇表大小: 12,345 个片段
  样本片段ID: ['1234', '5678', ...]

开始计算 1,936,962 个分子的Mol2Vec嵌入...
  - 方法: Mol2Vec (官方实现)
  - 片段半径: 1
  - 嵌入维度: 300
  - 聚合方式: 平均片段嵌入

计算Mol2Vec: 100%|██████████| 1936962/1936962 [05:23<00:00, 5987.12it/s]

计算完成!
  有效分子: 1,936,962 / 1,936,962 (100.00%)

嵌入统计:
  非零嵌入: 1,936,962
  平均L2范数: 2.456

✓ 验证通过!

✅ Mol2Vec嵌入预计算完成!
输出文件: ./datasets/moses2_mol2vec_300d.npy
数据形状: (1936962, 300)
```

### 未检测到mol2vec库（使用备用实现）

```
⚠️  未检测到mol2vec库，将使用备用实现
建议安装: pip install git+https://github.com/samoturk/mol2vec

Mol2Vec分子嵌入预计算工具
============================================================
实现方式: 备用实现

[其余输出类似]
```

---

## ✅ 验证安装

### 检查mol2vec是否安装

```bash
python -c "from mol2vec.features import mol2alt_sentence; print('✓ mol2vec已安装')"
```

**预期输出**:
```
✓ mol2vec已安装
```

### 检查模型文件

```bash
python -c "
import pickle
model = pickle.load(open('./mol2vec_pretrained/model_300dim.pkl', 'rb'))
print(f'模型维度: {model.wv.vector_size}')
print(f'词汇表大小: {len(model.wv):,}')
"
```

**预期输出**:
```
模型维度: 300
词汇表大小: 12,345
```

### 检查生成的嵌入文件

```bash
python -c "
import numpy as np
emb = np.load('./mol2vec_pretrained/moses2_mol2vec_300d.npy')
print(f'嵌入形状: {emb.shape}')
print(f'数据类型: {emb.dtype}')
print(f'平均L2范数: {np.linalg.norm(emb[emb.sum(axis=1)>0], axis=1).mean():.3f}')
"
```

**预期输出**:
```
嵌入形状: (1936962, 300)
数据类型: float32
平均L2范数: 2.456
```

---

## 🛠️ 故障排查

### 问题1: 无法导入mol2vec

```
ImportError: No module named 'mol2vec'
```

**解决方案**:
```bash
# 方案1: 直接pip安装
pip install git+https://github.com/samoturk/mol2vec

# 方案2: 手动安装
cd /home/user/D-GAT-CrossAttention-ECFP/
git clone https://github.com/samoturk/mol2vec.git
cd mol2vec
pip install -e .

# 方案3: 使用备用实现
# 脚本会自动使用备用实现,无需额外操作
```

### 问题2: 模型加载失败

```
错误: 无法加载模型
详细信息: EOFError: Ran out of input
```

**可能原因**:
- 模型文件下载不完整
- 模型文件损坏
- gensim版本不兼容

**解决方案**:
```bash
# 1. 重新下载模型文件
rm ./mol2vec_pretrained/model_300dim.pkl
wget http://hdl.handle.net/11499/112/model_300dim.pkl -P ./mol2vec_pretrained/

# 2. 检查gensim版本
pip show gensim

# 3. 安装正确版本
pip install gensim==4.3.0
```

### 问题3: 嵌入全为零

```
嵌入统计:
  非零嵌入: 0
```

**可能原因**:
- 模型词汇表与数据不匹配
- 片段半径参数不正确

**解决方案**:
```bash
# 检查模型词汇表
python -c "
import pickle
model = pickle.load(open('./mol2vec_pretrained/model_300dim.pkl', 'rb'))
print('样本词:', list(model.wv.key_to_index.keys())[:10])
"

# 尝试不同的半径参数
python precompute_mol2vec.py \
    --data_path ./datasets/moses2.csv \
    --model_path ./mol2vec_pretrained/model_300dim.pkl \
    --radius 1  # 或尝试 2
```

### 问题4: 内存不足

```
MemoryError: Unable to allocate array
```

**解决方案**:
```bash
# 方案1: 分批处理（需要修改脚本）
# 联系开发者获取分批处理版本

# 方案2: 增加swap空间
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## 📂 文件结构检查

确保你的目录结构如下:

```
D-GAT-CrossAttention-ECFP/
├── mol2vec/                          # mol2vec库（可选）
│   ├── mol2vec/
│   │   ├── __init__.py
│   │   ├── features.py               # 核心功能
│   │   └── helpers.py
│   └── setup.py
├── mol2vec_pretrained/               # 预训练模型目录
│   ├── model_300dim.pkl              # 预训练模型
│   └── moses2_mol2vec_300d.npy       # 生成的嵌入（预计算后）
├── datasets/
│   └── moses2.csv                    # 数据集
├── precompute_mol2vec.py             # 预计算脚本
└── diffumol/
    └── config_mol2vec.json           # 配置文件
```

---

## 🔄 与项目整合

### 更新配置文件

编辑 `diffumol/config_mol2vec.json`:

```json
{
  "use_fingerprint": true,
  "fp_dim": 2048,
  "fingerprint_path": "./moses2_ecfp4_2048.npy",

  "use_mol2vec": true,
  "mol2vec_dim": 300,
  "mol2vec_path": "./mol2vec_pretrained/moses2_mol2vec_300d.npy"
}
```

### 开始训练

```bash
python train.py --config diffumol/config_mol2vec.json
```

---

## 📚 参考资料

- **mol2vec GitHub**: https://github.com/samoturk/mol2vec
- **预训练模型**: http://hdl.handle.net/11499/112/model_300dim.pkl
- **论文**: Jaeger et al., "Mol2vec: Unsupervised Machine Learning Approach with Chemical Intuition" (2018)
- **论文链接**: https://pubs.acs.org/doi/10.1021/acs.jcim.7b00616

---

## 💡 最佳实践

1. **推荐安装官方mol2vec库** - 确保最佳兼容性和精度
2. **验证预训练模型** - 确保模型文件完整且可加载
3. **使用默认半径** - radius=1 与论文一致,通常效果最好
4. **检查嵌入质量** - 运行验证脚本确保嵌入非零
5. **保存中间结果** - 预计算的嵌入可以重复使用,节省时间

---

## ✅ 快速检查清单

开始训练前,确认以下项目:

- [ ] 已安装mol2vec库（或接受使用备用实现）
- [ ] 已下载model_300dim.pkl到mol2vec_pretrained/
- [ ] 已运行precompute_mol2vec.py
- [ ] 生成的嵌入文件形状正确: (N, 300)
- [ ] 嵌入统计显示非零嵌入数量合理
- [ ] 已更新config_mol2vec.json中的路径
- [ ] 所有路径都是绝对路径或相对于项目根目录

---

**准备就绪后,运行训练!** 🚀

```bash
cd /home/user/D-GAT-CrossAttention-ECFP/
python train.py --config diffumol/config_mol2vec.json
```
