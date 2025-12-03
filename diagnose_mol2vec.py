#!/usr/bin/env python
"""
诊断mol2vec模型和分子片段匹配问题
"""
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem

# 加载模型
print("加载模型...")
with open('./mol2vec_pretrained/model_300dim.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"模型向量维度: {model.wv.syn0.shape[1]}")
print(f"模型词汇表大小: {model.wv.syn0.shape[0]}")

# 获取词汇表的前10个词
print("\n模型词汇表样本:")
if hasattr(model.wv, 'index2word'):
    sample_words = model.wv.index2word[:10]
    print(f"词汇表类型: index2word (旧版本)")
    for i, word in enumerate(sample_words):
        print(f"  {i}: {word} (类型: {type(word)})")

# 测试一个简单分子
test_smiles = "CCO"  # 乙醇
print(f"\n测试分子: {test_smiles}")

mol = Chem.MolFromSmiles(test_smiles)
print(f"RDKit分子对象: {mol is not None}")

# 生成Morgan片段
info = {}
fp = AllChem.GetMorganFingerprint(mol, radius=1, bitInfo=info)

print(f"\n生成的片段数量: {len(info)}")
print("片段ID样本:")
for fragment_id in list(info.keys())[:5]:
    print(f"  {fragment_id} (类型: {type(fragment_id)})")

# 测试不同的片段表示方式
print("\n测试词汇表访问:")
test_ids = list(info.keys())[:3]

for fragment_id in test_ids:
    # 方式1: 直接使用整数
    try:
        if fragment_id in model.wv:
            print(f"✓ {fragment_id} (int) 在词汇表中")
        else:
            print(f"✗ {fragment_id} (int) 不在词汇表中")
    except Exception as e:
        print(f"✗ {fragment_id} (int) 访问失败: {e}")

    # 方式2: 转换为字符串
    try:
        str_id = str(fragment_id)
        if str_id in model.wv:
            print(f"✓ '{str_id}' (str) 在词汇表中")
        else:
            print(f"✗ '{str_id}' (str) 不在词汇表中")
    except Exception as e:
        print(f"✗ '{str_id}' (str) 访问失败: {e}")

# 检查词汇表中实际有什么
print("\n检查词汇表实际内容:")
if hasattr(model.wv, 'index2word'):
    # 尝试找到一个数字ID
    for word in model.wv.index2word[:20]:
        try:
            # 尝试转换为整数
            int_word = int(word)
            print(f"  词汇表包含: {word} -> 可转换为整数: {int_word}")
            break
        except:
            print(f"  词汇表包含: {word} (非整数)")

print("\n诊断完成！")
