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
    print(f"✓ 使用index2word")
    sample_words = model.wv.index2word[:20]
    print(f"前20个词: {sample_words}")

    # 尝试找到一个数字ID
    for word in sample_words:
        try:
            int_word = int(word)
            print(f"  找到整数词: {word} -> {int_word}")
        except:
            print(f"  非整数词: '{word}' (类型: {type(word)})")
elif hasattr(model.wv, 'vocab'):
    print(f"✓ 使用vocab")
    vocab_keys = list(model.wv.vocab.keys())[:20]
    print(f"前20个键: {vocab_keys}")
else:
    print("✗ 无法访问词汇表")

# 测试旧版本API的向量访问
print("\n测试旧版本API访问向量:")
if hasattr(model.wv, 'index2word') and len(model.wv.index2word) > 0:
    first_word = model.wv.index2word[0]
    print(f"第一个词: '{first_word}' (类型: {type(first_word)})")

    # 尝试获取向量
    try:
        vec = model.wv[first_word]
        print(f"✓ 成功获取向量，维度: {len(vec)}")
    except Exception as e:
        print(f"✗ 获取向量失败: {e}")

    # 尝试通过索引访问
    try:
        vec = model.wv.syn0[0]
        print(f"✓ 通过syn0[0]获取向量，维度: {len(vec)}")
    except Exception as e:
        print(f"✗ 通过syn0获取失败: {e}")

print("\n诊断完成！")
