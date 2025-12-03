#!/usr/bin/env python
"""
诊断mol2vec模型和分子片段匹配问题
测试GensimModelWrapper兼容性
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
from rdkit import Chem
from rdkit.Chem import AllChem

# 导入我们的兼容性包装器
from precompute_mol2vec import GensimModelWrapper

# 加载模型
print("加载模型...")
with open('./mol2vec_pretrained/model_300dim.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"模型向量维度: {model.wv.syn0.shape[1]}")
print(f"模型词汇表大小: {model.wv.syn0.shape[0]}")

# 创建包装器
print("\n创建GensimModelWrapper...")
wrapper = GensimModelWrapper(model)
print(f"✓ 包装器创建成功")
print(f"  - vocab_dict可用: {wrapper.vocab_dict is not None}")
print(f"  - index2word_list可用: {wrapper.index2word_list is not None}")
if wrapper.vocab_dict is not None:
    print(f"  - vocab_dict大小: {len(wrapper.vocab_dict):,}")
if wrapper.index2word_list is not None:
    print(f"  - index2word_list大小: {len(wrapper.index2word_list):,}")

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
print("\n测试词汇表访问 (使用包装器):")
test_ids = list(info.keys())[:3]

for fragment_id in test_ids:
    # 方式1: 直接使用整数
    str_id = str(fragment_id)

    print(f"\n测试片段 ID: {fragment_id}")

    # 使用包装器的contains方法
    if wrapper.contains(fragment_id):
        print(f"  ✓ {fragment_id} (int) 在词汇表中")
        vec = wrapper.get_vector(fragment_id)
        if vec is not None:
            print(f"    - 向量维度: {len(vec)}")
            print(f"    - 向量范数: {(vec**2).sum()**0.5:.3f}")
        else:
            print(f"    ✗ 但无法获取向量")
    else:
        print(f"  ✗ {fragment_id} (int) 不在词汇表中")

    # 方式2: 转换为字符串
    if wrapper.contains(str_id):
        print(f"  ✓ '{str_id}' (str) 在词汇表中")
        vec = wrapper.get_vector(str_id)
        if vec is not None:
            print(f"    - 向量维度: {len(vec)}")
            print(f"    - 向量范数: {(vec**2).sum()**0.5:.3f}")
        else:
            print(f"    ✗ 但无法获取向量")
    else:
        print(f"  ✗ '{str_id}' (str) 不在词汇表中")

# 首先列出模型的所有属性
print("\n=== 模型属性列表 ===")
print("model.wv 的属性:")
wv_attrs = [attr for attr in dir(model.wv) if not attr.startswith('_')]
print(f"  {wv_attrs[:20]}")  # 显示前20个

# 检查关键属性
print("\n关键属性检查:")
print(f"  hasattr(model.wv, 'index2word'): {hasattr(model.wv, 'index2word')}")
print(f"  hasattr(model.wv, 'index_to_key'): {hasattr(model.wv, 'index_to_key')}")
print(f"  hasattr(model.wv, 'vocab'): {hasattr(model.wv, 'vocab')}")
print(f"  hasattr(model.wv, 'key_to_index'): {hasattr(model.wv, 'key_to_index')}")
print(f"  hasattr(model.wv, 'syn0'): {hasattr(model.wv, 'syn0')}")
print(f"  hasattr(model.wv, 'vectors'): {hasattr(model.wv, 'vectors')}")

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
    print("✗ 标准属性不可用，尝试直接访问syn0...")
    if hasattr(model.wv, 'syn0'):
        print(f"✓ 找到syn0，形状: {model.wv.syn0.shape}")
        print("  这是一个非常旧的gensim模型")

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
