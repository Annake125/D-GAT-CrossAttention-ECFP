"""
预计算Mol2Vec分子嵌入 for MOSES2 Dataset
Mol2Vec: 基于Word2Vec的无监督分子表示学习
- 将分子片段(Morgan substructures)视为"词"
- 使用预训练的Word2Vec模型生成300维语义嵌入
- 提供与ECFP互补的语义相似性信息

参考论文: Jaeger et al., "Mol2vec: Unsupervised Machine Learning Approach
          with Chemical Intuition" (2018)
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from gensim.models import word2vec
from tqdm import tqdm
import argparse
import os
import sys
import pickle

# 关闭RDKit警告
RDLogger.DisableLog('rdApp.*')


class MolSentence:
    """将分子表示为Morgan片段的"句子" """
    def __init__(self, mol, radius=1):
        self.mol = mol
        self.radius = radius
        self.sentence = self._mol_to_sentence()

    def _mol_to_sentence(self):
        """提取Morgan片段作为"词汇" """
        if self.mol is None:
            return []

        # 获取Morgan圆形指纹的子结构标识符
        info = {}
        _ = AllChem.GetMorganFingerprint(self.mol, self.radius, bitInfo=info)

        # 将子结构ID转换为"词"
        mol_sentence = []
        for fragment_id, occurrences in info.items():
            # 将fragment_id转换为字符串作为"词"
            word = f"frag_{fragment_id}"
            mol_sentence.append(word)

        return mol_sentence if len(mol_sentence) > 0 else ['UNK']


def mol_to_sentence(smiles, radius=1):
    """
    将SMILES转换为Mol2Vec的"句子"表示
    Args:
        smiles: SMILES字符串
        radius: Morgan片段半径(默认1,对应mol2vec论文设置)
    Returns:
        list: 分子片段的列表
    """
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None

        mol_sent = MolSentence(mol, radius=radius)
        return mol_sent.sentence
    except Exception as e:
        return None


def compute_mol2vec_embedding(smiles, model, radius=1, vector_size=300):
    """
    使用预训练的Mol2Vec模型计算分子嵌入
    Args:
        smiles: SMILES字符串
        model: 预训练的Word2Vec模型
        radius: Morgan片段半径
        vector_size: 嵌入维度(默认300)
    Returns:
        numpy array: 分子嵌入向量(平均所有片段)
    """
    try:
        # 获取分子的"句子"表示
        sentence = mol_to_sentence(smiles, radius=radius)

        if sentence is None or len(sentence) == 0:
            return None

        # 查找每个片段的嵌入
        vectors = []
        for word in sentence:
            if word in model.wv:
                vectors.append(model.wv[word])
            # 如果片段不在词汇表中,跳过(稀有片段)

        if len(vectors) == 0:
            # 如果所有片段都未知,返回零向量
            return np.zeros(vector_size, dtype=np.float32)

        # 平均所有片段嵌入得到分子嵌入
        mol_embedding = np.mean(vectors, axis=0).astype(np.float32)

        return mol_embedding

    except Exception as e:
        return None


def load_mol2vec_model(model_path):
    """
    加载预训练的Mol2Vec模型
    支持多种格式: .pkl (pickle), .model (gensim), .bin (word2vec binary)
    """
    print(f"\n加载Mol2Vec模型: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    try:
        # 尝试pickle格式
        if model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"  格式: Pickle")

        # 尝试gensim格式
        elif model_path.endswith('.model'):
            model = word2vec.Word2Vec.load(model_path)
            print(f"  格式: Gensim")

        # 尝试word2vec binary格式
        elif model_path.endswith('.bin'):
            model = word2vec.Word2Vec.load_word2vec_format(model_path, binary=True)
            print(f"  格式: Word2Vec Binary")

        else:
            # 默认尝试pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"  格式: 自动检测 (Pickle)")

        # 检查模型属性
        vector_size = model.wv.vector_size
        vocab_size = len(model.wv)

        print(f"  嵌入维度: {vector_size}")
        print(f"  词汇表大小: {vocab_size:,} 个片段")

        # 检查几个常见片段是否存在
        sample_words = [w for w in list(model.wv.key_to_index.keys())[:5]]
        print(f"  样本片段: {sample_words}")

        return model, vector_size

    except Exception as e:
        print(f"\n错误: 无法加载模型")
        print(f"详细信息: {str(e)}")
        print(f"\n请确保模型文件格式正确:")
        print(f"  - .pkl: Pickle格式")
        print(f"  - .model: Gensim Word2Vec格式")
        print(f"  - .bin: Word2Vec binary格式")
        raise


def analyze_moses_dataset(df):
    """分析MOSES数据集的基本信息"""
    print("\n" + "="*60)
    print("MOSES数据集分析")
    print("="*60)

    # 列名标准化（转小写）
    df.columns = df.columns.str.lower()

    print(f"总分子数: {len(df):,}")
    print(f"列名: {', '.join(df.columns.tolist())}")

    # 检查SPLIT列
    if 'split' in df.columns:
        print(f"\n数据集划分 (SPLIT):")
        split_counts = df['split'].value_counts()
        for split_name, count in split_counts.items():
            print(f"  - {split_name:12s}: {count:7,} ({count/len(df)*100:5.2f}%)")
    else:
        print("\n警告: 未找到'split'列，无法统计数据集划分")

    # 检查SMILES质量
    print(f"\nSMILES列检测: {'✓' if 'smiles' in df.columns else '❌ 未找到'}")

    return df


def precompute_moses_mol2vec(csv_path, model_path, output_path=None, radius=1):
    """
    为MOSES数据集预计算Mol2Vec嵌入
    Args:
        csv_path: moses2.csv文件路径
        model_path: 预训练Mol2Vec模型路径
        output_path: 输出.npy文件路径（默认自动生成）
        radius: Morgan片段半径(默认1,与mol2vec论文一致)
    """
    print(f"\nMol2Vec分子嵌入预计算工具")
    print(f"="*60)
    print(f"输入数据: {csv_path}")
    print(f"模型文件: {model_path}")
    print(f"片段半径: {radius}")

    # 检查文件是否存在
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")

    # 加载Mol2Vec模型
    model, vector_size = load_mol2vec_model(model_path)

    # 读取数据
    print(f"\n加载数据...")
    df = pd.read_csv(csv_path)

    # 分析数据集
    df = analyze_moses_dataset(df)

    # 检查SMILES列（大小写不敏感）
    smiles_col = None
    for col in df.columns:
        if col.lower() == 'smiles':
            smiles_col = col
            break

    if smiles_col is None:
        raise ValueError(f"未找到SMILES列！可用列: {df.columns.tolist()}")

    smiles_list = df[smiles_col].tolist()
    num_mols = len(smiles_list)

    # 数据验证
    print(f"\n验证SMILES质量...")
    null_count = df[smiles_col].isnull().sum()
    if null_count > 0:
        print(f"发现 {null_count} 个空值SMILES")
        smiles_list = [s if pd.notna(s) else '' for s in smiles_list]

    # 开始计算Mol2Vec嵌入
    print(f"\n开始计算 {num_mols:,} 个分子的Mol2Vec嵌入...")
    print(f"  - 方法: Mol2Vec (无监督)")
    print(f"  - 片段半径: {radius}")
    print(f"  - 嵌入维度: {vector_size}")
    print(f"  - 聚合方式: 平均片段嵌入")

    embeddings = np.zeros((num_mols, vector_size), dtype=np.float32)
    invalid_indices = []
    invalid_smiles = []
    unknown_fragment_counts = []  # 统计未知片段

    for idx, smiles in enumerate(tqdm(smiles_list, desc="计算Mol2Vec", ncols=80)):
        if pd.isna(smiles) or smiles == '':
            invalid_indices.append(idx)
            invalid_smiles.append("(empty)")
            continue

        emb = compute_mol2vec_embedding(smiles, model, radius=radius, vector_size=vector_size)

        if emb is None:
            invalid_indices.append(idx)
            invalid_smiles.append(smiles[:50])  # 记录前50个字符
        else:
            embeddings[idx] = emb

            # 统计未知片段比例(可选)
            sentence = mol_to_sentence(smiles, radius=radius)
            if sentence:
                known = sum(1 for w in sentence if w in model.wv)
                unknown_fragment_counts.append(1.0 - known / len(sentence))

    # 统计结果
    valid_count = num_mols - len(invalid_indices)
    print(f"\n计算完成!")
    print(f"  有效分子: {valid_count:,} / {num_mols:,} ({valid_count/num_mols*100:.2f}%)")
    print(f"  无效分子: {len(invalid_indices):,}")

    if len(invalid_indices) > 0 and len(invalid_indices) <= 10:
        print(f"\n无效SMILES列表:")
        for i, (idx, smi) in enumerate(zip(invalid_indices, invalid_smiles), 1):
            print(f"    {i}. [行{idx}] {smi}")
    elif len(invalid_indices) > 10:
        print(f"\n无效SMILES过多 ({len(invalid_indices)}个)，仅显示前10个:")
        for i in range(10):
            idx = invalid_indices[i]
            smi = invalid_smiles[i]
            print(f"    {i+1}. [行{idx}] {smi}")
        print(f"    ...")

    # 嵌入统计
    non_zero_embs = np.sum(np.abs(embeddings).sum(axis=1) > 1e-6)
    avg_norm = np.mean(np.linalg.norm(embeddings[np.abs(embeddings).sum(axis=1) > 1e-6], axis=1))

    print(f"\n嵌入统计:")
    print(f"  非零嵌入: {non_zero_embs:,}")
    print(f"  平均L2范数: {avg_norm:.3f}")

    if len(unknown_fragment_counts) > 0:
        avg_unknown = np.mean(unknown_fragment_counts) * 100
        print(f"  平均未知片段比例: {avg_unknown:.2f}%")
        if avg_unknown > 30:
            print(f"  ⚠️  警告: 超过30%的片段未知,可能影响嵌入质量")
            print(f"      建议: 使用更大的预训练模型或在你的数据上微调")

    # 自动生成输出路径
    if output_path is None:
        base_name = os.path.splitext(csv_path)[0]
        output_path = f"{base_name}_mol2vec_{vector_size}d.npy"

    # 保存嵌入
    print(f"\n保存嵌入到: {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    np.save(output_path, embeddings)

    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  文件大小: {file_size_mb:.2f} MB")

    # 验证保存
    print(f"\n验证保存的文件...")
    loaded_emb = np.load(output_path)
    assert loaded_emb.shape == (num_mols, vector_size), "形状不匹配！"
    assert np.allclose(loaded_emb, embeddings), "数据不匹配！"
    print(f"✓ 验证通过！")

    print(f"\n" + "="*60)
    print(f"✅ Mol2Vec嵌入预计算完成!")
    print(f"="*60)
    print(f"输出文件: {output_path}")
    print(f"数据形状: {loaded_emb.shape}")
    print(f"\n使用方法:")
    print(f"   1. 修改 diffumol/config.json:")
    print(f'      "use_mol2vec": true,')
    print(f'      "mol2vec_dim": {vector_size},')
    print(f'      "mol2vec_path": "{output_path}"')
    print(f"   2. 运行训练: python train.py")
    print(f"="*60)

    return embeddings


def main():
    parser = argparse.ArgumentParser(
        description='为MOSES数据集预计算Mol2Vec分子嵌入',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本使用（使用预训练的300维模型）
  python precompute_mol2vec.py \\
      --data_path ./datasets/moses2.csv \\
      --model_path ./mol2vec_pretrained/model_300dim.pkl

  # 指定输出路径
  python precompute_mol2vec.py \\
      --data_path ./datasets/moses2.csv \\
      --model_path ./mol2vec_pretrained/model_300dim.pkl \\
      --output_path ./my_mol2vec_embeddings.npy

  # 使用不同的片段半径
  python precompute_mol2vec.py \\
      --data_path ./datasets/moses2.csv \\
      --model_path ./mol2vec_pretrained/model_300dim.pkl \\
      --radius 2

注意事项:
  1. 模型文件路径必须指向有效的Mol2Vec预训练模型
  2. 支持的模型格式: .pkl (pickle), .model (gensim), .bin (word2vec)
  3. 推荐使用300维预训练模型 (model_300dim.pkl)
  4. 片段半径默认为1 (与mol2vec论文一致)
        """
    )

    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='moses2.csv文件路径'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='预训练Mol2Vec模型路径 (.pkl, .model, 或 .bin)'
    )

    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='输出.npy文件路径（默认: <data_path>_mol2vec_<dim>d.npy）'
    )

    parser.add_argument(
        '--radius',
        type=int,
        default=1,
        choices=[1, 2, 3],
        help='Morgan片段半径 (默认=1, 与mol2vec论文一致)'
    )

    args = parser.parse_args()

    # 检查依赖
    try:
        from rdkit import Chem
        from gensim.models import word2vec
    except ImportError as e:
        print(f"错误: 缺少依赖库!")
        print(f"详细信息: {str(e)}")
        print(f"\n请安装:")
        print(f"  RDKit: conda install -c conda-forge rdkit")
        print(f"  Gensim: pip install gensim==4.3.0")
        sys.exit(1)

    # 执行预计算
    try:
        precompute_moses_mol2vec(
            csv_path=args.data_path,
            model_path=args.model_path,
            output_path=args.output_path,
            radius=args.radius
        )
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
