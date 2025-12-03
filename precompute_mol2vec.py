"""
预计算Mol2Vec分子嵌入 for MOSES2 Dataset
使用官方mol2vec库: https://github.com/samoturk/mol2vec

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
from rdkit import RDLogger
from tqdm import tqdm
import argparse
import os
import sys

# 关闭RDKit警告
RDLogger.DisableLog('rdApp.*')

# 尝试导入mol2vec库
try:
    from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
    from gensim.models import word2vec
    HAS_MOL2VEC = True
    print("✓ 检测到mol2vec库")
except ImportError:
    HAS_MOL2VEC = False
    print("⚠️  未检测到mol2vec库，将使用备用实现")
    print("建议安装: pip install git+https://github.com/samoturk/mol2vec")
    from gensim.models import word2vec
    from rdkit.Chem import AllChem


# ============== Gensim兼容层 ==============
class GensimModelWrapper:
    """
    兼容所有版本gensim的包装器
    处理从非常旧的版本(syn0, index2word)到最新版本(vectors, index_to_key)
    """
    def __init__(self, model):
        self.model = model
        self.wv = model.wv

        # 确定词汇表访问方式
        self.vocab_dict = None
        self.index2word_list = None

        # 尝试建立词汇表映射
        if hasattr(self.wv, 'key_to_index'):
            # Gensim 4.x+
            self.vocab_dict = self.wv.key_to_index
            self.index2word_list = self.wv.index_to_key
        elif hasattr(self.wv, 'vocab'):
            # Gensim 1.x-3.x
            try:
                self.vocab_dict = self.wv.vocab
                if hasattr(self.wv, 'index2word'):
                    self.index2word_list = self.wv.index2word
            except:
                pass
        elif hasattr(self.wv, 'index2word'):
            # 可能有index2word但没有vocab的版本
            try:
                self.index2word_list = self.wv.index2word
                # 手动构建vocab字典
                self.vocab_dict = {word: idx for idx, word in enumerate(self.index2word_list)}
            except:
                pass

    def get_vector(self, word):
        """
        获取词向量，兼容所有gensim版本
        返回: numpy array 或 None (如果词不存在)
        """
        try:
            # 方法1: 标准访问 (适用于大多数版本)
            if hasattr(self.wv, '__getitem__'):
                return self.wv[word]
        except (KeyError, ValueError):
            pass

        try:
            # 方法2: 通过index访问 (适用于旧版本)
            if self.vocab_dict is not None and word in self.vocab_dict:
                if hasattr(self.wv, 'syn0'):
                    # 非常旧的版本
                    if isinstance(self.vocab_dict[word], int):
                        idx = self.vocab_dict[word]
                    else:
                        idx = self.vocab_dict[word].index
                    return self.wv.syn0[idx]
                elif hasattr(self.wv, 'vectors'):
                    idx = self.vocab_dict[word] if isinstance(self.vocab_dict[word], int) else self.vocab_dict[word].index
                    return self.wv.vectors[idx]
        except:
            pass

        try:
            # 方法3: 直接从word获取 (某些版本)
            if hasattr(self.wv, 'word_vec'):
                return self.wv.word_vec(word)
        except:
            pass

        return None

    def contains(self, word):
        """
        检查词是否在词汇表中
        """
        # 方法1: 使用vocab_dict
        if self.vocab_dict is not None:
            return word in self.vocab_dict

        # 方法2: 尝试 __contains__
        try:
            return word in self.wv
        except:
            pass

        # 方法3: 尝试获取向量
        vec = self.get_vector(word)
        return vec is not None


# ============== 备用实现（如果没有安装mol2vec库） ==============
class MolSentenceBackup:
    """备用实现：将分子表示为Morgan片段的"句子" """
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

        # 将子结构ID转换为"词"（注意：这需要与预训练模型的格式匹配）
        mol_sentence = []
        for fragment_id in sorted(info.keys()):
            # mol2vec使用标识符作为词，这里需要转换为字符串
            word = str(fragment_id)
            mol_sentence.append(word)

        return mol_sentence if len(mol_sentence) > 0 else ['UNK']


def mol_to_sentence_backup(smiles, radius=1):
    """
    备用实现：将SMILES转换为Mol2Vec的"句子"表示
    """
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None

        mol_sent = MolSentenceBackup(mol, radius=radius)
        return mol_sent.sentence
    except Exception as e:
        return None


def compute_mol2vec_embedding_backup(smiles, model_wrapper, radius=1, vector_size=300):
    """
    备用实现：使用预训练的Mol2Vec模型计算分子嵌入

    参数:
        model_wrapper: GensimModelWrapper对象（兼容所有gensim版本）
    """
    try:
        sentence = mol_to_sentence_backup(smiles, radius=radius)

        if sentence is None or len(sentence) == 0:
            return None

        # 查找每个片段的嵌入
        vectors = []
        for word in sentence:
            try:
                if model_wrapper.contains(word):
                    vec = model_wrapper.get_vector(word)
                    if vec is not None:
                        vectors.append(vec)
            except:
                # 如果片段不在词汇表中,跳过
                pass

        if len(vectors) == 0:
            # 如果所有片段都未知,返回零向量
            return np.zeros(vector_size, dtype=np.float32)

        # 平均所有片段嵌入得到分子嵌入
        mol_embedding = np.mean(vectors, axis=0).astype(np.float32)
        return mol_embedding

    except Exception as e:
        return None


# ============== 官方mol2vec实现（推荐） ==============
def compute_mol2vec_embedding_official(smiles, model_wrapper, radius=1, vector_size=300):
    """
    使用官方mol2vec库计算分子嵌入

    参数:
        model_wrapper: GensimModelWrapper对象（兼容所有gensim版本）
    """
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None

        # 使用官方mol2vec的mol2alt_sentence函数
        # 这会生成与预训练模型兼容的分子句子
        sentence = mol2alt_sentence(mol, radius=radius)

        if sentence is None or len(sentence) == 0:
            return None

        # 计算句子的向量表示（平均所有词向量）
        vectors = []
        for word in sentence:
            try:
                if model_wrapper.contains(word):
                    vec = model_wrapper.get_vector(word)
                    if vec is not None:
                        vectors.append(vec)
            except:
                pass

        if len(vectors) == 0:
            return np.zeros(vector_size, dtype=np.float32)

        mol_embedding = np.mean(vectors, axis=0).astype(np.float32)
        return mol_embedding

    except Exception as e:
        return None


# ============== 统一接口 ==============
def compute_mol2vec_embedding(smiles, model_wrapper, radius=1, vector_size=300):
    """
    自动选择最佳实现来计算Mol2Vec嵌入

    参数:
        smiles: SMILES字符串
        model_wrapper: GensimModelWrapper对象（兼容所有gensim版本）
        radius: Morgan片段半径
        vector_size: 嵌入向量维度
    """
    if HAS_MOL2VEC:
        return compute_mol2vec_embedding_official(smiles, model_wrapper, radius, vector_size)
    else:
        return compute_mol2vec_embedding_backup(smiles, model_wrapper, radius, vector_size)


def load_mol2vec_model(model_path):
    """
    加载预训练的Mol2Vec模型
    支持多种格式: .pkl (pickle), .model (gensim), .bin (word2vec binary)
    """
    print(f"\n加载Mol2Vec模型: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    try:
        # 尝试不同的加载方式
        if model_path.endswith('.pkl'):
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"  格式: Pickle")

        elif model_path.endswith('.model'):
            model = word2vec.Word2Vec.load(model_path)
            print(f"  格式: Gensim")

        elif model_path.endswith('.bin'):
            model = word2vec.Word2Vec.load_word2vec_format(model_path, binary=True)
            print(f"  格式: Word2Vec Binary")

        else:
            # 默认尝试pickle
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"  格式: 自动检测 (Pickle)")

        # 检查模型属性 - 兼容不同版本的gensim
        # 首先尝试获取vector_size
        vector_size = None

        # 方法1: 标准API
        if vector_size is None and hasattr(model.wv, 'vector_size'):
            try:
                vector_size = model.wv.vector_size
            except:
                pass

        # 方法2: 直接从model
        if vector_size is None and hasattr(model, 'vector_size'):
            try:
                vector_size = model.vector_size
            except:
                pass

        # 方法3: 旧版本使用syn0
        if vector_size is None and hasattr(model.wv, 'syn0'):
            try:
                vector_size = model.wv.syn0.shape[1]
                print(f"  检测方法: syn0 (旧版本gensim)")
            except Exception as e:
                print(f"  警告: 从syn0获取维度失败: {e}")

        # 方法4: 新版本使用vectors
        if vector_size is None and hasattr(model.wv, 'vectors'):
            try:
                vector_size = model.wv.vectors.shape[1]
            except:
                pass

        # 方法5: 尝试获取第一个向量
        if vector_size is None:
            try:
                # 尝试多种方式获取第一个key
                if hasattr(model.wv, 'index2word') and len(model.wv.index2word) > 0:
                    first_key = model.wv.index2word[0]
                    vector_size = len(model.wv[first_key])
                elif hasattr(model.wv, 'index_to_key') and len(model.wv.index_to_key) > 0:
                    first_key = model.wv.index_to_key[0]
                    vector_size = len(model.wv[first_key])
                elif hasattr(model.wv, 'vocab'):
                    first_key = next(iter(model.wv.vocab.keys()))
                    vector_size = len(model.wv[first_key])
            except Exception as e:
                print(f"  警告: 从向量获取维度失败: {e}")

        # 最后的fallback
        if vector_size is None:
            vector_size = 300  # 默认假设300维
            print(f"  警告: 无法自动检测维度，使用默认值 300")

        # 兼容旧版本gensim (使用index2word) 和新版本 (使用index_to_key)
        try:
            if hasattr(model.wv, 'index_to_key'):
                vocab_size = len(model.wv.index_to_key)
                sample_words = list(model.wv.index_to_key)[:5]
            elif hasattr(model.wv, 'index2word'):
                vocab_size = len(model.wv.index2word)
                sample_words = list(model.wv.index2word)[:5]
            elif hasattr(model.wv, 'vocab'):
                vocab_size = len(model.wv.vocab)
                sample_words = list(model.wv.vocab.keys())[:5]
            elif hasattr(model.wv, 'key_to_index'):
                vocab_size = len(model.wv.key_to_index)
                sample_words = list(model.wv.key_to_index.keys())[:5]
            elif hasattr(model.wv, 'syn0'):
                # 非常旧的版本
                vocab_size = model.wv.syn0.shape[0]
                sample_words = ["(使用旧版本API)"]
            elif hasattr(model.wv, 'vectors'):
                vocab_size = model.wv.vectors.shape[0]
                sample_words = ["(无法获取词汇名称)"]
            else:
                vocab_size = 0
                sample_words = []
        except Exception as e:
            print(f"  警告: 无法获取词汇表信息: {e}")
            vocab_size = 0
            sample_words = []

        print(f"  嵌入维度: {vector_size}")
        print(f"  词汇表大小: {vocab_size:,} 个片段")

        if sample_words:
            print(f"  样本片段ID: {sample_words}")

        # 创建兼容性包装器
        model_wrapper = GensimModelWrapper(model)
        print(f"  ✓ 已创建gensim兼容性包装器")

        return model_wrapper, vector_size

    except Exception as e:
        print(f"\n错误: 无法加载模型")
        print(f"详细信息: {str(e)}")
        print(f"\n请确保:")
        print(f"  1. 模型文件格式正确 (.pkl, .model, 或 .bin)")
        print(f"  2. 使用与训练时相同的gensim版本")
        print(f"  3. 文件未损坏")
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
    print(f"实现方式: {'官方mol2vec库' if HAS_MOL2VEC else '备用实现'}")

    # 检查文件是否存在
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")

    # 加载Mol2Vec模型
    model_wrapper, vector_size = load_mol2vec_model(model_path)

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
    print(f"  - 方法: Mol2Vec ({'官方实现' if HAS_MOL2VEC else '备用实现'})")
    print(f"  - 片段半径: {radius}")
    print(f"  - 嵌入维度: {vector_size}")
    print(f"  - 聚合方式: 平均片段嵌入")

    embeddings = np.zeros((num_mols, vector_size), dtype=np.float32)
    invalid_indices = []
    invalid_smiles = []

    for idx, smiles in enumerate(tqdm(smiles_list, desc="计算Mol2Vec", ncols=80)):
        if pd.isna(smiles) or smiles == '':
            invalid_indices.append(idx)
            invalid_smiles.append("(empty)")
            continue

        emb = compute_mol2vec_embedding(smiles, model_wrapper, radius=radius, vector_size=vector_size)

        if emb is None:
            invalid_indices.append(idx)
            invalid_smiles.append(smiles[:50])  # 记录前50个字符
        else:
            embeddings[idx] = emb

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
        description='为MOSES数据集预计算Mol2Vec分子嵌入（使用官方mol2vec库）',
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

依赖安装:
  # 安装官方mol2vec库（推荐）
  pip install git+https://github.com/samoturk/mol2vec

  # 或者手动安装
  git clone https://github.com/samoturk/mol2vec.git
  cd mol2vec
  pip install -e .

  # 基础依赖
  pip install gensim==4.3.0 rdkit pandas numpy tqdm

注意事项:
  1. 推荐安装官方mol2vec库以获得最佳兼容性
  2. 如果没有安装mol2vec库，会使用备用实现（可能精度略低）
  3. 模型文件路径必须指向有效的预训练模型
  4. 支持的模型格式: .pkl (pickle), .model (gensim), .bin (word2vec)
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
    except ImportError:
        print(f"错误: 未安装RDKit!")
        print(f"请安装: conda install -c conda-forge rdkit")
        sys.exit(1)

    try:
        from gensim.models import word2vec
    except ImportError:
        print(f"错误: 未安装Gensim!")
        print(f"请安装: pip install gensim==4.3.0")
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
