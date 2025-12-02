#!/usr/bin/env python
"""
修复RDKit导入问题的包装脚本
设置环境变量避免加载Draw模块
"""
import os
import sys

# 设置RDKit为无头模式
os.environ['RDK_NOTHREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'

# 在导入RDKit之前设置matplotlib后端
import matplotlib
matplotlib.use('Agg')

# 现在可以安全导入
if __name__ == "__main__":
    try:
        # 测试导入
        from rdkit import Chem
        from rdkit.Chem import AllChem
        print("✓ RDKit基本功能正常")

        # 测试mol2vec（可能仍会失败）
        try:
            from mol2vec.features import mol2alt_sentence
            print("✓ mol2vec库可用")
        except ImportError as e:
            print(f"⚠️  mol2vec库导入失败: {e}")
            print("将使用备用实现")

    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)
