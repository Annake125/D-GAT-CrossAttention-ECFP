#!/usr/bin/env python
"""
Inspect the actual structure of the gensim model
"""
import pickle
import sys
import os

# Get model path from argument or use default
if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    model_path = './mol2vec_pretrained/model_300dim.pkl'

if not os.path.exists(model_path):
    print(f"Error: Model file not found: {model_path}")
    print(f"Current directory: {os.getcwd()}")
    print("\nUsage: python inspect_model.py <path_to_model.pkl>")
    sys.exit(1)

# Load model
print(f"Loading model from: {model_path}")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

print("\n=== Model Type ===")
print(f"Type: {type(model)}")
print(f"WV Type: {type(model.wv)}")

print("\n=== Model Attributes ===")
model_attrs = [attr for attr in dir(model) if not attr.startswith('_')]
print(f"Model attributes: {model_attrs[:20]}")

print("\n=== WV Attributes ===")
wv_attrs = [attr for attr in dir(model.wv) if not attr.startswith('_')]
print(f"WV attributes: {wv_attrs}")

print("\n=== Checking Key Attributes ===")
attrs_to_check = [
    'key_to_index', 'index_to_key', 'vocab', 'index2word',
    'index2entity', 'syn0', 'vectors', 'syn0norm', 'vectors_norm'
]

for attr in attrs_to_check:
    has_it = hasattr(model.wv, attr)
    print(f"  hasattr(model.wv, '{attr}'): {has_it}")
    if has_it:
        try:
            val = getattr(model.wv, attr)
            if hasattr(val, '__len__'):
                print(f"    -> Length: {len(val)}")
            if hasattr(val, 'shape'):
                print(f"    -> Shape: {val.shape}")
            if hasattr(val, 'keys'):
                keys = list(val.keys())[:5]
                print(f"    -> Sample keys: {keys}")
            if isinstance(val, list) and len(val) > 0:
                print(f"    -> First 5 items: {val[:5]}")
        except Exception as e:
            print(f"    -> Error accessing: {e}")

print("\n=== Checking Model Direct Attributes ===")
for attr in ['wv', 'vocabulary', 'syn0', 'syn1', 'trainables']:
    has_it = hasattr(model, attr)
    print(f"  hasattr(model, '{attr}'): {has_it}")

print("\n=== Trying to Access Vocabulary ===")
# Try different methods to get the vocabulary
try:
    # Method 1: Check if there's a vocab attribute anywhere
    if hasattr(model, 'vocabulary'):
        print("✓ Found model.vocabulary")
        if hasattr(model.vocabulary, 'index2word'):
            print(f"  - index2word length: {len(model.vocabulary.index2word)}")
            print(f"  - Sample: {model.vocabulary.index2word[:5]}")
except Exception as e:
    print(f"✗ model.vocabulary failed: {e}")

try:
    # Method 2: Check wv.vocab
    if hasattr(model.wv, 'vocab') and model.wv.vocab is not None:
        vocab = model.wv.vocab
        print(f"✓ Found model.wv.vocab (type: {type(vocab)})")
        if hasattr(vocab, 'keys'):
            keys = list(vocab.keys())[:5]
            print(f"  - Sample keys: {keys}")
            # Try to get the index for first key
            first_key = keys[0]
            print(f"  - vocab[{first_key}]: {vocab[first_key]}")
except Exception as e:
    print(f"✗ model.wv.vocab access failed: {e}")

try:
    # Method 3: Direct __dict__ inspection
    print("\n=== model.wv.__dict__ keys ===")
    dict_keys = list(model.wv.__dict__.keys())
    print(f"Keys: {dict_keys}")

    for key in dict_keys:
        val = model.wv.__dict__[key]
        print(f"\n  {key}:")
        print(f"    Type: {type(val)}")
        if hasattr(val, '__len__'):
            try:
                print(f"    Length: {len(val)}")
            except:
                pass
        if hasattr(val, 'shape'):
            print(f"    Shape: {val.shape}")
        if hasattr(val, 'keys'):
            try:
                sample = list(val.keys())[:3]
                print(f"    Sample keys: {sample}")
            except:
                pass
        if isinstance(val, list) and len(val) > 0 and len(val) < 100:
            print(f"    Values: {val[:5]}")
except Exception as e:
    print(f"✗ __dict__ inspection failed: {e}")
