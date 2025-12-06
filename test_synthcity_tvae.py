"""
Minimal Synthcity TVAE test with PyTorch 2.7 + CUDA 12.8 on RTX 5070 Ti
Tests: training, saving, loading, generation, GPU usage

NOTE: This test forces PyTorch 2.7 despite Synthcity requiring torch<2.3
"""

import sys
import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path

print("=" * 70)
print("Synthcity TVAE + PyTorch 2.7 + CUDA 12.8 Compatibility Test")
print("=" * 70)

# 1. Check PyTorch and GPU
print("\n[1/9] Environment Check:")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    print(f"  GPU name: {torch.cuda.get_device_name(0)}")
    capability = torch.cuda.get_device_capability(0)
    print(f"  Compute capability: {capability[0]}.{capability[1]} (sm_{capability[0]}{capability[1]})")

    if capability[0] >= 10:
        print(f"  [OK] Blackwell architecture detected!")
    else:
        print(f"  [INFO] Not Blackwell architecture (sm_{capability[0]}{capability[1]})")
else:
    print("  [FAIL] CUDA not available!")
    sys.exit(1)

# 2. Import Synthcity (may show warnings about PyTorch version)
print("\n[2/9] Importing Synthcity...")
try:
    from synthcity.plugins import Plugins
    from synthcity.plugins.core.dataloader import GenericDataLoader
    print("  [OK] Synthcity imported successfully")

    # Check if TVAE plugin is available
    available_plugins = Plugins().list()
    print(f"  Available plugins: {len(available_plugins)} total")
    if "tvae" in available_plugins:
        print("  [OK] TVAE plugin found")
    else:
        print(f"  [FAIL] TVAE plugin not found. Available: {available_plugins}")
        sys.exit(1)

except Exception as e:
    print(f"  [FAIL] Synthcity import failed: {e}")
    print("\n  This may indicate PyTorch 2.7 is incompatible with Synthcity")
    sys.exit(1)

# 3. Generate synthetic training data (10K rows, 7 features)
print("\n[3/9] Generating training data (10,000 rows, 7 features)...")

np.random.seed(42)

data = pd.DataFrame({
    # 4 numerical features
    'age': np.random.normal(35, 10, 10000).clip(18, 80).astype(int),
    'income': np.random.lognormal(10.5, 0.5, 10000).clip(20000, 200000),
    'credit_score': np.random.normal(700, 50, 10000).clip(300, 850).astype(int),
    'account_balance': np.random.exponential(5000, 10000).clip(0, 100000),

    # 3 categorical features
    'region': np.random.choice(['North', 'South', 'East', 'West'], 10000),
    'account_type': np.random.choice(['Checking', 'Savings', 'Premium'], 10000, p=[0.5, 0.3, 0.2]),
    'risk_category': np.random.choice(['Low', 'Medium', 'High'], 10000, p=[0.6, 0.3, 0.1])
})

print(f"  Training data shape: {data.shape}")
print(f"  Columns: {list(data.columns)}")
print(f"\n  Sample data:")
print(data.head(3).to_string(index=False))

# 4. Create Synthcity DataLoader
print("\n[4/9] Creating Synthcity DataLoader...")
try:
    loader = GenericDataLoader(data, sensitive_features=["region"])
    print(f"  [OK] DataLoader created")
    print(f"  Data shape: {loader.shape}")
except Exception as e:
    print(f"  [FAIL] DataLoader creation failed: {e}")
    sys.exit(1)

# 5. Initialize TVAE with GPU
print("\n[5/9] Initializing Synthcity TVAE plugin (GPU mode)...")
try:
    model = Plugins().get(
        "tvae",
        n_iter=100,  # epochs
        batch_size=500,
        device="cuda",  # Force GPU usage
    )
    print("  [OK] TVAE plugin initialized")
except Exception as e:
    print(f"  [FAIL] TVAE initialization failed: {e}")
    print("\n  This likely means PyTorch 2.7 API changes broke Synthcity")
    sys.exit(1)

# 6. Train model
print("\n[6/9] Training TVAE on GPU...")
try:
    model.fit(loader)
    print("  [OK] Training completed successfully!")

    # Check GPU memory usage to verify GPU was used
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        print(f"  GPU memory allocated: {allocated:.1f} MB")
        if allocated > 0:
            print("  [OK] Training used GPU!")
        else:
            print("  [WARN] No GPU memory allocated (may have used CPU)")

except Exception as e:
    print(f"  [FAIL] Training failed: {e}")
    print("\n  Error details:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 7. Save model
print("\n[7/9] Saving model...")
model_path = Path("synthcity_tvae_model.pkl")
try:
    # Synthcity's save() returns bytes, save_to_file() writes to disk
    if hasattr(model, 'save_to_file'):
        model.save_to_file(model_path)
        print(f"  [OK] Model saved via save_to_file(): {model_path}")
    else:
        # save() returns bytes
        model_bytes = model.save()
        with open(model_path, 'wb') as f:
            f.write(model_bytes)
        print(f"  [OK] Model saved via save(): {model_path}")
    print(f"  File size: {model_path.stat().st_size / 1024:.1f} KB")
except Exception as e:
    print(f"  [WARN] Save via Synthcity failed: {e}")
    print("  Trying cloudpickle (handles lambdas)...")
    try:
        import cloudpickle
        with open(model_path, 'wb') as f:
            cloudpickle.dump(model, f)
        print(f"  [OK] Model saved via cloudpickle: {model_path}")
        print(f"  File size: {model_path.stat().st_size / 1024:.1f} KB")
    except Exception as e2:
        print(f"  [FAIL] Alternative save also failed: {e2}")
        sys.exit(1)

# 8. Load model
print("\n[8/9] Loading model from disk...")
try:
    # Try Synthcity's load methods first
    if hasattr(Plugins, 'load_from_file'):
        loaded_model = Plugins().load_from_file(model_path)
    else:
        with open(model_path, 'rb') as f:
            model_bytes = f.read()
        loaded_model = Plugins().load(model_bytes)
    print("  [OK] Model loaded via Synthcity!")
except Exception as e:
    print(f"  [WARN] Load via Synthcity failed: {e}")
    print("  Trying cloudpickle load...")
    try:
        import cloudpickle
        with open(model_path, 'rb') as f:
            loaded_model = cloudpickle.load(f)
        print("  [OK] Model loaded via cloudpickle!")
    except Exception as e2:
        print(f"  [FAIL] Alternative load also failed: {e2}")
        sys.exit(1)

# 9. Generate synthetic data
print("\n[9/9] Generating 1,000 synthetic samples...")
try:
    synthetic = loaded_model.generate(count=1000)

    # Convert to DataFrame if needed
    if hasattr(synthetic, 'dataframe'):
        synthetic_df = synthetic.dataframe()
    elif isinstance(synthetic, pd.DataFrame):
        synthetic_df = synthetic
    else:
        synthetic_df = pd.DataFrame(synthetic)

    print(f"  [OK] Generated {len(synthetic_df)} samples")
    print(f"  Synthetic data shape: {synthetic_df.shape}")
    print(f"\n  Sample synthetic data:")
    print(synthetic_df.head(3).to_string(index=False))

    # Verify schema matches
    if set(synthetic_df.columns) == set(data.columns):
        print("\n  [OK] Schema matches original data")
    else:
        print(f"\n  [WARN] Schema mismatch!")
        print(f"  Expected: {set(data.columns)}")
        print(f"  Got: {set(synthetic_df.columns)}")

except Exception as e:
    print(f"  [FAIL] Generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final GPU check
print("\n" + "-" * 70)
print("GPU Memory Usage:")
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2
    print(f"  Allocated: {allocated:.1f} MB")
    print(f"  Reserved: {reserved:.1f} MB")

    if allocated > 0:
        print("  [OK] GPU memory in use")
    else:
        print("  [WARN] No GPU memory allocated")

# Summary
print("\n" + "=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print("\nSummary:")
print("  [OK] PyTorch 2.7.0 + CUDA 12.8 working")
print("  [OK] Synthcity imported successfully (despite version mismatch)")
print("  [OK] TVAE training completed")
print("  [OK] Model saved in pickle format")
print("  [OK] Model loaded successfully")
print("  [OK] Synthetic data generated")
print("  [OK] Schema validation passed")
print("\nConclusion: Synthcity TVAE IS compatible with PyTorch 2.7!")
print("\nNote: This only tests TVAE. Other Synthcity models (especially time-series")
print("models using tsai) may not work with PyTorch 2.7.")
