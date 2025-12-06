# Synthcity on PyTorch 2.7 + CUDA 12.8 Test

Minimal test project for Synthcity TVAE compatibility with PyTorch 2.7.0 and CUDA 12.8 on NVIDIA RTX 5070 Ti (sm_120).

## Purpose

Test if:
1. PyTorch 2.7.0 + CUDA 12.8 supports RTX 5070 Ti (Blackwell sm_120)
2. Synthcity TVAE works with PyTorch 2.7+ (despite requiring torch<2.3)
3. GPU acceleration works for training and generation
4. Model saving/loading works (pickle format used by sdpype)

## Strategy

Uses uv's `override-dependencies` to force PyTorch 2.7.0 installation, bypassing Synthcity's constraint of `torch<2.3` (from tsai dependency).

**Expected behavior**: Some Synthcity features (especially tsai-based time series models) may break, but basic TVAE should work if PyTorch API hasn't changed significantly.

## Hardware Target

- **GPU**: NVIDIA GeForce RTX 5070 Ti
- **Compute Capability**: 12.0 (sm_120, Blackwell architecture)
- **Driver**: CUDA 13.0+ compatible

## Setup

```bash
# Install dependencies with uv (will override PyTorch constraints)
uv sync

# Verify GPU is detected
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## Run Test

```bash
uv run python test_synthcity_tvae.py
```

## Expected Output

- PyTorch detects RTX 5070 Ti
- No sm_120 compatibility errors
- Synthcity TVAE initializes
- TVAE trains on GPU
- Model saves successfully
- Model loads successfully
- Generation works on GPU
- Synthetic data matches schema

## Known Risks

**Warning**: This setup bypasses Synthcity's PyTorch version requirements. Potential issues:

- Time series models (using tsai) will likely fail
- Some PyTorch API changes in 2.7 may cause errors
- Only testing TVAE - other models may break

## Dependencies

- Python 3.11
- PyTorch 2.7.0 (CUDA 12.8) - FORCED via override-dependencies
- Synthcity 0.2.x
- pandas, numpy

## References

- [Synthcity GitHub](https://github.com/vanderschaarlab/synthcity)
- [PyTorch 2.7 Release](https://pytorch.org/)
- [uv override-dependencies](https://docs.astral.sh/uv/)
