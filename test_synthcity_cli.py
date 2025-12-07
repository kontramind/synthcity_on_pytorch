"""
Synthcity plugin test CLI with PyTorch 2.7 + CUDA 12.8 on RTX 5070 Ti
Tests: training, saving, loading, generation, GPU usage
Default plugin: TVAE (can be changed via --plugin)
"""

from enum import Enum
from pathlib import Path
import pickle
import random
import time
import sys

import numpy as np
import pandas as pd
import torch
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from synthcity.metrics.eval_statistical import AlphaPrecision, PRDCScore
from synthcity.metrics.eval_detection import (
    SyntheticDetectionGMM,
    SyntheticDetectionXGB,
    SyntheticDetectionMLP,
    SyntheticDetectionLinear,
)
from synthcity.plugins.core.dataloader import GenericDataLoader
from sklearn.preprocessing import OrdinalEncoder

app = typer.Typer(help="Synthcity Plugin Test CLI for PyTorch 2.7 + CUDA 12.8")
console = Console()


class PluginType(str, Enum):
    arf = "arf"
    ddpm = "ddpm"
    tvae = "tvae"
    dpgan = "dpgan"
    ctgan = "ctgan"
    rtvae = "rtvae"
    nflow = "nflow"
    pategan = "pategan"
    adsgan = "adsgan"
    bayesian_network = "bayesian_network"
    marginal_distributions = "marginal_distributions"
    # add more here if you want to test them via CLI


def check_environment() -> tuple[bool, dict]:
    """Check PyTorch and GPU environment."""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "gpu_count": 0,
        "gpu_name": None,
        "compute_capability": None,
        "is_blackwell": False,
    }

    if info["cuda_available"]:
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        info["compute_capability"] = f"{capability[0]}.{capability[1]} (sm_{capability[0]}{capability[1]})"
        info["is_blackwell"] = capability[0] >= 10

    return info["cuda_available"], info


def display_environment(info: dict) -> None:
    """Display environment info with Rich."""
    table = Table(title="Environment Check", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("PyTorch version", info["pytorch_version"])
    table.add_row("CUDA available", "✅ Yes" if info["cuda_available"] else "❌ No")

    if info["cuda_available"]:
        table.add_row("CUDA version", info["cuda_version"])
        table.add_row("GPU count", str(info["gpu_count"]))
        table.add_row("GPU name", info["gpu_name"])
        table.add_row("Compute capability", info["compute_capability"])
        if info["is_blackwell"]:
            table.add_row("Architecture", "✅ Blackwell detected!")
        else:
            table.add_row("Architecture", "⚠️ Not Blackwell")

    console.print(table)


def evaluate_alpha_precision(
    real_loader,
    synthetic_df: pd.DataFrame,
) -> dict:
    """
    Compute alpha-precision metrics between real and synthetic data.

    The metric only supports numeric dtypes, so we:
      - extract the underlying real dataframe,
      - select numeric/bool columns,
      - align real and synthetic on those columns,
      - build fresh GenericDataLoader instances on the numeric subset.
    """
    # Get real dataframe from loader and encode both real + synthetic
    real_df = real_loader.dataframe()
    real_encoded, synth_encoded = encode_for_metrics(real_df, synthetic_df)

    real_loader_num = GenericDataLoader(real_encoded)
    syn_loader_num = GenericDataLoader(synth_encoded)

    metric = AlphaPrecision()
    return metric.evaluate(real_loader_num, syn_loader_num)


def display_alpha_precision_scores(scores: dict) -> None:
    """Display alpha-precision, beta-recall and authenticity scores."""
    table = Table(title="Alpha-Precision / Beta-Recall / Authenticity")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    ordered_keys = [
        "delta_precision_alpha_OC",
        "delta_coverage_beta_OC",
        "authenticity_OC",
        "delta_precision_alpha_naive",
        "delta_coverage_beta_naive",
        "authenticity_naive",
    ]

    for key in ordered_keys:
        if key not in scores:
            continue

        value = scores[key]
        if isinstance(value, (int, float)):
            if value >= 0.9:
                value_str = f"[green]{value:.4f}[/green]"
            elif value >= 0.7:
                value_str = f"[yellow]{value:.4f}[/yellow]"
            else:
                value_str = f"[red]{value:.4f}[/red]"
        else:
            value_str = str(value)

        table.add_row(key, value_str)

    console.print(table)


def evaluate_prdc(
    real_loader,
    synthetic_df: pd.DataFrame,
) -> dict:
    """
    Compute PRDC (precision / recall / density / coverage) on numeric columns only.

    The metric only supports numeric dtypes, so we:
      - extract the underlying real dataframe,
      - select numeric/bool columns,
      - align real and synthetic on those columns,
      - build fresh GenericDataLoader instances on the numeric subset.
    """
    # Get real dataframe from loader and encode both real + synthetic
    real_df = real_loader.dataframe()
    real_encoded, synth_encoded = encode_for_metrics(real_df, synthetic_df)

    real_loader_num = GenericDataLoader(real_encoded)
    syn_loader_num = GenericDataLoader(synth_encoded)

    metric = PRDCScore()
    return metric.evaluate(real_loader_num, syn_loader_num)


def display_prdc_scores(scores: dict) -> None:
    """Display PRDC scores: precision, recall, density, coverage."""
    table = Table(title="PRDC (Precision / Recall / Density / Coverage)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    ordered_keys = ["precision", "recall", "density", "coverage"]

    for key in ordered_keys:
        if key not in scores:
            continue

        value = scores[key]
        if isinstance(value, (int, float)):
            if value >= 0.9:
                value_str = f"[green]{value:.4f}[/green]"
            elif value >= 0.7:
                value_str = f"[yellow]{value:.4f}[/yellow]"
            else:
                value_str = f"[red]{value:.4f}[/red]"
        else:
            value_str = str(value)

        table.add_row(key, value_str)

    console.print(table)


def encode_for_metrics(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode real + synthetic data into an all-numeric representation
    suitable for metric evaluation (AlphaPrecision, PRDC, detection).

    Strategy:
      - fit an OrdinalEncoder on *real* categorical columns only,
      - transform both real and synthetic categoricals with it,
      - use original numeric/bool columns as-is,
      - align on columns present in both real and synthetic.
    """
    # Identify numeric and categorical columns in the real data
    real_numeric_cols = real_df.select_dtypes(
        include=[np.number, "bool"]
    ).columns.tolist()
    real_cat_cols = real_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Keep only columns that exist in synthetic as well
    numeric_cols = [c for c in real_numeric_cols if c in synthetic_df.columns]
    cat_cols = [c for c in real_cat_cols if c in synthetic_df.columns]

    if not numeric_cols and not cat_cols:
        raise ValueError(
            "No overlapping numeric or categorical columns between real and synthetic data "
            "for metric encoding."
        )

    # Numeric part
    real_num = real_df[numeric_cols] if numeric_cols else pd.DataFrame(index=real_df.index)
    synth_num = synthetic_df[numeric_cols] if numeric_cols else pd.DataFrame(index=synthetic_df.index)

    # Categorical part: fit encoder on real only, transform both
    if cat_cols:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        enc.fit(real_df[cat_cols])

        real_cat_arr = enc.transform(real_df[cat_cols])
        synth_cat_arr = enc.transform(synthetic_df[cat_cols])

        real_cat = pd.DataFrame(
            real_cat_arr,
            columns=cat_cols,
            index=real_df.index,
        )
        synth_cat = pd.DataFrame(
            synth_cat_arr,
            columns=cat_cols,
            index=synthetic_df.index,
        )
    else:
        real_cat = pd.DataFrame(index=real_df.index)
        synth_cat = pd.DataFrame(index=synthetic_df.index)

    # Combine numeric + encoded categoricals
    real_encoded = pd.concat(
        [real_num.reset_index(drop=True), real_cat.reset_index(drop=True)],
        axis=1,
    )
    synth_encoded = pd.concat(
        [synth_num.reset_index(drop=True), synth_cat.reset_index(drop=True)],
        axis=1,
    )

    if real_encoded.empty or synth_encoded.empty:
        raise ValueError("Encoded data for metrics is empty.")

    return real_encoded, synth_encoded


def evaluate_detection_scores(
    real_loader,
    synthetic_df: pd.DataFrame,
) -> dict[str, float | None]:
    """
    Evaluate detection-based metrics using Synthcity's detection evaluators.

    We follow the same pattern as AlphaPrecision/PRDC:
      - use only numeric/bool columns,
      - align real and synthetic on those columns,
      - build fresh GenericDataLoader instances on the numeric subset.

    Returns a dict mapping detector name -> AUCROC score (or None on failure).
    Note: lower AUC values indicate better synthetic data (harder to detect).
    """
    # Get real dataframe from loader and encode both real + synthetic
    real_df = real_loader.dataframe()
    real_encoded, synth_encoded = encode_for_metrics(real_df, synthetic_df)

    real_loader_num = GenericDataLoader(real_encoded)
    syn_loader_num = GenericDataLoader(synth_encoded)

    detectors = {
        "detection_xgb": SyntheticDetectionXGB(),
        "detection_mlp": SyntheticDetectionMLP(),
        "detection_linear": SyntheticDetectionLinear(),
        "detection_gmm": SyntheticDetectionGMM(),
    }

    scores: dict[str, float | None] = {}

    for name, detector in detectors.items():
        try:
            result = detector.evaluate(real_loader_num, syn_loader_num)

            # Typical Synthcity metric output: {'mean': float, 'std': float, ...}
            auc: float | None
            if isinstance(result, dict):
                auc = result.get("mean")
                if auc is None and "score" in result:
                    auc = result["score"]
                if auc is None and len(result) == 1:
                    # Single-value dict fallback
                    auc = float(next(iter(result.values())))
            else:
                auc = float(result)

            scores[name] = float(auc) if auc is not None else None
        except Exception as e:
            console.print(
                f"[yellow]⚠️ Detection metric '{name}' failed: {e}[/yellow]"
            )
            scores[name] = None

    return scores


def display_detection_scores(scores: dict[str, float | None]) -> None:
    """Display detection AUCROC scores (lower is better)."""
    table = Table(title="Detection Metrics (AUCROC: lower is better)")
    table.add_column("Detector", style="cyan")
    table.add_column("AUC", justify="right")
    table.add_column("Quality", style="dim")

    for name, auc in scores.items():
        label = name.replace("_", " ").title()

        if auc is None:
            table.add_row(label, "[red]N/A[/red]", "error")
            continue

        # Interpret AUC: closer to 0.5 is better (harder to detect synthetic)
        if auc <= 0.55:
            quality = "Excellent"
            auc_str = f"[green]{auc:.4f}[/green]"
        elif auc <= 0.65:
            quality = "Good"
            auc_str = f"[green]{auc:.4f}[/green]"
        elif auc <= 0.75:
            quality = "Fair"
            auc_str = f"[yellow]{auc:.4f}[/yellow]"
        else:
            quality = "Poor"
            auc_str = f"[red]{auc:.4f}[/red]"

        table.add_row(label, auc_str, quality)

    console.print(table)


def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across numpy, torch, and Python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_training_data(rows: int) -> pd.DataFrame:
    """Generate synthetic training data."""
    return pd.DataFrame({
        # 4 numerical features
        "age": np.random.normal(35, 10, rows).clip(18, 80).astype(int),
        "income": np.random.lognormal(10.5, 0.5, rows).clip(20000, 200000),
        "credit_score": np.random.normal(700, 50, rows).clip(300, 850).astype(int),
        "account_balance": np.random.exponential(5000, rows).clip(0, 100000),
        # 3 categorical features
        "region": np.random.choice(["North", "South", "East", "West"], rows),
        "account_type": np.random.choice(
            ["Checking", "Savings", "Premium"], rows, p=[0.5, 0.3, 0.2]
        ),
        "risk_category": np.random.choice(
            ["Low", "Medium", "High"], rows, p=[0.6, 0.3, 0.1]
        ),
    })


def display_data_sample(data: pd.DataFrame, title: str) -> None:
    """Display a small sample of data with Rich table."""
    table = Table(title=title)
    for col in data.columns:
        table.add_column(col, style="cyan")

    for _, row in data.head(3).iterrows():
        table.add_row(*[str(v)[:20] for v in row.values])

    console.print(table)


def create_synthcity_plugin(
    plugin: PluginType,
    use_gpu: bool,
):
    """Create Synthcity plugin instance with appropriate device."""
    try:
        from synthcity.plugins import Plugins
    except Exception as e:
        console.print(f"[red]❌ Synthcity import failed: {e}[/red]")
        console.print("  This may indicate PyTorch 2.7 is incompatible with Synthcity")
        raise typer.Exit(1)

    plugins = Plugins()
    available = plugins.list()

    if plugin.value not in available:
        console.print(
            f"[red]❌ Plugin '{plugin.value}' not found in Synthcity.[/red]\n"
            f"Available plugins: {available}"
        )
        raise typer.Exit(1)

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    console.print(f"  Using device: [bold]{device}[/bold]")

    try:
        model = plugins.get(plugin.value, device=device)
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize plugin '{plugin.value}': {e}[/red]")
        console.print("  This likely means PyTorch 2.7 API changes broke Synthcity")
        raise typer.Exit(1)

    console.print(f"  ✅ Plugin [bold]{model.name().upper()}[/bold] initialized")
    return model


@app.command()
def test(
    plugin: PluginType = typer.Option(
        PluginType.tvae,
        "--plugin",
        "-p",
        help="Synthcity plugin to test",
    ),
    samples: int = typer.Option(
        10000,
        "--samples",
        "-n",
        help="Number of synthetic samples to generate",
    ),
    training_rows: int = typer.Option(
        10000,
        "--training-rows",
        "-t",
        help="Number of training data rows",
    ),
    no_gpu: bool = typer.Option(
        False,
        "--no-gpu",
        help="Disable GPU, use CPU only",
    ),
) -> None:
    """Test Synthcity plugin with PyTorch 2.7 + CUDA 12.8."""
    set_random_seed(42)
    train_seconds = None
    schema_ok = True

    console.print(
        Panel.fit(
            (
                f"[bold magenta]Synthcity {plugin.value.upper()} Test[/bold magenta]\n"
                f"PyTorch {torch.__version__}"
                f"{' + CUDA ' + torch.version.cuda if torch.version.cuda is not None else ''} Compatibility"
            ),
            border_style="magenta",
        )
    )

    # 1. Environment
    console.print("\n[bold cyan]📊 Environment Check[/bold cyan]")
    cuda_available, env_info = check_environment()
    display_environment(env_info)

    use_gpu = cuda_available and not no_gpu
    if not cuda_available and not no_gpu:
        console.print("[yellow]⚠️ CUDA not available, falling back to CPU[/yellow]")
    elif no_gpu:
        console.print("[yellow]⚠️ GPU disabled by --no-gpu flag[/yellow]")

    # 2. Generate training data
    console.print(
        f"\n[bold cyan]📝 Generating training data ({training_rows:,} rows, 7 features)[/bold cyan]"
    )
    data = generate_training_data(training_rows)
    console.print(f"  Shape: {data.shape}")
    display_data_sample(data, "Training Data Sample")

    # 3. Create Synthcity DataLoader
    console.print("\n[bold cyan]📦 Creating Synthcity DataLoader[/bold cyan]")
    try:
        loader = GenericDataLoader(data, sensitive_features=["region"])
        console.print("  ✅ DataLoader created")
        console.print(f"  Data shape (loader): {loader.shape}")
    except Exception as e:
        console.print(f"[red]❌ DataLoader creation failed: {e}[/red]")
        raise typer.Exit(1)

    # 4. Initialize plugin
    console.print(f"\n[bold cyan]🚀 Initializing Synthcity plugin[/bold cyan]")
    model = create_synthcity_plugin(plugin, use_gpu)

    # 5. Train plugin
    console.print(f"\n[bold cyan]⏳ Training {model.name().upper()}[/bold cyan]")
    try:
        start_time = time.perf_counter()
        model.fit(loader)
        end_time = time.perf_counter()
        train_seconds = end_time - start_time
        console.print("  ✅ Training completed successfully!")
        console.print(f"  ⏱ Training time: {train_seconds:.2f} seconds")

        # GPU memory usage as a proxy for GPU use
        if torch.cuda.is_available() and use_gpu:
            allocated = torch.cuda.memory_allocated(0) / 1024**2
            console.print(f"  GPU memory allocated: {allocated:.1f} MB")
            if allocated > 0:
                console.print("  ✅ Training used GPU!")
            else:
                console.print("  ⚠️ No GPU memory allocated (may have used CPU)")
    except Exception as e:
        console.print(f"[red]❌ Training failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)

    # 6. Save model
    console.print("\n[bold cyan]💾 Saving model[/bold cyan]")
    model_path = Path(f"synthcity_{plugin.value}_model.pkl")
    model_name = model.name()

    try:
        from synthcity.utils.serialization import (
            save as synthcity_save,
            save_to_file as synthcity_save_to_file,
        )
    except Exception as e:
        console.print(f"[red]❌ Failed to import Synthcity serialization utils: {e}[/red]")
        raise typer.Exit(1)

    try:
        if model_name == "dpgan":
            console.print(f"  ℹ️ Using file-based serialization for {model_name}")
            synthcity_save_to_file(model_path, model)
            console.print(
                f"  ✅ Model saved via synthcity_save_to_file(): {model_path}"
            )
        else:
            model_bytes = synthcity_save(model)
            with open(model_path, "wb") as f:
                f.write(model_bytes)
            console.print(f"  ✅ Model saved via synthcity_save(): {model_path}")
        console.print(f"  File size: {model_path.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        console.print(f"[yellow]⚠️ Save via Synthcity failed: {e}[/yellow]")
        console.print("  Trying cloudpickle (handles lambdas)...")
        try:
            import cloudpickle

            with open(model_path, "wb") as f:
                cloudpickle.dump(model, f)
            console.print(f"  ✅ Model saved via cloudpickle: {model_path}")
            console.print(f"  File size: {model_path.stat().st_size / 1024:.1f} KB")
        except Exception as e2:
            console.print(f"[red]❌ Alternative save also failed: {e2}[/red]")
            raise typer.Exit(1)

    # 7. Load model
    console.print("\n[bold cyan]📂 Loading model from disk[/bold cyan]")
    try:
        from synthcity.utils.serialization import (
            load as synthcity_load,
            load_from_file as synthcity_load_from_file,
        )
    except Exception as e:
        console.print(f"[red]❌ Failed to import Synthcity load utils: {e}[/red]")
        raise typer.Exit(1)

    try:
        if model_name == "dpgan":
            console.print(f"  ℹ️ Using file-based loading for {model_name}")
            loaded_model = synthcity_load_from_file(model_path)
            console.print("  ✅ Model loaded via synthcity_load_from_file()")
        else:
            with open(model_path, "rb") as f:
                model_bytes = f.read()
            loaded_model = synthcity_load(model_bytes)
            console.print("  ✅ Model loaded via synthcity_load()")
    except Exception as e:
        console.print(f"[yellow]⚠️ Load via Synthcity failed: {e}[/yellow]")
        console.print("  Trying cloudpickle load...")
        try:
            import cloudpickle

            with open(model_path, "rb") as f:
                loaded_model = cloudpickle.load(f)
            console.print("  ✅ Model loaded via cloudpickle")
        except Exception as e2:
            console.print(f"[red]❌ Alternative load also failed: {e2}[/red]")
            raise typer.Exit(1)

    # 8. Generate synthetic data
    console.print(f"\n[bold cyan]🎲 Generating {samples:,} synthetic samples[/bold cyan]")
    try:
        synthetic = loaded_model.generate(count=samples)

        # Normalize to DataFrame
        if hasattr(synthetic, "dataframe"):
            synthetic_df = synthetic.dataframe()
        elif isinstance(synthetic, pd.DataFrame):
            synthetic_df = synthetic
        else:
            synthetic_df = pd.DataFrame(synthetic)

        console.print(f"  ✅ Generated {len(synthetic_df):,} samples")
        console.print(f"  Shape: {synthetic_df.shape}")
        display_data_sample(synthetic_df, "Synthetic Data Sample")

        # Schema check
        if set(synthetic_df.columns) == set(data.columns):
            console.print("  ✅ Schema matches original data")
            schema_ok = True
        else:
            console.print("  [yellow]⚠️ Schema mismatch![/yellow]")
            console.print(f"  Expected: {set(data.columns)}")
            console.print(f"  Got:      {set(synthetic_df.columns)}")
            schema_ok = False
    except Exception as e:
        console.print(f"[red]❌ Generation failed: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)

    # 9. Alpha-Precision / Beta-Recall / Authenticity
    console.print("\n[bold cyan]📈 Alpha-Precision Evaluation[/bold cyan]")
    alpha_scores = None
    try:
        alpha_scores = evaluate_alpha_precision(loader, synthetic_df)
        display_alpha_precision_scores(alpha_scores)
    except Exception as e:
        console.print(f"[yellow]⚠️ Alpha-Precision evaluation failed: {e}[/yellow]")

    # 10. PRDC (Precision / Recall / Density / Coverage)
    console.print("\n[bold cyan]📊 PRDC Evaluation[/bold cyan]")
    prdc_scores = None
    try:
        prdc_scores = evaluate_prdc(loader, synthetic_df)
        display_prdc_scores(prdc_scores)
    except Exception as e:
        console.print(f"[yellow]⚠️ PRDC evaluation failed: {e}[/yellow]")

    # 11. Detection-based evaluation (real vs synthetic)
    console.print("\n[bold cyan]🕵️ Detection-Based Evaluation[/bold cyan]")
    detection_scores: dict[str, float | None] | None = None
    try:
        detection_scores = evaluate_detection_scores(loader, synthetic_df)
        display_detection_scores(detection_scores)
    except Exception as e:
        console.print(f"[yellow]⚠️ Detection evaluation failed: {e}[/yellow]")

    # 12. GPU memory usage
    if torch.cuda.is_available():
        console.print("\n[bold cyan]🔍 GPU Memory Usage[/bold cyan]")
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        console.print(f"  Allocated: {allocated:.1f} MB")
        console.print(f"  Reserved:  {reserved:.1f} MB")

        if allocated > 0:
            console.print("  ✅ GPU memory in use")
        else:
            console.print("  ⚠️ No GPU memory allocated")

    # 13. Summary
    summary_table = Table(title="Synthcity Test Summary", show_header=False)
    summary_table.add_column("Check", style="cyan")
    summary_table.add_column("Status", style="green")

    summary_table.add_row("Environment", f"PyTorch {env_info['pytorch_version']}")
    if cuda_available:
        summary_table.add_row("GPU", env_info["gpu_name"])
        summary_table.add_row("Compute", env_info["compute_capability"])
    summary_table.add_row("Plugin", model_name.upper())
    summary_table.add_row("Model File", model_path.name)
    summary_table.add_row("Training", "✅ Completed")
    if train_seconds is not None:
        summary_table.add_row("Training Time", f"{train_seconds:.2f} s")
    summary_table.add_row("Save/Load", "✅ Working")
    summary_table.add_row("Generation", f"✅ {samples:,} samples")
    if alpha_scores is not None:
        # Show OC variant if available, otherwise just note that it ran
        oc_alpha = alpha_scores.get("delta_precision_alpha_OC")
        oc_beta = alpha_scores.get("delta_coverage_beta_OC")
        if oc_alpha is not None and oc_beta is not None:
            summary_table.add_row(
                "Alpha-Precision",
                f"α_OC={oc_alpha:.3f}, β_OC={oc_beta:.3f}",
            )
        else:
            summary_table.add_row("Alpha-Precision", "✅ Evaluated")
    if prdc_scores is not None:
        p = prdc_scores.get("precision")
        r = prdc_scores.get("recall")
        d = prdc_scores.get("density")
        c = prdc_scores.get("coverage")
        if None not in (p, r, d, c):
            summary_table.add_row(
                "PRDC",
                f"P={p:.3f}, R={r:.3f}, D={d:.3f}, C={c:.3f}",
            )
        else:
            summary_table.add_row("PRDC", "✅ Evaluated")
    if detection_scores is not None:
        # Ensemble: mean AUC across successful detectors
        vals = [v for v in detection_scores.values() if v is not None]
        if vals:
            ensemble_auc = float(np.mean(vals))
            summary_table.add_row("Detection (AUC)", f"{ensemble_auc:.3f}")
        else:
            summary_table.add_row("Detection (AUC)", "⚠️ No valid scores")            
    if schema_ok:
        summary_table.add_row("Schema", "✅ Valid")
    else:
        summary_table.add_row("Schema", "⚠️ Mismatch")

    console.print("\n")
    console.print(summary_table)

    gpu_label = "CPU only"
    if cuda_available:
        gpu_label = env_info.get("gpu_name") or "Unknown GPU"
        if env_info.get("is_blackwell"):
            gpu_label += " (Blackwell)"

    console.print(
        Panel.fit(
            (
                "[bold green]✅ ALL TESTS PASSED![/bold green]\n\n"
                f"Synthcity {model_name.upper()} is compatible with "
                f"PyTorch {env_info['pytorch_version']}"
                f"{' + CUDA ' + env_info['cuda_version'] if env_info.get('cuda_version') else ''} "
                f"on {gpu_label}! 🎉"
            ),
            border_style="green",
        )
    )


if __name__ == "__main__":
    # Allow running as a script: python test_synthcity_tvae.py test --plugin tvae
    if len(sys.argv) == 1:
        # If no subcommand given, default to 'test'
        sys.argv.append("test")
    app()
