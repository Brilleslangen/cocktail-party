import hydra
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import wandb
from omegaconf import DictConfig

# Define nice colors for each model (Plotly hex codes are fine)
type_colors = {
    'conv': '#1f77b4',  # Blue
    'mamba': '#ff7f0e',  # Orange
    'transformer': '#2ca02c',  # Green
    'liquid': '#d62728'  # Red
}


def get_model_type(model_name):
    for t in type_colors:
        if t in model_name:
            return t
    return 'conv'  # fallback


def plot_group(df, models, group_label, filename_suffix):
    y_max = df['gmacs'].max()
    x_min = df['context_size_ms'].min()
    x_max = df['context_size_ms'].max()
    x_vals = np.arange(1000, 0, -100)

    fig = go.Figure()
    for model in models:
        model_data = df[df['model'] == model].sort_values('context_size_ms')
        model_type = get_model_type(model)
        fig.add_trace(
            go.Scatter(
                x=model_data['context_size_ms'],
                y=model_data['gmacs'],
                mode='lines+markers',
                name=model.capitalize(),
                line=dict(color=type_colors.get(model_type, '#888'), width=3),
                marker=dict(size=10),
                opacity=0.95
            )
        )

    # Shaded regions, threshold lines, and annotations (same as before)...
    fig.add_shape(
        type='line', x0=0, x1=1, y0=256, y1=256, xref='paper', yref='y',
        line=dict(color='red', dash='dash'), layer="above"
    )
    fig.add_shape(
        type='line', x0=0, x1=1, y0=2048, y1=2048, xref='paper', yref='y',
        line=dict(color='red', dash='dash'), layer="above"
    )
    fig.add_shape(
        type="rect", x0=0, x1=1, xref="paper", y0=256, y1=2048, yref="y",
        fillcolor="rgba(255, 0, 0, 0.1)", line=dict(width=0), layer="below"
    )
    fig.add_shape(
        type="rect", x0=0, x1=1, xref="paper", y0=2048, y1=y_max, yref="y",
        fillcolor="rgba(255, 0, 0, 0.15)", line=dict(width=0), layer="below"
    )
    fig.add_annotation(
        x=0.01, y=256, xref="paper", yref="y", xanchor="left", yanchor="bottom",
        text="256 GMAC/s: Ethos U-85 LOW", showarrow=False,
        font=dict(color="red", size=18)
    )
    fig.add_annotation(
        x=0.99, y=2048, xref="paper", yref="y", xanchor="right", yanchor="bottom",
        text="2048 GMAC/s: Ethos U-85 MAX", showarrow=False,
        font=dict(color="red", size=18)
    )

    fig.update_layout(
        title=f'Computational Cost vs Context Size ({group_label})',
        xaxis_title='Context Size (ms)',
        yaxis_title='Computational Cost (GMACs/s)',
        xaxis=dict(
            autorange='reversed',
            tickmode='array',
            tickvals=x_vals,
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            rangemode='tozero',
            tickfont=dict(size=14)
        ),
        legend=dict(
            font=dict(size=14),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        width=1000,
        height=700,
        margin=dict(l=60, r=40, t=70, b=60)
    )
    Path("plots_computation").mkdir(exist_ok=True)
    fig.write_image(f"plots_computation/macs_vs_context_size_{filename_suffix}.png", scale=2)
    fig.write_html(f"plots_computation/macs_vs_context_size_{filename_suffix}.html")
    print(f"Saved plot to plots_computation/macs_vs_context_size_{filename_suffix}.png/.html")
    fig.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    for model in models:
        model_data = df[df['model'] == model]
        print(f"\n{model.capitalize()}:")
        print(
            f"  Min GMACs: {model_data['gmacs'].min():.3f} (at {model_data.loc[model_data['gmacs'].idxmin(), 'context_size_ms']:.0f}ms)")
        print(
            f"  Max GMACs: {model_data['gmacs'].max():.3f} (at {model_data.loc[model_data['gmacs'].idxmax(), 'context_size_ms']:.0f}ms)")
        print(f"  Range: {model_data['gmacs'].max() - model_data['gmacs'].min():.3f} GMACs")
        pct_increase = ((model_data['gmacs'].max() - model_data['gmacs'].min()) / model_data['gmacs'].min()) * 100
        print(f"  Increase: {pct_increase:.1f}%")


@hydra.main(version_base="1.3", config_path="../configs", config_name="runs/efficiency/compute_macs_context")
def main(cfg: DictConfig):
    # Read the CSV file
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        job_type='analysis',
        name='macs_vs_context_download'
    )

    # Suppose you set this in your config:
    # cfg.csv_artifact = "your-entity/your-project/macs_vs_context:latest"
    artifact = run.use_artifact("macs_vs_context:latest", type="csv")
    artifact_dir = artifact.download(root="artifacts")
    csv_candidates = list(Path(artifact_dir).glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV file found in artifact at {artifact_dir}")
    csv_path = str(csv_candidates[0])

    df = pd.read_csv(csv_path)
    df = df[~df['model'].str.lower().str.contains("test")]
    all_models = df['model'].unique()

    l_models = [m for m in all_models if m.lower().startswith('l')]
    s_models = [m for m in all_models if m.lower().startswith('s')]

    # Plot for L models
    plot_group(df[df['model'].isin(l_models)], l_models, "Large Models (L-*)", "L")

    # Plot for S models
    plot_group(df[df['model'].isin(s_models)], s_models, "Small Models (S-*)", "S")


if __name__ == "__main__":
    Path("plots_computation").mkdir(exist_ok=True)
    main()
