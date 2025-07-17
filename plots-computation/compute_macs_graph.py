import hydra
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import wandb
from omegaconf import DictConfig

# Define nice colors for each model (Plotly hex codes are fine)
model_colors = {
    'l-conv-sym': '#1f77b4',  # Blue
    'l-mamba-sym': '#ff7f0e',  # Orange
    'l-transformer-sym': '#2ca02c',  # Green
    'l-liquid-sym': '#d62728'  # Red
}

# Define nice names for legend
model_names = {
    'l-conv-sym': 'L-TCN',
    'l-mamba-sym': 'L-Mamba',
    'l-transformer-sym': 'L-Transformer',
    'l-liquid-sym': 'L-Liquid'
}


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
    print(df)
    x_min = df['context_size_ms'].min()
    x_max = df['context_size_ms'].max()

    # Sort models for consistent order
    models = [m for m in model_names if m in df['model'].unique()]
    x_vals = np.arange(1000, 0, -100)

    # 1. Linear scale plot
    fig = go.Figure()
    for model in models:
        model_data = df[df['model'] == model].sort_values('context_size_ms')
        fig.add_trace(
            go.Scatter(
                x=model_data['context_size_ms'],
                y=model_data['gmacs'],
                mode='lines+markers',
                name=model_names.get(model, model),
                line=dict(color=model_colors.get(model, None), width=3),
                marker=dict(size=10),
                opacity=0.95
            )
        )

    fig.update_layout(
        title='Computational Cost vs Context Size for Large Streaming Models',
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
    fig.write_image("plots_computation/macs_vs_context_size.png", scale=2)
    fig.write_html("plots_computation/macs_vs_context_size.html")
    print("Saved plot to plots_computation/macs_vs_context_size.png/.html")

    fig.add_shape(
        type='line',
        x0=x_min,
        x1=x_max,
        y0=256,
        y1=256,
        line=dict(color='red', dash='dash'),
        layer="above"
    )

    fig.add_shape(
        type='line',
        x0=x_min,
        x1=x_max,
        y0=2048,
        y1=2048,
        line=dict(color='red', dash='dash'),
        layer="above"
    )

    fig.add_annotation(
        x=0,  # far left of plotting area
        y=256,
        xref="paper",
        yref="y",
        xanchor="left",
        yanchor="bottom",
        text="256 GMAC/s: Ethos U-85 LOW",
        showarrow=False,
        font=dict(color="red", size=18)
    )

    fig.add_annotation(
        x=1,  # far right of plotting area
        y=2048,
        xref="paper",
        yref="y",
        xanchor="right",
        yanchor="bottom",
        text="2048 GMAC/s: Ethos U-85 MAX",
        showarrow=False,
        font=dict(color="red", size=18)
    )

    # Show plot in browser (interactive)
    fig.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    for model in models:
        model_data = df[df['model'] == model]
        print(f"\n{model_names.get(model, model)}:")
        print(
            f"  Min GMACs: {model_data['gmacs'].min():.3f} (at {model_data.loc[model_data['gmacs'].idxmin(), 'context_size_ms']:.0f}ms)")
        print(
            f"  Max GMACs: {model_data['gmacs'].max():.3f} (at {model_data.loc[model_data['gmacs'].idxmax(), 'context_size_ms']:.0f}ms)")
        print(f"  Range: {model_data['gmacs'].max() - model_data['gmacs'].min():.3f} GMACs")
        pct_increase = ((model_data['gmacs'].max() - model_data['gmacs'].min()) / model_data['gmacs'].min()) * 100
        print(f"  Increase: {pct_increase:.1f}%")


if __name__ == "__main__":
    Path("plots_computation").mkdir(exist_ok=True)
    main()
