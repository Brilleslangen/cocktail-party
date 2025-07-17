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
    models = [m for m in df['model'].unique() if "test" not in m.lower()]
    x_vals = np.arange(1000, 0, -100)

    # 1. Linear scale plot
    fig = go.Figure()
    for model in models:
        model_data = df[df['model'] == model].sort_values('context_size_ms')
        model_type = get_model_type(model)
        fig.add_trace(
            go.Scatter(
                x=model_data['context_size_ms'],
                y=model_data['gmacs'],
                mode='lines+markers',
                name=model.capitalize(),  # Capitalize first letter
                line=dict(color=type_colors.get(model_type, '#888'), width=3),
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

    fig.add_shape(
        type='line',
        x0=0,
        x1=1,
        y0=256,
        y1=256,
        xref='paper',  # << key!
        yref='y',
        line=dict(color='red', dash='dash'),
        layer="above"
    )

    fig.add_shape(
        type='line',
        x0=0,
        x1=1,
        y0=2048,
        y1=2048,
        xref='paper',
        yref='y',
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

    fig.add_shape(
        type="rect",
        x0=0,
        x1=1,
        xref="paper",
        y0=256,
        y1=3000,
        yref="y",
        fillcolor="rgba(255, 0, 0, 0.1)",
        line=dict(width=0),
        layer="below"
    )

    fig.add_shape(
        type="rect",
        x0=0,
        x1=1,
        xref="paper",
        y0=2048,
        y1=3000,
        yref="y",
        fillcolor="rgba(255, 0, 0, 0.2)",
        line=dict(width=0),
        layer="below"
    )

    fig.write_image("plots_computation/macs_vs_context_size.png", scale=2)
    fig.write_html("plots_computation/macs_vs_context_size.html")
    print("Saved plot to plots_computation/macs_vs_context_size.png/.html")

    # Show plot in browser (interactive)
    fig.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    for model in models:
        model_data = df[df['model'] == model]
        print(f"\n{models.get(model, model)}:")
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
