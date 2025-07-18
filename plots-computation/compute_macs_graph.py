import hydra
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import wandb
from omegaconf import DictConfig
from plotly.subplots import make_subplots

model_type_colors = {
    "conv": "#b8bc1b",
    "transformer": "#379393",
    "mamba": "#5b5bd3",
    "liquid": "#d35959",
}


def get_model_type(model_name):
    for t in model_type_colors:
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

    artifact = run.use_artifact("macs_vs_context:latest", type="csv")
    artifact_dir = artifact.download(root="artifacts")
    csv_candidates = list(Path(artifact_dir).glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV file found in artifact at {artifact_dir}")
    csv_path = str(csv_candidates[0])

    df = pd.read_csv(csv_path)
    df = df[~df['model'].str.lower().str.contains("test")]
    models = [m for m in df['model'].unique() if "test" not in m.lower()]
    x_vals = np.arange(1000, 0, -100)

    # Split large and small models
    l_models = [m for m in models if m.lower().startswith('l')]
    s_models = [m for m in models if m.lower().startswith('s')]

    y_max_all = df['gmacs'].max()
    y_max_small = df[df['model'].isin(s_models)]['gmacs'].max()
    x_min = df['context_size_ms'].min()
    x_max = df['context_size_ms'].max()

    # Make custom grid for big+small (tall+short) subplot
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{'rowspan': 2}, None],  # main plot tall, subplot only in lower right
            [None, {}],
        ],
        column_widths=[0.7, 0.3],
        row_heights=[0.5, 0.5],
        horizontal_spacing=0.04,
        vertical_spacing=0.03,
        subplot_titles=["All Models - Asymmetric Windows - Chunk Length 4 ms", "Small Models Magnified"]
    )

    # --- Left Plot: All Models (tall, row=1:2, col=1) ---
    # Add proxy legend traces for type legend
    for model_type, color in model_type_colors.items():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color=color, width=3),
                name=model_type.capitalize(),
                showlegend=True
            ),
            row=1, col=1
        )

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        name="<br>",  # Or: name="&nbsp;"*20 (but HTML entities are not rendered in legend)
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=0),
        name=" " * 40,  # This will appear as a wide empty slot in the legend.
        showlegend=True
    ))

    # Main plot (all models): row=1, col=1 (spans both rows)
    for model in models:
        model_data = df[df['model'] == model].sort_values('context_size_ms')
        model_type = get_model_type(model)
        fig.add_trace(
            go.Scatter(
                x=model_data['context_size_ms'],
                y=model_data['gmacs'],
                mode='lines+markers',
                line=dict(color=model_type_colors.get(model_type, '#888'), width=3),
                marker=dict(size=10),
                opacity=0.95,
                showlegend=False
            ),
            row=1, col=1
        )

    # --- Right Plot: Only Small Models (short, row=1, col=2) ---
    for model in s_models:
        model_data = df[df['model'] == model].sort_values('context_size_ms')
        model_type = get_model_type(model)
        fig.add_trace(
            go.Scatter(
                x=model_data['context_size_ms'],
                y=model_data['gmacs'],
                mode='lines+markers',
                line=dict(color=model_type_colors.get(model_type, '#888'), width=3, dash="dot"),
                marker=dict(size=10),
                opacity=0.95,
                showlegend=False
            ),
            row=2, col=2
        )

    # --- Add threshold lines and fills ---

    fig.add_shape(
        type='line',
        x0=1060, x1=-40,
        y0=256, y1=256,
        xref='paper', yref='y',
        line=dict(color='red', dash='dash'),
        layer="above",
        row=1, col=1
    )
    fig.add_shape(
        type='line',
        x0=1063, x1=-40,
        y0=2048, y1=2048,
        xref='paper', yref='y',
        line=dict(color='red', dash='dash'),
        layer="above",
        row=1, col=1
    )
    fig.add_shape(
        type="rect",
        x0=1063,
        x1=-41,
        xref="x",
        y0=256,
        y1=3010,
        yref="y",
        fillcolor="rgba(255, 0, 0, 0.10)",
        line=dict(width=0),
        layer="below"
    )

    fig.add_shape(
        type="rect",
        x0=1063,
        x1=-41,
        xref="x",
        y0=2048,
        y1=3010,
        yref="y",
        fillcolor="rgba(255, 0, 0, 0.15)",
        line=dict(width=0),
        layer="below"
    )
    fig.add_annotation(
        x=750,
        y=265,
        xref="paper",
        yref="y",
        xshift=-200,
        xanchor="left",
        yanchor="bottom",
        text="256 GMAC/s: Ethos-U85 LOW",
        showarrow=False,
        font=dict(color="darkred", size=18),
        row=1, col=1
    )
    fig.add_annotation(
        x=445,
        y=2048,
        xref="x",
        yref="y",
        xanchor="right",
        yanchor="bottom",
        text="2048 GMAC/s: Ethos-U85 MAX",
        showarrow=False,
        font=dict(color="darkred", size=18),
        row=1, col=1,
        xshift=300
    )

    # Small models plot: row=1, col=2 (short)
    fig.add_shape(
        type='line',
        x0=1100, x1=-20,
        y0=3.85, y1=3.85,
        xref='paper', yref='y',
        line=dict(color='red', dash='dash'),
        layer="above",
        opacity=0.5,
        row=2, col=2
    )
    fig.add_annotation(
        x=800,
        y=6,
        xref="x",
        yref="y",
        xanchor="right",
        yanchor="bottom",
        text="""<span style='display: block; text-align: right;'>3.85 GMAC/s:          <br>DEEPSONIC (in use)</span>""",
        showarrow=False,
        font=dict(color="darkred", size=14),
        row=2, col=2,
        xshift=80
    )

    # --- Layout ---
    fig.update_layout(
        xaxis=dict(
            autorange='reversed',
            tickmode='array',
            tickvals=x_vals,
            tickfont=dict(size=14),
            title='Context Size (ms)',
            gridcolor='rgba(180,180,180,0.5)',
            zerolinecolor='rgba(100,100,100,0.3)',
        ),
        xaxis2=dict(
            autorange='reversed',
            tickmode='array',
            tickvals=x_vals,
            tickfont=dict(size=14),
            title='Context Size (ms)',
            gridcolor='rgba(180,180,180,0.5)',
            zerolinecolor='rgba(100,100,100,0.3)',
        ),
        yaxis=dict(
            rangemode='tozero',
            tickfont=dict(size=14),
            title='Computational Cost (GMACs/s)',
            gridcolor='rgba(180,180,180,0.5)',
            zerolinecolor='rgba(100,100,100,0.3)',

        ),
        yaxis2=dict(
            rangemode='tozero',
            tickfont=dict(size=14),
            title='',
            range=[0, y_max_small],
            gridcolor='rgba(180,180,180,0.5)',
            zerolinecolor='rgba(100,100,100,0.3)',
        ),
        legend=dict(
            font=dict(size=14),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1,
            x=0.76,
            y=0.9,
            xanchor='left',
            yanchor='top',
        ),
        width=1400,
        height=700,
        margin=dict(l=60, r=40, t=70, b=60),
    )

    Path("plots-computation/outputs").mkdir(exist_ok=True)
    fig.write_image("plots-computation/outputs/macs_vs_context_size_sbs.png", scale=2)
    fig.write_html("plots-computation/outputs/macs_vs_context_size_sbs.html")
    print("Saved plot to plots-computation/outputs/macs_vs_context_size_sbs.png/.html")
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


if __name__ == "__main__":
    Path("plots-computation").mkdir(exist_ok=True)
    main()
