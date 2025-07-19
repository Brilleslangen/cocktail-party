import numpy as np
import plotly.graph_objects as go

# =================================
# | Model              |   GMAC/s |
# |--------------------|----------|
# | l-conv-sym         | 15.66538 |
# | l-mamba-sym        | 14.76449 |
# | l-transformer-sym  | 14.87042 |
# | l-liquid-sym       | 15.05877 |
# | l-conv-asym        | 89.29738 |
# | l-mamba-asym       | 88.55624 |
# | l-transformer-asym | 89.55902 |
# | l-liquid-asym      | 90.32193 |
# | m-conv-sym         |  4.21003 |
# | m-mamba-sym        |  3.81629 |
# | m-transformer-sym  |  3.83464 |
# | m-liquid-sym       |  3.81812 |
# | m-conv-asym        | 22.89739 |
# | m-mamba-asym       | 22.86703 |
# | m-transformer-asym | 23.16070 |
# | m-liquid-asym      | 22.87797 |
# | s-conv-sym         |  0.99517 |
# | s-mamba-sym        |  0.82096 |
# | s-transformer-sym  |  0.80942 |
# | s-liquid-sym       |  0.77868 |
# | s-conv-asym        |  4.89573 |
# | s-mamba-asym       |  4.89502 |
# | s-transformer-asym |  4.90679 |
# | s-liquid-asym      |  4.64133 |
# | s-mamba-short-asym |  1.09256 |
# ==================================

# === Large models (L) ===
large_sym_mac_dict = {
    "L-conv-sym":         15.66538,
    "L-mamba-sym":        14.76449,
    "L-transformer-sym":  14.87042,
    "L-liquid-sym":       15.05877,
}
large_asym_mac_dict = {
    "L-conv-asym":        89.29738,
    "L-mamba-asym":       88.55624,
    "L-transformer-asym": 89.55902,
    "L-liquid-asym":      90.32193,
}

# === Medium models (M) ===
medium_sym_mac_dict = {
    "M-conv-sym":         4.21003,
    "M-mamba-sym":        3.81629,
    "M-transformer-sym":  3.83464,
    "M-liquid-sym":       3.81812,
}
medium_asym_mac_dict = {
    "M-conv-asym":        22.89739,
    "M-mamba-asym":       22.86703,
    "M-transformer-asym": 23.16070,
    "M-liquid-asym":      22.87797,
}

# === Small models (S) ===
small_sym_mac_dict = {
    "S-conv-sym":         0.99517,
    "S-mamba-sym":        0.82096,
    "S-transformer-sym":  0.80942,
    "S-liquid-sym":       0.77868,
}
small_asym_mac_dict = {
    "S-conv-asym":        4.89573,
    "S-mamba-asym":       4.89502,
    "S-transformer-asym": 4.90679,
    "S-liquid-asym":      4.64133,
}
small_short_mac_dict = {
    "S-mamba-short-asym": 1.09256,
}

# Merge for plotting with double spacers
plot_dict = {}
plot_dict.update(small_sym_mac_dict)
plot_dict.update(small_short_mac_dict)
plot_dict.update({"": None})
plot_dict.update(small_asym_mac_dict)
plot_dict.update({"  ": None})
plot_dict.update(medium_sym_mac_dict)
plot_dict.update({" ": None})
plot_dict.update(medium_asym_mac_dict)
plot_dict.update({"   ": None})
plot_dict.update(large_sym_mac_dict)

x_labels = list(plot_dict.keys())
mac_values = [v if v is not None else np.nan for v in plot_dict.values()]



def get_model_type(model_name):
    if "conv" in model_name:
        return "conv"
    if "transformer" in model_name:
        return "transformer"
    if "mamba" in model_name:
        return "mamba"
    if "liquid" in model_name:
        return "liquid"
    return "other"


model_type_colors = {
    "conv": "#b8bc1b",
    "transformer": "#379393",
    "mamba": "#5b5bd3",
    "liquid": "#d35959",
    "other": "gray"
}

bar_colors = [
    model_type_colors[get_model_type(label)]
    if label.strip() else "rgba(0,0,0,0)"  # transparent for all kinds of empty spacers
    for label in x_labels
]

fig = go.Figure()

fig.add_trace(go.Bar(
    x=x_labels,
    y=mac_values,
    name='MAC/s',
    marker_color=bar_colors
))

# Add horizontal line at 2000 GMAC/s
fig.add_shape(
    type='line',
    x0=-0.5, x1=len(x_labels) - 0.5,
    y0=2000, y1=2000,
    line=dict(color='red', dash='dash'),
)

fig.add_annotation(
    x=0,
    y=5,
    xanchor="left",
    yanchor="bottom",
    xshift=-15,
    text="3.85 GMAC/s: DEEPSONIC chip deployed in current hearing aid",
    showarrow=False,
    font=dict(color="darkred", size=18)  # Larger text size here!
)

fig.add_shape(
    type="rect",
    x0=-0.5,
    x1=len(x_labels) - 0.5,
    y0=3.85,
    y1=max(mac_values) * 1.3,
    fillcolor="rgba(255, 0, 0, 0.15)",
    line=dict(width=0),
    layer="below"
)

fig.add_shape(
    type='line',
    x0=-0.5,
    x1=len(x_labels) - 0.5,
    y0=3.85,
    y1=3.85,
    line=dict(color='red', dash='dash'),
    layer="above"
)

fig.update_layout(
    title={
            'text': "MAC/s Histogram for All Model Variants",
            'y':0.93,  # Vertical position (0 is bottom, 1 is top)
            'x':0.5,   # Center horizontally
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=26)
        },
    xaxis_title='Model',
    yaxis_title='GMAC/s',
    yaxis=dict(range=[0, max(mac_values) * 1.3]),
    bargap=0.2,
    font=dict(size=18),  # All text larger
    title_font=dict(size=24),  # Chart title larger
    xaxis_title_font=dict(size=20),
    yaxis_title_font=dict(size=20)
)

fig.show()

if __name__ == "__main__":
    fig.write_html("macs_histogram.html")
    fig.write_image("macs_histogram.png")
