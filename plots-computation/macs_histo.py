import numpy as np
import plotly.graph_objects as go

# Example dictionary: {"Model Name": MACs in GMAC/s}
# mac_dict = {
#     "S-conv-sym": 0.99517,
#     "S-transformer-sym": 0.80942,
#     "S-mamba-sym": 0.82096,
#     "S-liquid-sym": 0.7786,
#
#     "S-mamba-short-asym": 1.09,
#
#     "L-transformer-sym": 14.87,
#     "L-mamba-sym": 14.76,
#     "L-liquid-sym": 15.06,
#
#     "S-conv-asym": 4.90,
#     "S-transformer-asym": 4.91,
#     "S-mamba-asym": 4.90,
#     "S-liquid-asym": 4.64,
# }

sym_mac_dict = {
    "S-conv-sym": 0.99517,
    "S-transformer-sym": 0.80942,
    "S-mamba-sym": 0.82096,
    "S-liquid-sym": 0.7786,
}
short_mac_dict = {
    "S-mamba-short-asym": 1.09,
}
large_mac_dict = {
    "L-transformer-sym": 14.87,
    "L-mamba-sym": 14.76,
    "L-liquid-sym": 15.06,
}
asym_mac_dict = {
    "S-conv-asym": 4.90,
    "S-transformer": 4.91,
    "S-mamba-asym": 4.90,
    "S-liquid-asym": 4.64,
}
# Two spacers for more visible group separation
spacer = {"": None}
double_spacer = {"": None, " ": None}

# Merge for plotting with double spacers
plot_dict = {}
plot_dict.update(sym_mac_dict)
plot_dict.update(double_spacer)
plot_dict.update(short_mac_dict)
plot_dict.update(double_spacer)
plot_dict.update(large_mac_dict)
plot_dict.update(double_spacer)
plot_dict.update(asym_mac_dict)

x_labels = list(plot_dict.keys())
mac_values = [v if v is not None else np.nan for v in plot_dict.values()]


def get_model_type(model_name):
    if "conv" in model_name:
        return "Conv"
    if "transformer" in model_name:
        return "Transformer"
    if "mamba" in model_name:
        return "Mamba"
    if "liquid" in model_name:
        return "Liquid"
    return "Other"


model_type_colors = {
    "Conv": "#C565C7",
    "Transformer": "#E57439",
    "Mamba": "#A0C75C",
    "Liquid": "#5BC5DB",
    "Other": "gray"
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
    font=dict(color="red", size=18)  # Larger text size here!
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
    title='MAC/s per Model',
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
