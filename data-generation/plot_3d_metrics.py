import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Your data
df = pd.DataFrame({
    'Model': [
        'L-mamba', 'L-transformer', 'L-conv', 'L-liquid',
        'M-mamba', 'M-transformer', 'M-conv', 'M-liquid',
        'S-mamba', 'S-transformer', 'S-conv', 'S-liquid'
    ],
    'ESTOI': [0.730, 0.562, 0.760, 0.582, 0.653, 0.572, 0.587, 0.448, 0.632, 0.491, 0.549, 0.690],
    'PESQ': [1.583, 1.334, 1.684, 1.559, 1.472, 1.322, 1.443, 1.099, 1.402, 1.211, 1.189, 1.226],
    'BINAQUAL': [0.108, 0.083, 0.119, 0.079, 0.096, 0.089, 0.086, 0.052, 0.095, 0.083, 0.089, 0.083],
})

# Helper function to make cuboids centered at (x,y)
def make_bar(x, y, z, width, depth, name, color):
    # center the bars around their x,y coords
    x0, x1 = x - width/2, x + width/2
    y0, y1 = y - depth/2, y + depth/2
    return go.Mesh3d(
        x=[x0, x1, x1, x0, x0, x1, x1, x0],
        y=[y0, y0, y1, y1, y0, y0, y1, y1],
        z=[0, 0, 0, 0, z, z, z, z],
        opacity=0.7,
        color=color,
        flatshading=True,
        name=name,
        showscale=False
    )

fig = go.Figure()

colors = px.colors.qualitative.Plotly

width = 0.03
depth = 0.003

# Create one bar per model
for i, row in df.iterrows():
    bar = make_bar(
        x=row['PESQ'],
        y=row['BINAQUAL'],
        z=row['ESTOI'],
        width=width,
        depth=depth,
        name=row['Model'],
        color=colors[i % len(colors)]
    )
    fig.add_trace(bar)

# Set labels clearly
fig.update_layout(
    scene=dict(
        xaxis_title='PESQ',
        yaxis_title='BINAQUAL',
        zaxis_title='ESTOI',
        xaxis=dict(backgroundcolor="white"),
        yaxis=dict(backgroundcolor="white"),
        zaxis=dict(backgroundcolor="white"),
        camera_eye=dict(x=1.5, y=1.5, z=1)
    ),
    title="3D Histogram-style Plot of ESTOI, PESQ, and BINAQUAL per Model",
    margin=dict(l=0, r=0, b=0, t=50),
    legend_title_text='Models',
)

fig.show()
