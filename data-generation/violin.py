import pandas as pd
import plotly.express as px

# 'estoi', 'pesq', 'binaqual', 'mc_si_sdr_i'

# Load your CSV data
df = pd.read_csv('outputs/S-liquid-offline_1.08M_metrics.csv')

# Melt DataFrame to suitable format for violin plots
df_melted = df.melt(value_vars=['confusion_rate'],
                    var_name='Metric', value_name='Value')

fig = px.violin(df_melted, y='Value', x='Metric', color='Metric', box=False, points='all')

fig.update_layout(
    title="Metric Distribution Violin Plot",
    yaxis_title="Metric Values",
    xaxis_title="Metrics",
)

fig.show()

