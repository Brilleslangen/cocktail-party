import plotly.graph_objects as go
import torchaudio
import numpy as np


COLORS = {"blue": "#1976D2", "orange": "#DF672A", "green": "#388E3C"}


def load_waveform(audio_path: str):
    waveform, sample_rate = torchaudio.load(audio_path)
    return waveform.numpy(), sample_rate


def plot_waveforms(audio_path_1: str, audio_path_2: str, audio_path_3: str, title_1="Waveform 1", title_2="Waveform 2", title_3="Waveform 3"):
    waveform_1, sr_1 = load_waveform(audio_path_1)
    waveform_2, sr_2 = load_waveform(audio_path_2)
    waveform_3, sr_3 = load_waveform(audio_path_3)

    if sr_1 != sr_2:
        raise ValueError("Sample rates do not match!")

    time_1 = np.arange(waveform_1.shape[1]) / sr_1
    time_2 = np.arange(waveform_2.shape[1]) / sr_2
    time_3 = np.arange(waveform_2.shape[1]) / sr_3

    # Combined channels
    fig_combined = go.Figure()
    fig_combined.add_trace(go.Scatter(x=time_1, y=(waveform_1[0] + waveform_1[1]) / 2, mode='lines', name=f'{title_1}', line=dict(color=COLORS["blue"])))
    fig_combined.add_trace(go.Scatter(x=time_3, y=(waveform_3[0] + waveform_3[1]) / 2, mode='lines', name=f'{title_3}', line=dict(color=COLORS["green"], dash='dash')))
    fig_combined.add_trace(go.Scatter(x=time_2, y=(waveform_2[0] + waveform_2[1]) / 2, mode='lines', opacity=0.6, name=f'{title_2}', line=dict(color=COLORS["orange"], dash='dash')))
    
    fig_combined.update_layout(
        title={"text": "Waveform Comparison", "font": {"size": 38}},
        xaxis={"title": {"text": "Time (seconds)", "font": {"size": 30}}},
        yaxis={"title": {"text": "Amplitude", "font": {"size": 30}}},
        legend={"title": {"text": "Waveforms", "font": {"size": 26}}, "font": {"size": 26}},
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig_combined.show()


def main():
    output = "/home/erlend/Documents/cocktail-party/outputs/inference/mix_clip_000847_clip_002307_separated.wav"
    mix = "/home/erlend/Documents/cocktail-party/artifacts/static-2-spk-noise:v12/test/mixture/mix_clip_000847_clip_002307.wav"
    clean = "/home/erlend/Documents/cocktail-party/artifacts/static-2-spk-noise:v12/test/clean/clean_clip_000847_clip_002307.wav"
    plot_waveforms(mix, clean, output, "Noisy Mixture", "Target", "Model Output")


if __name__ == "__main__":
    main()

