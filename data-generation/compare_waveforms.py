import plotly.graph_objects as go
import torchaudio
import torch
import numpy as np

COLORS = {"blue": "#1976D2", "orange": "#DF672A", "green": "#388E3C", "purple": "#56237D"}

def load_waveform(audio_path: str):
    waveform, sample_rate = torchaudio.load(audio_path)
    return waveform, sample_rate

def compute_spectrogram(waveform: torch.Tensor, analysis_window=512, hop_size=32):
    spectrogram = torch.stft(
        waveform,
        n_fft=analysis_window,
        hop_length=hop_size,
        window=torch.hann_window(analysis_window),
        return_complex=True
    )
    return torch.abs(spectrogram).numpy()

def plot_spectrogram(spectrogram, sample_rate, hop_size, title, channel_label):
    time_bins = spectrogram.shape[-1]
    freq_bins = spectrogram.shape[-2]

    times = np.arange(time_bins) * hop_size / sample_rate
    freqs = np.linspace(0, sample_rate / 2, freq_bins)

    fig = go.Figure(data=go.Heatmap(
        z=20 * np.log10(spectrogram + 1e-8),  # dB scale
        x=times,
        y=freqs,
        colorscale='Viridis'
    ))

    fig.update_layout(
        title={"text": f"{title} ({channel_label} channel)", "font": {"size": 32}},
        xaxis={"title": {"text": "Time (seconds)", "font": {"size": 24}}},
        yaxis={"title": {"text": "Frequency (Hz)", "font": {"size": 24}}},
        template="plotly_white"
    )

    fig.show()

def plot_all_spectrograms(audio_path_1, audio_path_2, audio_path_3, titles):
    waveforms = []
    sample_rates = []

    for path in [audio_path_1, audio_path_2, audio_path_3]:
        waveform, sample_rate = load_waveform(path)
        waveforms.append(waveform)
        sample_rates.append(sample_rate)

    if not all(sr == sample_rates[0] for sr in sample_rates):
        raise ValueError("Sample rates do not match!")

    for wf, title in zip(waveforms, titles):
        for channel, channel_label in enumerate(['Left', 'Right']):
            spec = compute_spectrogram(wf[channel])
            plot_spectrogram(spec, sample_rates[0], hop_size=32, title=title, channel_label=channel_label)

def main():
    output = "/home/erlend/Documents/cocktail-party/outputs/inference/mix_clip_000847_clip_002307_separated.wav"
    mix = "/home/erlend/Documents/cocktail-party/artifacts/static-2-spk-noise:v12/test/mixture/mix_clip_000847_clip_002307.wav"
    clean = "/home/erlend/Documents/cocktail-party/artifacts/static-2-spk-noise:v12/test/clean/clean_clip_000847_clip_002307.wav"

    plot_all_spectrograms(
        mix, clean, output,
        ["Noisy Mixture", "Target", "Model Output"]
    )

if __name__ == "__main__":
    main()
