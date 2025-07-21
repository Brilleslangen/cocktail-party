# Facing-to-Focus at The Cocktail Party
Individuals with hearing impairments often struggle to follow a single conversation in noisy, multi-speaker environments. Conventional hearing aids typically capture sound through microphones, convert it into a digital representation, and amplify the signal before streaming it into the wearer’s ears. During this process, many subtle auditory cues that allow the auditory system to focus on a single source - an ability commonly referred to as the “cocktail party effect” - are diminished or lost. As a result, focusing on one speaker among many remains challenging for hearing aid users. This thesis explores how computational models can replicate the cocktail-party effect by leveraging multi-modal signals, such as audio and spatial data, for selective speech isolation, implicitly decoding user focus and amplifying the target speaker's voice while effectively suppressing competing speakers and background noise.

## Societal Motivation
Today, approximately 1.5 billion people are affected by hearing impairments, and the global prevalence is rapidly increasing. The World Health Organization projects that 2.5 billion people will experience hearing loss by 2050. Impaired hearing significantly impacts many aspects of life, leading to social isolation, communication difficulties, fatigue, and cognitive decline. Thus, it is critical that assistive technologies restore as many capabilities of natural hearing as possible. The inability to focus on and interpret a single voice in noisy environments remains a top complaint among hearing aid users. Solving this cocktail-party problem could greatly enhance social participation, safety, and overall quality of life for millions. This project aims to move beyond basic amplification toward intelligent, context-aware hearing aids that effectively address challenging listening situations. Competent technologies are available, and now is the time to make them accessible to those who need them most.

## Research Qiuestions
This research addresses the following objectives:

- **Architecture Performance Comparison:** How do Mamba-2 and Liquid Neural
  Networks compare to Temporal Convolutional Networks and Transformers when used
  for selective isolation in noisy, multi-speaker scenarios?
- **Real-time Suitability of Stateful Architectures:** Under strict latency and
  computational constraints, can stateful models achieve superior streaming
  performance compared to stateless approaches?
- **Effectiveness of Binaural Cues for Selectivity:** Can implicit and explicit
  spatial cues provide reliable criteria for accurate target speaker selection
  and isolation?
- **Intelligibility vs. Computational Constraints:** Can neural models deliver
  significant intelligibility improvements while remaining feasible on embedded
  hearing aid hardware?


## Current Methodology

We utilize a modular Binaural TasNet architecture specifically designed for multi-modal integration and efficient real-time audio processing. At the core of this system, the separator module produces destructive interference masks. These masks, when applied to the original mixed audio signals, effectively cancel unwanted sounds, leaving the desired target speech clearly audible.

Specifically, our approach involves:
+ Creating realistic, challenging datasets with binaural audio and spatial features through advanced simulations incorporating Head-Related Transfer Functions (HRTFs), which mimic how ears receive sound from different directions.
+ Comparing the efficacy of different neural architectures within the TasNet framework for generating high-quality interference masks.
+ Assessing the models’ robustness, efficiency, and overall quality of isolated audio outputs.

Ultimately, this thesis aims to contribute to the advancement of hearing aid technologies, bridging the gap between computational speech isolation models and practical, real-world applications. By restoring clearer, more natural hearing experiences in challenging social environments, this work seeks to significantly improve the quality of life and social participation for individuals with hearing impairments.
Today, approximately 1.5 billion people are affected by hearing impairments, and the global prevalence is rapidly increasing. The World Health Organization projects that 2.5 billion people will experience hearing loss by 2050. Impaired hearing significantly impacts many aspects of life, leading to social isolation, communication difficulties, fatigue, and cognitive decline. Thus, it is critical that assistive technologies restore as many capabilities of natural hearing as possible. The inability to focus on and interpret a single voice in noisy environments remains a top complaint among hearing aid users. Solving this cocktail-party problem could greatly enhance social participation, safety, and overall quality of life for millions. This project aims to move beyond basic amplification toward intelligent, context-aware hearing aids that effectively address challenging listening situations. Competent technologies are available, and now is the time to make them accessible to those who need them most..


## Running Experiments
To queue runs for an experiment, pass them as a comma-separated list to `train.py`:

```bash
source ./venv/bin/activate 
python -m src.executables.train --config-name=runs/1-offline/tcn
```

We separate the experiments into respective folders.

# Focus-Directed Speech Isolation in Multi-Speaker Environments for Hearing Aid Applications

## Introduction
Imagine a bustling restaurant filled with overlapping conversations, clinking glasses, and cutlery scraping plates. Most listeners can effortlessly tune out the chaos and lock onto a single voice when it matters. For hearing aid users, however, this "cocktail party" scenario often becomes an impenetrable wall of noise. The aim of this project is to bridge that gap by empowering hearing aids with intelligent speech isolation. We explore deep-learning techniques that allow a user to face their chosen speaker and have that voice automatically enhanced while distracting chatter is suppressed.

Most people can tune out distractions and zoom in on one voice, a phenomenon known as the **cocktail-party effect**. Conventional hearing aids, however, blur together all nearby voices and background noise, forcing listeners to strain to catch a single conversation. This work explores how modern deep learning can replicate the cocktail-party effect by leveraging binaural audio and spatial cues to automatically enhance whichever speaker the listener is facing.

## The Challenge and Our Approach
Modern hearing aids are excellent at amplifying sound, yet they struggle in busy
multi-speaker scenarios. Our goal is to let a listener simply face the desired
talker and have that conversation isolated while competing voices fade into the
background. We explore advanced neural architectures that leverage spatial cues
to achieve this focus-driven speech isolation.

## Motivation

### Hearing Loss and the Limitations of Hearing Aids
Hearing loss affects billions worldwide and can lead to social disconnection and
reduced quality of life. Although hearing aids amplify sound, they often compress
and blur speech in noisy settings, leaving users overwhelmed and fatigued.

### Selective Speech Isolation and Deep Learning
Recent hearing aids include neural networks for noise reduction, but reliably
isolating one speaker remains an open problem. Research into TasNet,
Transformers, Structured State Space Models such as Mamba-2, and Liquid Neural
Networks offers promising avenues for selective speech enhancement.

### Architectures for Multi-Speaker Speech Separation
TasNet pioneered effective time-domain processing, while Transformers improve
separation with self-attention. Mamba-2 and Liquid Neural Networks provide
competitive performance with reduced computational cost, making them attractive
for resource-limited hardware.

### Adapting Offline Architectures to Streaming Environments
Real-world hearing aids require end-to-end latency below 10 ms. Adapting offline
models for streaming demands efficient windowing and strategies for maintaining
context. We compare stateless and stateful designs in this strict latency regime.

### Target Selectivity and Speaker Extraction
Reliable speaker selection is crucial. Audio-visual and spatial cues can assist
in identifying the target talker, yet practical systems must work without
predefined enrollment samples. Our work focuses on autonomous methods that rely
on binaural cues alone.

## Problem Definition
We seek to design hearing aid technology that robustly and selectively isolates
speech while operating within the constraints of real-world hardware. Key
research questions include:


## Proposed System and Evaluation Pipeline
Our modular binaural separation framework features stacked residual blocks with
interchangeable separator modules. We evaluate Mamba-2, Transformer, and Liquid
Neural Network cores alongside a Binaural Conv-TasNet baseline. Training and
evaluation use a true-streaming pipeline built from BRIR and HRIR simulations,
with challenging low-SNR mixtures. Metrics include multi-channel SI-SDRi,
ESTOI, PESQ, BINAQUAL, model size, and multiply-accumulate operations per
second.

Ultimately, this work aims to close the gap between computational speech
isolation research and practical, real-world hearing aid applications.

## Running Experiments
To queue runs for an experiment, pass them as a comma-separated list to `train.py`:

```bash
source ./venv/bin/activate
python -m src.executables.train --config-name=runs/1-offline/tcn,runs/1-offline/liquid,runs/1-offline/mamba

