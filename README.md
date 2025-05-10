# Focus-Directed Speech Isolation in Multi-Speaker Environments for Hearing Aid Applications
Individuals with hearing impairments often struggle to follow a single conversation in noisy, multi-speaker environments. Conventional hearing aids typically capture sound through microphones, convert it into a digital representation, and amplify the signal before streaming it into the wearer’s ears. During this process, many subtle auditory cues that allow the auditory system to focus on a single source—an ability commonly referred to as the “cocktail party effect”—are diminished or lost. As a result, focusing on one speaker among many remains challenging for hearing aid users. This thesis explores how computational models can replicate the cocktail-party effect by leveraging multi-modal signals, such as audio and spatial data, for selective speech isolation, implicitly decoding user focus and amplifying the target speaker's voice while effectively suppressing competing speakers and background noise.

## Societal Motivation
Today, approximately 1.5 billion people are affected by hearing impairments, and the global prevalence is rapidly increasing. The World Health Organization projects that 2.5 billion people will experience hearing loss by 2050. Impaired hearing significantly impacts many aspects of life, leading to social isolation, communication difficulties, fatigue, and cognitive decline. Thus, it is critical that assistive technologies restore as many capabilities of natural hearing as possible. The inability to focus on and interpret a single voice in noisy environments remains a top complaint among hearing aid users. Solving this cocktail-party problem could greatly enhance social participation, safety, and overall quality of life for millions. This project aims to move beyond basic amplification toward intelligent, context-aware hearing aids that effectively address challenging listening situations. Competent technologies are available, and now is the time to make them accessible to those who need them most.

## Research Goals
This research addresses the following objectives:

+ Evaluate whether audio-spatial modalities (such as binaural audio signals and computed spatial features) alone are sufficient compared to integrating visual-spatial data for accurate selectivity and effective speech isolation.
+ Investigate and compare the performance and real-time efficiency of three state-of-the-art neural network architectures: Transformers, Structured State Space Models, and Liquid Neural Networks.
+ Determine optimal trade-offs between computational efficiency and speech isolation quality to ensure real-time feasibility in resource-constrained hearing aid hardware.

## Current Methodology

We utilize a modular Binaural TasNet architecture specifically designed for multi-modal integration and efficient real-time audio processing. At the core of this system, the separator module produces destructive interference masks. These masks, when applied to the original mixed audio signals, effectively cancel unwanted sounds, leaving the desired target speech clearly audible.

Specifically, our approach involves:
+ Creating realistic, challenging datasets with binaural audio and spatial features through advanced simulations incorporating Head-Related Transfer Functions (HRTFs), which mimic how ears receive sound from different directions.
+ Comparing the efficacy of different neural architectures within the TasNet framework for generating high-quality interference masks.
+ Assessing the models’ robustness, efficiency, and overall quality of isolated audio outputs.

Ultimately, this thesis aims to contribute to the advancement of hearing aid technologies, bridging the gap between computational speech isolation models and practical, real-world applications. By restoring clearer, more natural hearing experiences in challenging social environments, this work seeks to significantly improve the quality of life and social participation for individuals with hearing impairments.

