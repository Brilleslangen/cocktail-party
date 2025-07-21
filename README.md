# Facing-to-Focus at The Cocktail Party
Individuals with hearing impairments often struggle to follow a single conversation in noisy, multi-speaker environments. Conventional hearing aids typically capture sound through microphones, convert it into a digital representation, and amplify the signal before streaming it into the wearer’s ears. During this process, many subtle auditory cues that allow the auditory system to focus on a single source - an ability commonly referred to as the “cocktail party effect” - are diminished or lost. As a result, focusing on one speaker among many remains challenging for hearing aid users. This thesis explores how computational models can replicate the cocktail-party effect by leveraging multi-modal signals, such as audio and spatial data, for selective speech isolation, implicitly decoding user focus and amplifying the target speaker's voice while effectively suppressing competing speakers and background noise.

## Societal Motivation
Today, approximately 1.5 billion people are affected by hearing impairments, and the global prevalence is rapidly increasing. The World Health Organization projects that 2.5 billion people will experience hearing loss by 2050. Impaired hearing significantly impacts many aspects of life, leading to social isolation, communication difficulties, fatigue, and cognitive decline. Thus, it is critical that assistive technologies restore as many capabilities of natural hearing as possible. The inability to focus on and interpret a single voice in noisy environments remains a top complaint among hearing aid users. Solving this cocktail-party problem could greatly enhance social participation, safety, and overall quality of life for millions. This project aims to move beyond basic amplification toward intelligent, context-aware hearing aids that effectively address challenging listening situations. Competent technologies are available, and now is the time to make them accessible to those who need them most..


## Running Experiments
To queue runs for an experiment, pass them as a comma-separated list to `train.py`:

```bash
source ./venv/bin/activate 
python -m src.executables.train --config-name=runs/1-offline/tcn
```

We separate the experiments into respective folders.
