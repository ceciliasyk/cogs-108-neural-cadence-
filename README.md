# Neural Cadence: Modeling Affective Responses to Music with fMRI Data

## Overview

Neural Cadence is a collaborative neuroimaging project investigating how
the brain responds to different types of music, with a focus on auditory
processing, motivation, and affective engagement.

Using a joint EEG-fMRI dataset of participants listening to classical,
synthetic, and baseline stimuli, our team explored whether different music
conditions produce differences in neural activity within auditory and
reward-related brain regions.

This project was completed as part of COGS 108: Neural Analytics at the
University of California, Irvine.

## Research Question

How do neural responses differ when listening to classical versus
synthetic music, particularly in auditory and motivation-related regions
such as the nucleus accumbens?

## Dataset

We used the Daly et al. (2019) joint EEG-fMRI dataset containing recordings
collected while participants listened to classical music, synthetic music,
and washout stimuli.

Our analysis focused primarily on the fMRI data from Subjects 1 and 2.

## Analysis

Our workflow included:

- fMRI preprocessing using fMRIPrep
- Region-of-interest (ROI) analysis
- Neurosynth-based masks targeting auditory and motivation-related regions
- Multi-run General Linear Model (GLM) analysis using Nilearn
- Confound regression and run-specific intercepts
- Classical vs. synthetic and music vs. baseline contrasts
- Whole-brain z-map visualization
- False discovery rate (FDR) correction

## Multi-Run GLM

Because each fMRI run contained a single condition rather than multiple
events, traditional event-based modeling was not appropriate for the
dataset.

To address this, we developed a multi-run GLM approach that:

1. Concatenated five fMRI runs into a continuous time series
2. Constructed a unified design matrix containing music conditions and
   nuisance regressors
3. Added run-specific intercepts to account for baseline differences
   between scans
4. Fit a first-level GLM using Nilearn's `FirstLevelModel`
5. Defined contrasts for classical vs. synthetic music and music vs.
   baseline

The GLM pipeline was validated using Nilearn's auditory fMRI dataset,
where the listening vs. baseline contrast produced expected activation
in bilateral auditory regions.

## Challenges & Limitations

The project presented several practical challenges common to real-world
neuroimaging analysis, including:

- Long fMRIPrep preprocessing times
- Data and file corruption issues
- Docker and environment configuration problems
- Missing affective arousal ratings
- Separate scanning runs for individual music stimuli
- Limited time and computational resources

Because of these constraints, the complete classical vs. synthetic GLM
analysis could not be applied to the final project dataset within the
project timeline.

These limitations required us to adapt our original analysis plan and
evaluate which conclusions could be supported by the available data.


## Tools & Technologies

Python • Nilearn • fMRIPrep • Neurosynth • Jupyter • GitHub • Docker

## Repository Contents

This repository contains collaborative code and materials developed by
the Neural Cadence team throughout the project.

For a concise overview of the completed work, see:

- **Final Group Report** — detailed methodology, individual contributions,
  challenges, and results
- **Final Presentation** — visual overview of the research question,
  analysis workflow, challenges, and findings

## Team

Cecilia Kuang • Josephine Nwaugha • Sophie On • Sydney Liu • Theo Benhabiles

Department of Cognitive Sciences  
University of California, Irvine  
COGS 108: Neural Analytics
