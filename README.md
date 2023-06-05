> ⚠️‎‎‎ **For more information regarding state-of-the-art disc labeling techniques, a recent initiative has led to the creation of an open-source benchmark: the code is available here https://github.com/spinalcordtoolbox/disc-labeling-benchmark**

# Intervertebral disc labeling with the hourglass approach

## Description

This is the continuation of the work made by [Reza Azad](https://www.linkedin.com/in/reza-azad-37a652109/), [Lucas Rouhier](https://www.linkedin.com/in/lucas-rouhier-1aa36a131/?originalSubdomain=ca), and [Julien Cohen-Adad](https://scholar.google.ca/citations?user=6cAZ028AAAAJ&hl=en) on a deeplearning based architecture called `Stacked Hourglass Network` to detect and classify vertebral discs automatically on MR images.

Their work was [published](https://dl.acm.org/doi/abs/10.1007/978-3-030-87589-3_42) in a paper called "Stacked Hourglass Network with a Multi-level Attention Mechanism: Where to Look for Intervertebral Disc Labeling" for a MICCAI Workshop in 2021

This repository will be used to train and test the `Stacked Hourglass Network` in different MR case scenario.

## Cross references

In this section are referenced repositories and issues related to this work.

### Issues

* Summary [issue](https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3793) by @joshuacwnewton on the work previously done to replace the current `sct_label_vertebrae`. 

### Older repositories

* [Original](https://github.com/rezazad68/Deep-Intervertebral-Disc-Labeling) hourglass work done by @rezazad68.
* [Method comparison](https://github.com/NathanMolinier/intervertebral-disc-labeling/blob/master/README.md) between hourglass and the current non deepllearning based method `sct_label_vertebrae` implemented in the [spinalcordtoolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox).
* [Retrained model](https://github.com/ivadomed/model_label_intervertebral-disc_t1-t2_hourglass-net) by @joshuacwnewton to fix [straight vs. curved input data](https://github.com/ivadomed/ivadomed/pull/852#discussion_r710455668) issues. 

## Getting Started

To get started with this repository, follow the steps below:

1. Clone the repository to your local machine using the command:
```Bash
git clone https://github.com/spinalcordtoolbox/disc-labeling-hourglass.git
cd disc-labeling-hourglass/
```

2. Set up the required environment and dependencies.
```Bash
conda create -n myenv python=3.8
conda activate myenv
pip install -r requirements.txt
pip install -e
```
(in development) --> cf https://github.com/spinalcordtoolbox/disc-labeling-hourglass/issues/18

3. Gather only vertebral data (from [BIDS](https://bids.neuroimaging.io/) format)
```Bash
python src/dlh/data_management/gather_data.py --datapath DATAPATH -o VERTEBRAL_DATA --suffix-img SUFFIX_IMG --suffix-label SUFFIX_LABEL
```

4. Train hourglass on the vertebral data
> 🐝 Currently the training is monitored using wandb (please check [here](https://wandb.ai/site)). Please log using `wandb login` in the command or train offline with `wandb offline`
```Bash
python src/dlh/train/main.py --datapath VERTEBRAL_DATA -c t2 --ndiscs 15
```

## Contributions and Feedback

Contributions to this repository are welcome. If you have developed a new method or have improvements to existing methods, please submit a pull request. Additionally, feedback and suggestions for improvement are highly appreciated. Feel free to open an issue to report bugs, propose new features, or ask questions.

## License

For more information regarding the license, please refere to the LICENSE file.
