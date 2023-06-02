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
