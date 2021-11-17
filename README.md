Networks for sequence prediction
================================

[![DOI](https://zenodo.org/badge/426944818.svg)](https://zenodo.org/badge/latestdoi/426944818)

This repository contains the code to reproduce the results in the paper: <br />
[Gated recurrence enables simple and accurate sequence prediction in stochastic, changing, and structured environments](https://doi.org/10.1101/2021.05.03.442240) <br />
by Cedric Foucault and Florent Meyniel.

If you use this code or our results in your research,
please cite our paper using for instance this citation:

```bibtex
@article{foucaultmeyniel2021gated,
  title={Gated recurrence enables simple and accurate sequence prediction in stochastic, changing, and structured environments},
  journal={bioRxiv},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}
```


## Requirements

A few Python packages are required to run the code.
To install them, run the following pip command in your terminal:
```bash
$ pip install -r requirements.txt
```

If you don't want your existing packages to be replaced,
you can create a new
[conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)
and activate it before running the above pip command.

This repository was tested with
Python 3.7.11 and Python 3.9.7 on macOS 11.6 and Ubuntu 20.04,
and will probably work with any version of Python 3
and any recent (2018+) version of macOS and Linux.


## Instructions to reproduce the results

We use [GNU Make](https://www.gnu.org/software/make/)
to make our results easy to reproduce with a single command.
If you are using Linux or macOS, you most likely already have Make installed.
To make sure, run `make --version`.
If you don't have Make, to install it, on Linux, run `sudo apt-get install build-essential`,
on macOS, run `xcode-select --install` (or install Xcode through the App Store).
On Windows, you can install Make through Chocolatey or WSL, as explained
[here](https://pakstech.com/blog/make-windows/).

You do not need to know how Make works in order to run the code,
just follow the instructions in the below sections, which tell you how to:

* Reproduce quickly all the [analyses of trained agents](#analyses-of-trained-agents)
reported in the paper (results in Figure 2 to 7, in their Figure supplements,
and in the text of the corresponding sections) using the pre-trained agent models
that we provide for convenience.

* Reproduce the [training of the agent models](#training-of-the-agent-models),
replacing the pre-trained agent models with newly trained ones which can then
be tested using the same commands as with the pre-trained agents.

* Reproduce the [complexity analyses](#complexity-analyses)
(reported in Figure 8 and in the text of the corresponding section),
which do not use any pre-trained agents.


### Analyses of trained agents

You can reproduce all the analyses of trained agents either by

1. Running each analysis individually with the make commands detailed in the below subsections:
    * [Performance](#performance)
    * [Prediction sequences](#prediction-sequences)
    * [Effective learning rate](#effective-learning-rate)
    * [Readouts](#readouts)
    * [Dynamics](#dynamics)
    * [Causal test of precision-weighting](#causal-test-of-precision-weighting)
    * [Higher-level inference](#higher-level-inference)

or:

2. Using a convenience make command that will run all of the above sequentially:

```bash
$ make clean_analyses_trained_agents   # removes all existing files in the corresponding results subdirectories and in the trained_models/decoders directory
$ make analyses_trained_agents         # runs all the Python scripts needed to generate these results
```

Each analysis takes about a few minutes to complete, except the last one (higher-level inference) which takes about an hour, for a total of about two hours for all analyses.

#### Performance

To reproduce the analyses of performance
(reported in Figure 2b, Figure 6c, Figure 6—Figure supplement 1,
and in the text of the corresponding sections), run:

```bash
$ make clean_performance
$ make performance
```
The results are saved in `results/performance`.

#### Prediction sequences

To reproduce the example prediction sequences (Figure 3a and 6b), run:

```bash
$ make clean_prediction_sequence
$ make prediction_sequence
```

The results are saved in `results/prediction_sequence`.

#### Effective learning rate

To reproduce the analyses of the effective learning rate (Figure 3b and Figure 3—Figure supplement 1), run:

```bash
$ make clean_learning_rate
$ make learning_rate
```

The results are saved in `results/learning_rate`.

#### Readouts

To reproduce the readout of precision and of bigram probabilities
and reproduce the correlational analyses on the read estimates
(Figure 4a, 6d, and the associated text), run:

```bash
$ make clean_readout
$ make readout
```

The results are saved in `results/readout` and
the decoder models that were created in the process are saved in `trained_models/decoders`.

#### Dynamics

To reproduce the analyses of the internal dynamics (Figure 4b), run:

```bash
$ make clean_dynamics
$ make dynamics
```

The results are saved in `results/dynamics`.

#### Causal test of precision-weighting

To reproduce the perturbation experiment testing the causal role of a network's precision
on its subsequent effective learning rate (Figure 5 and associated text), run:

```bash
$ make clean_perturbation_experiment
$ make perturbation_experiment
```

The results are saved in `results/perturbation_experiment`.


#### Higher-level inference

To reproduce the tests of the higher-level inference (Figure 7 and associated text), run:

```bash
$ make clean_higher_level_inference
$ make higher_level_inference
```

The results are saved in `results/higher_level_inference`.


### Training of the agent models

To reproduce the training of the agent models, run the below command.
This training may take a while, from a few tens of minutes to a few hours depending on your computing resources.


```bash
$ make clean_training_agents
$ make training_agents
```

The agent models are saved in `trained_models/agents`,
and the training curves (reported in Figure 8—figure supplement 1) are saved in `results/training`.


### Complexity analyses

To reproduce the complexity analyses (Figure 8 and associated text), run the below command.
It may take a long time, from tens of minutes to hours to days,
because it trains and tests new network models for each environment, network architecture, and number of units,
which is especially long when the number of units is large (up to 1000 units).

```bash
$ make clean_complexity
$ make complexity
```

The results are saved in `results/complexity`.

