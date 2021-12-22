# ALVariability/
> Danylo Lavrentovich
> de Bivort Lab 2021

Computational model of the *Drosophila* antennal lobe (AL) for [*Neural correlates of individual odor preference in Drosophila*](http://lab.debivort.org/odor-loci-of-individuality/).

This GitHub repository contains Jupyter notebooks and Python scripts for setting up, running, and analyzing a leaky-integrate-and-fire model of 3,062 *Drosophila* olfactory receptor neurons (ORNs), local neurons (LNs), and projection neurons (PNs). Neurons and connections between them are retrieved from the *hemibrain* [Python API](https://connectome-neuprint.github.io/neuprint-python/docs/).  Odors are presented to the simulated AL by activating ORNs using [DoOR](http://neuro.uni-konstanz.de/DoOR/default.html), a database of glomerulus - odor responses. The model is used to study how circuit variation is linked to physiological variation in PN odor responses that is relevant to olfactory preference behavior. A complete repository, with added data files too large for sharing via GitHub, is also available at the first link.

### Directory structure

A brief walkthrough of the directories in this repository:

**utilities/** contains helpful auxiliary functions used in setting up, running, and plotting outputs of AL circuit models

**run_model/** contains shell and Python scripts for running circuits or batches of circuits, either manually or using a computing cluster. See next section for more details

**analysis/** contains Jupyter notebooks used for selecting a model from a parameter sweep, plotting model outputs, and performing PCA on odor responses of simulated flies to compare to PCA on calcium responses in real flies

**odor_imputation/** contains Jupyter notebooks/MATLAB scripts for imputing missing glomerulus-odor responses for odors used to calibrate and run the model

**datasets/** contains downloaded literature files for model tuning and evaluation

**connectomics/** contains a Jupyter notebook that retrieves AL neurons from the *hemibrain* Python API and sets the list of neurons / synapses used in the model

### Running a model

Quick instructions for simulating an AL: 

- `run_model/export_resampling_sim_settings.py` is the key file for setting up a model before it is run. Edit this file directly to set desired parameters, such as the names + durations of odor stimuli, the cell populations to bootstrap, which LNs to set as excitatory, etc.
- Once you are done adding desired model settings, be sure to edit the `run_tag` string in `run_model/export_resampling_sim_settings.py` to set the name of a new directory that will 1) store the input parameters for the desired model and 2) be where the model is actually run. Execute `python run_model/export_resampling_sim_settings.py` to create that new directory with the name set in `run_tag`
- The created model directory will contain a Python pickled dictionary, `sim_params_seed.p`, that contains the input parameters of the model
- The created model directory will also contain a Python script, `run_sim.py`. If you navigate to the created model directory and run `python run_sim.py` , this script integrates the leaky-integrate-and-fire equations and runs the model based on the input parameters set in `sim_params_seed.p`
  - `run_sim.py` is a Python script that is a copy of `run_model/run_sim_template.py`. This template script is not meant to be heavily editable for individual jobs. Rather, once a copy of this template is in a model directory with input parameters defined in `sim_params_seed.p`, this script reads in those input parameters, loads them into an instance of the `Sim` class (defined in `utils/simulation_class.py`), then uses `run_LIF`to run the model and saves model outputs such as the spike times, voltage traces, firing rates, etc. 
- To replace manually running Python scripts, you can run `run_model/autolaunchsim.sh` to perform all of the above steps (execute `export_resampling_sim_settings.py` to create a new model directory with input parameters and execute `run_sim.py` within that new directory to run the model and save outputs)
- If you are on a research computing cluster, instead of executing `python run_sim.py` within the model directory, you can submit it as a job by executing `./submit_job.sh`. Similar to `run_sim.py`, `submit_job.sh` is a copy of `run_model/submit_to_cluster_template.sh` that is outputted by `run_model/export_resampling_sim_settings.py` 
- For running a series of jobs, for instance when doing multiple cell-type bootstraps, a script like `run_model/launch_resamples.sh` will feed desired input arguments into `export_resampling_sim_settings.py`, run it, navigate to the created directory, and run the model on a computing cluster via `submit_job.sh`

### Implementation details

Python 3.6 was used for all Jupyter notebooks and Python scripts. You can install the Anaconda environment used for the project using `alvar.yml` in this repository:

```
conda env export --name ALVar > alvar.yml
```