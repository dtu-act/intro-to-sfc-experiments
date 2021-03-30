# An introduction to sound field control experiments

This repository contains a Jupyter notebook with code and explanations for kickstarting your sound field control experiments. It touches on transfer-function measurement, control filter design and system validation. A most recent version of this repository should be found at [github.com/dtu-act/intro-to-sfc-experiments](https://github.com/dtu-act/intro-to-sfc-experiments).

A version of this repository with already measured data for example 2 may be found at `O:\act\Labfacilities\SoundFieldControlFacility\intro-to-sfc-experiments`.

# Installation

This introduction requires Python (and some knowledge on how to use it). On the computer at ACT's sound field control room, everything is already installed in the Conda environment `base`. Normal users can not edit the environment, that is, they can not install any packages. However, you can create an environment for your own user in which you can install all the packages you like. To create a new conda environment, open the Anaconda Prompt and change to the directory of this document. Use the `environment.yml` file to create a new conda environment with all required packages for this project:

    $ conda create -f environment.yml

Activate the new environment with

    $ conda activate sfc

Now you can install new packages with `conda install ...` or `pip install ...`.

To open the notebook your browser, type

    $ jupyter lab

in the Anaconda Prompt. In there, open the file `intro to sfc experiments.ipynb`.

Good luck!