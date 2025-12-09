# LLM-AAI
LLM-AAI is a Python library for running Large Language Model (LLM) agents inside the [Animal-AI](https://github.com/Kinds-of-Intelligence-CFI/animal-ai?tab=readme-ov-file) (AAI) environment.

![alt text](figures/aai-llm-summary-schematic.png)

## Installation
LLM-AAI can be manually downloaded by cloning this GitHub repository.
If needed, see GitHub's [guide](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) on cloning repositories.

### Package management
This project is managed using [Pixi](https://pixi.sh/latest/), a tool for environment and dependency management. It combines conda-style environment management with `cargo`/`poetry`-style dependency management.

To install pixi, follow the instructions [here](https://pixi.sh/latest/#autocompletion).

To install the environment and dependencies for this project, run:
```shell
cd llm-aai
pixi install
```

To open a cross-platform shell for the project's environment, in the command-line of your choice, simply run:

```shell
pixi shell
```

This is equivalent to activating a conda environment. You can then run the scripts in this folder, or run `python` to interact with the interpreter.

To exit the pixi shell, run `exit`.

If you wish to add new packages, run:

```shell
pixi add new-package
```

## Usage
### Getting started
At the top-level of the repository, duplicate the [user_settings_template.py](user_settings_template.py) file, keep the copy at the top-level, and rename the copy to ```user_settings.py```.
In the newly created ```user_settings.py``` file, fill in the blank variables.

**Warning**: private data should not be placed in the [user_settings_template.py](user_settings_template.py) file.
This template file is tracked by git and updating it will be reflected remotely on GitHub.
The local copy ```user_settings.py``` can and should contain private data (for example, API keys) and is ignored by git via the [gitignore](.gitignore).

### Description of a run in LLM-AAI
The AAI framework contains batteries of cognitively-inspired digitally-embodied task instances or arenas, encoded as YAML files. See the AAI GitHub [repository](https://github.com/Kinds-of-Intelligence-CFI/animal-ai?tab=readme-ov-file), whose README points to the latest AAI paper.

The smallest building block of the LLM-AAI experiment flow is the arena run. During an arena run, an LLM controls the movements of the AAI agent, in an attempt to maximise its episode reward - in the same way as a natural or reinforcement learning agent does.

For more details about LLM-AAI, please refer to the our paper mentioned in the [Authors and Acknowledgements](#authors-and-acknowledgements) section below.

### How to run a suite of experiments
To launch multiple runs in one-click, the most important run constants are made parameterizable and are provided in the [options.yaml](options.yaml) file.

Examples of these parameters are:
- the AAI arena YAML file(s) to test the LLM in
- the LLM family (for example, Claude, Gemini, or GPT) to test
- The AAI seed(s) to test for

All the parameters (except for the ```llm_family``` and ```llm_model```) can be provided either as single values or as lists.
When lists are provided, one experiment will be launched for every combination of provided parameters, resulting in multiple experiments per suite.

Experiments can themselves contain multiple arena runs. This occurs if a folder of AAI arena YAML files is specified as the ```aai_config_path``` option rather than a single arena file.

The output folder for a suite is created and populated at runtime as a timestamped folder within the [outputs](outputs) directory.

At the end of a run, the timestamped suite output folder (for example, ```2024-10-01_14-18-09```) contains experiment folders (for example, ```aai_seeds_0``` and ```aai_seeds_1```) which themselves each contain arena run folders (for example, ```arena_1```, ```arena_2```, and ```arena_3```).

To launch a suite of experiments, after editing the [options.yaml](options.yaml) as desired, run the following command at the top-level of this project directory:
```shell
python -m scripts.main
```

### How to view a replay of a run
LLM-AAI can replay runs from a previous experiment. The `view_replay_in_aai` script demonstrates how to use the 'recording' llm to do this.

The script takes the path of a `.pkl` file generated from a previous run for a particular arena and replays it, for example a replay might be started with:

```shell
python -m scripts.view_replay_in_aai "outputs\2024-10-24_17-27-18\aai_seeds_6\sanity_green\llm_session_history_20241024172943.pkl"
```

## Support
For any questions, please contact Matteo G. Mecattaf and/or Ben Slater at the following addresses, respectively:
- mgmecattaf AT gmail DOT com
- bas58 AT cam DOT ac DOT uk

## Contributing
While we are not expecting external contributions at this stage, we are very happy to discuss avenues for collaboration.
If you have an idea for this project and would like to get involved, please reach out to the addresses under [Support](#support).

## Authors and Acknowledgements
You can find the [paper](https://arxiv.org/abs/2410.23242v1) associated with this project on arXiv. The results from the paper were generated from the version of this codebase at commit `e339118`.

To cite our work, please use the following:
```
@misc{mecattaf2024littleconversationlittleaction,
      title={A little less conversation, a little more action, please: Investigating the physical common-sense of LLMs in a 3D embodied environment}, 
      author={Matteo G. Mecattaf and Ben Slater and Marko Tešić and Jonathan Prunty and Konstantinos Voudouris and Lucy G. Cheke},
      year={2024},
      eprint={2410.23242},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2410.23242}, 
}
```

## License
[GNU AGPLv3](https://choosealicense.com/licenses/agpl-3.0/)