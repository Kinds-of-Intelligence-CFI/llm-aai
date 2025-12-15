import pickle
from os import listdir
from os.path import isfile, join
from typing import Dict, List, Literal, Union
import os
import traceback
import random

import numpy as np
import yaml
from animalai.environment import AnimalAIEnvironment

import user_settings
from src.definitions.prompts.observations import IN_SESSION_MSG_TO_LLM, YIELD_OBS_MESSAGE, \
    PREVIOUS_RESPONSE_IS_INVALID
from src.definitions.prompts.prompts import (
    CHAINS_OF_THOUGHT,
    COMMANDS,
    GOALS,
    MISC,
    OBSERVATIONS,
    PREAMBLES,
    create_background_prompt,
    NUM_INITIAL_OBS, N_SHOT,
)
from src.experimentation.experiments.experiment import Experiment
from src.llm_scripting.minimal_parser import minimal_parser, YIELD_OBS, ActionTuple
from src.llms.llm import PromptElement, PROMPT_CONTENTS
from src.llms.human import \
    LLMMessageParam
from src.llms.llm_to_api_key import llm_to_api_key
from src.llms.session_factory import LLMSessionFactory
from src.utilities.utils import get_change_in_total_reward, populate_csv, try_mkdir, check_episode_pass
from src.vision.camera import CameraSystem
from src.definitions.cardinal_directions import action_name_to_action_tuple
from mlagents_envs.base_env import (
    DecisionSteps,
    TerminalSteps,
)
from src.definitions.constants import FRAMES_BETWEEN_OBS

EpisodeEndReasons = Literal[
    "NON_ZERO_TERMINAL_REWARD",  # Time limit, success, or failure in env
    "RUNTIME_ERROR",
    "CONVERSATION_TURNS_EXCEEDED",
    "REASON_UNKNOWN",
    "MESSAGE_PARSING_ERROR"
]
ARENA_LOOP_SUFFIX = lambda loop: f"_loop_{loop}"
MESSAGE_PARSING_ERROR_MESSAGE = "Parsing response fails: "


def append_text_to_prompt(prompt: PROMPT_CONTENTS, text: str) -> PROMPT_CONTENTS:
    """Append a string to the prompt
    If the final block is a string, it will be appended directly to that string
    Otherwise, a new block will be added containing that string
    Returns a reference to the updated list
    """
    if len(prompt) == 0:
        return [
            (PromptElement.Text, text)
        ]
    final_block_type, final_block_contents = prompt.pop()
    if final_block_type.value == PromptElement.Image.value:
        return prompt + [
            (PromptElement.Image, final_block_contents),
            (PromptElement.Text, text)
        ]
    else:
        return prompt + [
            (PromptElement.Text, final_block_contents + text)
        ]


class Experiment1(Experiment):
    """In this experiment, one llm session is started per arena config."""

    def __init__(self, options: Dict):
        super().__init__(options=options)
        self._api_key = llm_to_api_key[self.options["llm_family"]] if self.options["llm_family_switch"] is None else llm_to_api_key[self.options["llm_family_switch"]]
        self._arena_config_paths = self._generate_arena_config_paths()
        self._result_folder_path = join(self.options["output_folder_path"], "results/")
        try_mkdir(self.options["output_folder_path"])
        try_mkdir(self._result_folder_path)
        self._save_options_to_output_directory(
            output_dir=self.options["output_folder_path"]
        )

        # Initialise result arrays
        self._arena_names = np.array([])
        self._episode_rewards = np.array([])
        self._episode_end_reasons = np.array([])

    def run(self) -> None:
        history_index = 0
        vision_system = CameraSystem()

        background_prompt = create_background_prompt(
            preamble=PREAMBLES[self.options["preamble"]],
            observation=vision_system.observation_prompt,  # TODO: see discussion in VisionSystem.
            goal=GOALS[self.options["goal"]],
            commands=COMMANDS[self.options["commands"]](self.options["max_conversation_turns"]),
            chain_of_thought=CHAINS_OF_THOUGHT[self.options["chain_of_thought"]],
            misc=MISC[self.options["misc"]],
        )

        session = self._get_llm_session(background_prompt)
        message: PROMPT_CONTENTS = self._create_initial_message()
        episode_end_reason: EpisodeEndReasons = "REASON_UNKNOWN"

        for loop_index in range(self.options["num_arena_loops"]):
            for config_index, config_path in enumerate(self._arena_config_paths):
                config_name = os.path.basename(config_path).split(".")[-2]

                # Only add a suffix to the arena folder name if there is more than one loop to perform over the arenas.
                if self.options["num_arena_loops"] > 1:
                    config_output_path = join(
                        self.options["output_folder_path"],
                        config_name + ARENA_LOOP_SUFFIX(loop_index),
                    )
                else:
                    config_output_path = join(
                        self.options["output_folder_path"], config_name
                    )
                try_mkdir(config_output_path)

                if self.options["verbose"]:
                    print(f"Starting to solve: {config_path}")

                if not self.options["learn_across_arenas"]:
                    message = self._create_initial_message()
                    session = self._get_llm_session(background_prompt)
                    history_index = 0
                env = AnimalAIEnvironment(
                    file_name=user_settings.ENV_PATH,
                    arenas_configurations=config_path,
                    seed=self.options["aai_seeds"],
                    play=self.options["play"],
                    inference=self.options["watch_agent_interact"],
                    log_folder=user_settings.LOG_FOLDER,
                    base_port=5005 + (self.options["aai_seeds"] % 100) + config_index,
                    resolution=self.options["resolution"]
                )
                try:
                    behavior = list(env.behavior_specs.keys())[0]
                    env.step()  # Need to make a first step in order to get an observation.
                    message = append_text_to_prompt(message, MISC["send_off_with_start_of_episode_message"])
                    dec, term = env.get_steps(behavior)
                    total_reward = get_change_in_total_reward(dec,term)
                    if len(term.reward) > 0:
                        raise RuntimeError("Episode unexpectedly ended before taking any actions")
                    message, done, change_total_reward = self._update_message_with_obs(
                        message,
                        env,
                        vision_system,
                        f"{config_output_path}/obs-{0}.{0}.jpg",
                        behavior
                    )
                    total_reward += change_total_reward
                    if done:
                        raise RuntimeError("Episode unexpectedly ended during initial obs")
                    # Add any additional initial obs
                    for i in range(1, NUM_INITIAL_OBS):
                        message, done, change_total_reward = self._update_message_with_obs(
                            message,
                            env,
                            vision_system,
                            f"{config_output_path}/obs-{0}.{i}.jpg",
                            behavior
                        )
                        total_reward += change_total_reward
                    if done:
                        raise RuntimeError("Episode ended unexpectedly immediately after initial obs")
                    done = False
                    # The number of times the LLM has been prompted
                    turn = 0

                    while not done and turn < self.options["max_conversation_turns"]:
                        message = append_text_to_prompt(message,
                                                        IN_SESSION_MSG_TO_LLM(env.get_obs_dict(dec.obs)["health"],
                                                                              self.options[
                                                                                  "max_conversation_turns"] - turn))
                        if self.options["manually_prompt_llm"]:
                            input("Keep prompting LLM API?")
                        response = session.prompt(
                            message
                        )
                        if self.options["verbose"]:
                            print(f"LLM response: {response}")
                        # Reset the message since we've used its contents
                        message = []
                        turn += 1

                        ok, actions = minimal_parser(response)
                        if not ok:
                            print(MESSAGE_PARSING_ERROR_MESSAGE+response)
                            message = append_text_to_prompt(message, PREVIOUS_RESPONSE_IS_INVALID)
                            actions = [action_name_to_action_tuple["NOOP"]]
                        # Widen the type definition of actions to include YIELD_OBS()
                        actions: List[Union[ActionTuple, YIELD_OBS]]
                        # Always end in an observation
                        actions.append(YIELD_OBS())
                        i = -1
                        while not done and len(actions) > 0:
                            i += 1
                            action = actions.pop(0)
                            if action == YIELD_OBS():
                                message, done, change_total_reward = self._update_message_with_obs(
                                    message,
                                    env,
                                    vision_system,
                                    f"{config_output_path}/obs-{turn}.{i}.jpg",
                                    behavior,
                                    # Don't wait on the final timestep
                                    len(actions) > 0
                                )
                                total_reward += change_total_reward
                                continue
                            env.set_actions(behavior, action)
                            env.step()
                            dec, term = env.get_steps(behavior)
                            done = len(term.reward) > 0
                            total_reward += get_change_in_total_reward(dec, term)
                    if done:
                        # TODO: Remove hardcoded 0 and handle multi arena configs
                        ep_pass = check_episode_pass(total_reward, config_path, 0)
                        episode_end_reason = "NON_ZERO_TERMINAL_REWARD"
                        message = append_text_to_prompt(message, MISC["end_of_episode_message"](ep_pass))
                        if not ep_pass:
                            message = append_text_to_prompt(message, "Failure reason: Ran out of health.\n")

                    elif turn >= self.options["max_conversation_turns"]:
                        # Agents accrue a small -ve reward each timestep
                        # So run down the clock on the episode so that agents don't benefit by running out of scripts
                        # TODO: Handle the case where the the episode has no time limit
                        if self.options["verbose"]:
                            print("Reached max_conversation_turns: completing the level with NOOPs (note this will hang if the episode has no time limit)")
                        while not done:
                            env.set_actions(behavior, action=action_name_to_action_tuple["NOOP"])
                            env.step()
                            dec, term = env.get_steps(behavior)
                            done = len(term.reward) > 0
                            total_reward += get_change_in_total_reward(dec, term)
                        ep_pass = check_episode_pass(total_reward, config_path, 0)
                        episode_end_reason = "CONVERSATION_TURNS_EXCEEDED"
                        message = append_text_to_prompt(message, MISC["end_of_episode_message"](ep_pass))
                        if not ep_pass:
                            message = append_text_to_prompt(message, "Failure reason: No more scripts can be sent this level.\n")

                except Exception as e:
                    # TODO: Discuss whether this is the best way.
                    if not self.options["learn_across_arenas"]:
                        session.write_to_file(path=self.options["output_folder_path"])
                    if str(e).startswith(MESSAGE_PARSING_ERROR_MESSAGE):
                        episode_end_reason = "MESSAGE_PARSING_ERROR"
                    else:
                        episode_end_reason = "RUNTIME_ERROR"
                    print(traceback.format_exc())
                finally:
                    env.close()
                    session.write_to_file(
                        path=f"{config_output_path}/", write_from_index=history_index
                    )
                    history_index = len(session.history)

                    session.save_cost_arrays(cost_folder_path=config_output_path)
                    if self.options["verbose"]:
                        print(f"Reward garnered for {config_path}: {total_reward}")

                    self._arena_names = np.append(self._arena_names, config_name)
                    self._episode_rewards = np.append(
                        self._episode_rewards, total_reward
                    )
                    self._episode_end_reasons = np.append(
                        self._episode_end_reasons, episode_end_reason
                    )

                    np.save(
                        join(self._result_folder_path, "arena_names.npy"),
                        self._arena_names,
                    )
                    np.save(
                        join(self._result_folder_path, "episode_rewards.npy"),
                        self._episode_rewards,
                    )

                    if self.options["verbose"]:
                        print(f"Episode end reason: {episode_end_reason}")
                    np.save(
                        join(self._result_folder_path, "episode_end_reason.npy"),
                        self._episode_end_reasons,
                    )

            # TODO: Discuss whether this is the best way.
            if not self.options["learn_across_arenas"]:
                session.write_to_file(path=self.options["output_folder_path"])

    def _get_llm_session(self, background_prompt: str):
        if self.options["llm_family_switch"] is not None:
            return LLMSessionFactory.create_llm_session(
                name=self.options["llm_family"],
                api_key=self._api_key,
                model=self.options["llm_model"],
                system_prompt=background_prompt,
                switch_session=LLMSessionFactory.create_llm_session(
                    name=self.options["llm_family_switch"],
                    api_key=self._api_key,
                    model=self.options["llm_model_switch"],
                ),
            )
        return LLMSessionFactory.create_llm_session(
            name=self.options["llm_family"],
            api_key=self._api_key,
            model=self.options["llm_model"],
            system_prompt=background_prompt,
        )

    def _generate_arena_config_paths(self) -> List[str]:
        if (
                ".yaml" in self.options["aai_config_path"]
                or ".yml" in self.options["aai_config_path"]
        ):
            # A single arena was specified for this experiment
            config_paths = [self.options["aai_config_path"]]
        else:
            # A folder of arenas was specified for this experiment
            config_paths = []
            for element in listdir(self.options["aai_config_path"]):
                if isfile(join(self.options["aai_config_path"], element)) and (
                        ".yaml" in element or ".yml" in element
                ):
                    config_paths += [join(self.options["aai_config_path"], element)]
        return sorted(config_paths)

    def _save_options_to_output_directory(self, output_dir: str) -> None:
        # Copied function from ExperimentSuite to keep them decoupled for the time being
        with open(f"{output_dir}/options.yaml", "w") as outfile:
            yaml.dump(self.options, outfile, default_flow_style=False)

    def _update_message_with_obs(
            self,
            message: PROMPT_CONTENTS,
            env: AnimalAIEnvironment,
            vision_system: CameraSystem,
            save_path: str,
            behavior: str,
            wait: bool = True
    ) -> tuple[PROMPT_CONTENTS, bool, float]:
        message = append_text_to_prompt(message, YIELD_OBS_MESSAGE)
        _, visual_obs_b64 = vision_system.get_observation(
            env=env,
            save=self.options["save_observations"],
            save_path=save_path,
            show=self.options["show_observations"],
        )
        message += [
            (PromptElement.Image, visual_obs_b64)
        ]
        # Note we don't include the reward from the current get_steps; we assume this has already been counted
        total_reward = 0
        _, term = env.get_steps(behavior)
        if len(term.reward) > 0:
            return message, True, total_reward
        if wait:
            for _ in range(FRAMES_BETWEEN_OBS):
                env.set_actions(behavior_name=behavior, action=action_name_to_action_tuple["NOOP"])
                env.step()
                dec, term = env.get_steps(behavior)
                total_reward += get_change_in_total_reward(dec, term)
                if len(term.reward) > 0:
                    return message, True, total_reward
        return message, False, total_reward

    def _create_initial_message(self) -> PROMPT_CONTENTS:
        n_shot_path = self.options["n_shot_examples_path"]
        """Creates the initial message that is passed to the LLM, prior to any interaction with the LLM.

        Note:
        - Currently only used for n-shot examples
        """
        # TODO: add new line between initial_message and send-off
        def _add_single_example_run_to_message(message: PROMPT_CONTENTS,
                                               pickle_history_path: str) -> PROMPT_CONTENTS:
            message += [
                (PromptElement.Text, N_SHOT["example_prefix"])
            ]
            with open(pickle_history_path, "rb") as file:
                n_shot_example: list[LLMMessageParam] = pickle.load(file)

            for index, message_param in enumerate(n_shot_example):
                if index == 0:
                    # TODO: Is this how we want to avoid duplicate background prompts?
                    # Skip first element of the n_shot example as it should be the background prompt.
                    message_param["content"].pop(0)
                message += [
                    (PromptElement.Text, N_SHOT["character_prefix"](message_param["role"]))
                ]
                prompt_contents = message_param["content"]
                for (content_type, content) in prompt_contents:
                    if content_type.value == PromptElement.Text.value:
                        message = append_text_to_prompt(message, content)
                    else:
                        message += [
                            (PromptElement.Image, content)
                        ]
            return message

        initial_message = []
        if n_shot_path is None:
            return initial_message
        else:
            if n_shot_path.endswith(".pkl"):
                # A single pickle file was provided.
                initial_message = append_text_to_prompt(initial_message, N_SHOT["one_example"])
                initial_message = _add_single_example_run_to_message(initial_message, n_shot_path)
            else:
                # A directory of pickle files was provided.
                num_examples = len([file_name for file_name in os.listdir(n_shot_path)])
                initial_message = append_text_to_prompt(initial_message, N_SHOT["n_examples"](num_examples))
                for pickle_file_name in os.listdir(n_shot_path):
                    pickle_file_path = os.path.join(n_shot_path, pickle_file_name)
                    initial_message = _add_single_example_run_to_message(initial_message, pickle_file_path)
            initial_message = append_text_to_prompt(initial_message, "\n")
            return initial_message
