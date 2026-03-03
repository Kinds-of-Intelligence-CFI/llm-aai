from typing import Optional, Union, List
from src.llm_scripting.minimal_parser import minimal_parser
from src.llms.llm import LLMAPI, LLMSession, BASE64_STRING, PROMPT_CONTENTS, PromptElement, LLMMessageParam
import pickle
import numpy as np
import tiktoken
import re
import os

DEFAULT_RECORDING_LOCATION = None
RESOLUTION = 512


def load_responses(location: str) -> list[str]:
    with open(location, "rb") as file:
        return pickle.load(file)


class RecordingSession(LLMSession):
    def __init__(self,
                 api_key: str,
                 model: str,
                 switch_session: Optional[LLMSession] = None,
                 system_prompt: Optional[str] = None,
                 thinking_budget: Optional[int] = None
                 ) -> None:
        print(f"Human LLM ignoring key and model {api_key}, {model}")
        if system_prompt is not None:
            print(f"RecordingSession ignoring system_prompt")
        if thinking_budget is not None:
            print(f"RecordingSession ignoring thinking_budget")
        print(f"*** ASSUMING {RESOLUTION}*{RESOLUTION} IMAGES FOR TOKENISATION ***")
        self._history: list[LLMMessageParam] = []
        self.input_costs = np.array([])
        self.output_costs = np.array([])
        self.responses = load_responses(os.environ.get('RECORDING_LOCATION')) if os.environ.get(
            'RECORDING_LOCATION') is not None else load_responses(DEFAULT_RECORDING_LOCATION)
        self.switch_session = switch_session
        if switch_session is not None:
            print(" *** Recording session running with a switch, so costs will be recorded as zero until the switch ***")
            self.input_costs = self.switch_session.input_costs
            self.output_costs = self.switch_session.output_costs

    def prompt(
            self,
            prompt_contents: PROMPT_CONTENTS,
            resp_prefix: Optional[str] = None,
    ) -> str:
        prompt = prompt_contents
        self._history.append(
            LLMMessageParam(role="user", content=prompt_contents)
        )

        # Input is whole history including the latest prompt
        self.input_costs = np.append(self.input_costs, self._current_tokens_in_history())
        if resp_prefix is not None:
            print("**** ignoring prefix ****")
        # print(f"Prompt: {prompt}")
        if not self.responses and self.switch_session is None:
            raise ValueError("RecordingLLMSession: no responses remaining and no session to switch to")
        if not self.responses:
            response = self.switch_session.prompt(prompt_contents)
        else:
            response = self.responses.pop(0)
            if self.switch_session is not None:
                # If including a switch keep the switch session up to date
                self.switch_session.artificial_prompt(prompt_contents, [(PromptElement.Text, response)])
        if self.switch_session is None:
            # If not acting as a switch record cost estimates
            # Input is whole history including the latest prompt
            self.input_costs = np.append(self.input_costs, self._current_tokens_in_history())
            # Output is only the most recent prompt
            self.output_costs = np.append(self.output_costs, self._get_tokens_str(response))
        recorded_prompt_contents = [(PromptElement.Text, response)]
        self._history.append(
            LLMMessageParam(role="assistant", content=recorded_prompt_contents)
        )
        thoughts = re.findall(r"Think\((.*?)\)", response)
        for thought in thoughts:
            print(f"Thought: {thought}")
        return response

    def artificial_prompt(
        self,
        prompt_contents: PROMPT_CONTENTS,
        response_contents: PROMPT_CONTENTS,
    ) -> None:
        raise NotImplementedError()

    @property
    def history(self):
        if self.switch_session is not None:
            # Defer to switch session history if it exists
            return self.switch_session.history
        return [block for block in self._history]

    def _current_tokens_in_history(self) -> int:
        tokens_per_block = [
            self._get_tokens_str(prompt_block) if block_type == PromptElement.Text else self._get_tokens_img(prompt_block) for block_type, prompt_block in self._history
        ]
        return sum(tokens_per_block)

    # NOTE: If using the recordingSession to do cost estimation ensure these functions reflect the tokenisation being used
    def _get_tokens_str(self, prompt: str) -> int:
        # Approximate text tokens with GPT 4o encoding
        encoding = tiktoken.encoding_for_model("gpt-4o")
        return len(encoding.encode(prompt))

    def _get_tokens_img(self, _: str) -> int:
        # Estimates anthropic tokens ref: https://docs.anthropic.com/en/docs/build-with-claude/vision
        # "you can estimate the number of tokens used through this algorithm: tokens = (width px * height px)/750"
        return np.ceil((RESOLUTION * RESOLUTION) / 750)

    def load_from_history_file(self,
                               file: str) -> None:
        raise NotImplementedError()

    @staticmethod
    def get_assistant_commands_from_pkl_history(path_to_pkl: str) -> List[str]:
        with open(path_to_pkl, "rb") as file:
            history: List[LLMMessageParam] = pickle.load(file)
        return [element["content"][0][1] for element in history if element["role"] == "assistant"]

    @staticmethod
    def create_pkl_command_recording_from_text_file(source_txt_path: str,
                                                    result_pkl_path: str) -> None:
        """Create and return the path to a pickle file containing a list of string command-sequences.

        Make sure that the source text file has one valid command-sequence entry on every new line.
        The resulting pickle file will contain a list of command-sequences (with n commands, n>=1).
        """
        assert os.path.isfile(source_txt_path)
        assert source_txt_path.endswith(".txt")
        assert result_pkl_path.endswith(".pkl")

        source_txt_file = open(source_txt_path, "r")
        result_command_list = []
        for line_ix, line in enumerate(source_txt_file):
            # Only valid commands can be used to create a pkl command recording.
            assert minimal_parser(line)[0], f"The command sequence on line number {line_ix}: '{line}' is not valid."
            result_command_list += [line]
        with open(result_pkl_path, "wb") as result_pkl:
            pickle.dump(result_command_list, result_pkl)

class RecordingAPI(LLMAPI):
    """
    A dummy API to test accessing LLM APIs without actually doing it
    """

    def start_session(
            self,
            switch_API: Optional[LLMAPI] = None
        ):
        if switch_API is not None:
            switch_session = switch_API.start_session()
            return RecordingSession("", "", switch_session)
        return RecordingSession("", "")


if __name__ == "__main__":
    # Validating creating a pkl recording
    source_txt_path = "path/to/text/file/with/one/command/sequence/per/line.txt"
    RecordingSession.create_pkl_command_recording_from_text_file(source_txt_path=source_txt_path,
                                                                 result_pkl_path=source_txt_path.split(".")[0] + ".pkl")