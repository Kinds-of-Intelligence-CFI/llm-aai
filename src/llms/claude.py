# https://docs.anthropic.com/claude/docs/quickstart-guide
from typing import List, Literal, Optional, Union

import anthropic
import numpy as np
from anthropic.types import MessageParam, TextBlockParam
from anthropic.types.image_block_param import ImageBlockParam
import base64
import httpx
import pickle

import user_settings
from src.llms.llm import BASE64_STRING, LLMAPI, LLMSession, PromptElement, PROMPT_CONTENTS

SupportedAnthropicModels = Literal[
    "claude-3-opus-20240229",
    "claude-3-5-sonnet-20240620",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307"
]


class AnthropicSession(LLMSession):

    # Stops generation beyond these
    stop_sequences = ["<EOS>"]

    def __init__(self,
                 api_key: str,
                 # https://docs.anthropic.com/claude/docs/models-overview
                 model: SupportedAnthropicModels,
                 system_prompt: Optional[str] = None,
                 ) -> None:
        super().__init__()
        self._client = anthropic.Anthropic(api_key=api_key)
        self._history: list[MessageParam] = []
        self._model = model
        self._system_prompt = system_prompt

    def prompt(self,
                prompt_contents: PROMPT_CONTENTS,
                resp_prefix: Optional[str] = None
            ):
        prompt = self._prompt_contents_to_prompt(prompt_contents)

        # Remove cache_control from all previous messages to avoid exceeding the 4 breakpoint limit
        # Anthropic allows max 4 cache_control markers per request
        # Since our context only grows, we only need cache_control on the most recent message
        for message in self._history:
            if isinstance(message["content"], list):
                for content_block in message["content"]:
                    if isinstance(content_block, dict) and "cache_control" in content_block:
                        del content_block["cache_control"]

        self._history.append(MessageParam(role='user', content=prompt))

        # Add cache control to the most recent user message
        # This creates a cache breakpoint that caches EVERYTHING up to this point
        # Total breakpoints: system prompt (1) + this message (1) = 2 (well under limit of 4)
        if len(self._history[-1]["content"]) > 0:
            self._history[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        if resp_prefix is not None:
            self._history.append(MessageParam(role='assistant', content=resp_prefix))

        # Build the API call parameters
        api_params = {
            "model": self._model,
            "max_tokens": 1024,
            "temperature": 0.0,
            "messages": self._history,
            "stop_sequences": AnthropicSession.stop_sequences,
        }

        # Add system prompt with cache control if provided
        if self._system_prompt:
            api_params["system"] = [
                {
                    "type": "text",
                    "text": self._system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ]

        message = self._client.messages.create(**api_params)
        response_content = message.content

        self.input_costs = np.append(self.input_costs, message.usage.input_tokens)
        self.output_costs = np.append(self.output_costs, message.usage.output_tokens)

        if len(response_content) > 1:
            # TODO: When do we get multiple responses?
            raise ValueError(f"Unexpected multiple returns: {response_content}")
        if resp_prefix is not None:
            self._history.pop()
            self._history.append(MessageParam(role='assistant', content=resp_prefix + ''.join(
                [block.text for block in response_content])))
        else:
            self._history.append(MessageParam(role='assistant', content=response_content))
        return response_content[0].text

    def artificial_prompt(
        self,
        prompt_contents: PROMPT_CONTENTS,
        response_contents: PROMPT_CONTENTS,
    ) -> None:
        """ Add an artificial prompt-response to the conversation history"""
        self._history.append(
            MessageParam(
                role='user',
                content=self._prompt_contents_to_prompt(prompt_contents)
            )
        )
        self._history.append(
            MessageParam(
                role='assistant',
                content=self._prompt_contents_to_prompt(response_contents)
            )
        )

    def load_from_history_file(self,
                               file: str) -> None:
        print("Resetting Claude history for file load")
        assert file[-4:] == ".pkl", "File must be a pickle (.pkl)"
        with open(file, "rb") as f:
            try:
                self._history = pickle.load(f)
            finally:
                f.close()

    @property
    def history(self) -> list[MessageParam]:
        return [message for message in self._history]

    @staticmethod
    def _prompt_contents_to_prompt(
        prompt_contents: PROMPT_CONTENTS
    ) -> List[Union[TextBlockParam, ImageBlockParam]]:
        # Confirm we have at least one text element
        assert len([True for type, _ in prompt_contents if type.value == PromptElement.Text.value]) > 0, "Must have at least 1 text element"
        return [
            ImageBlockParam(
                type="image",
                source={
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": contents
                }
            ) if type.value == PromptElement.Image.value else TextBlockParam(
                type="text",
                text=contents
            ) for type, contents in prompt_contents
        ]

    @staticmethod
    def get_assistant_commands_from_pkl_history(path_to_pkl: str) -> List[str]:
        with open(path_to_pkl, "rb") as file:
            history: List[MessageParam] = pickle.load(file)
        return [
                response["content"][0].text
                for response in history
                if response["role"] == "assistant"
            ]

class AnthropicAPI(LLMAPI):
    def __init__(self,
                 api_key: str,
                 # https://docs.anthropic.com/claude/docs/models-overview
                 model: SupportedAnthropicModels,
                 system_prompt: Optional[str] = None
                 ) -> None:
        self._api_key = api_key
        self._model = model
        self._system_prompt = system_prompt

    def start_session(self):
        return AnthropicSession(self._api_key, self._model, self._system_prompt)


if __name__ == "__main__":
    #  Check it works
    session = AnthropicAPI(user_settings.CLAUDE_API_KEY,
                           "claude-3-haiku-20240307"
                           ).start_session()
    # -- Basic check --
    print(session.prompt([(PromptElement.Text, "Hello")], None))
    print(session.prompt([(PromptElement.Text, "That's interesting")], None))
    print(session.history)
    session.write_to_file("claude_test_history")

    # -- check loading history
    session.load_from_history_file(r"a-saved-pickle-location")
    print(session.history)
    print(session.prompt([(PromptElement.Text, "Testing!")], None))
    print(session.history)

    # -- Image check --
    cat_picture = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
    dog_picture = "https://upload.wikimedia.org/wikipedia/commons/b/b5/Iridescent.green.sweat.bee1.jpg"
    media_type = "image/jpeg"
    cat_data = base64.b64encode(httpx.get(cat_picture).content).decode("utf-8")
    dog_data = base64.b64encode(httpx.get(dog_picture).content).decode("utf-8")
    print(session.prompt([
        (PromptElement.Image, cat_data),
        (PromptElement.Image, dog_data),
        (PromptElement.Text, "What is the difference between these two images?")
    ]))
    print(session.history)

    # -- Artificial prompt check --
    session.artificial_prompt(
        [(PromptElement.Text, "Hello! How are you!")],
        [(PromptElement.Text, "Huh? Who are you and what do you want?")]
    )
    session.artificial_prompt(
        [(PromptElement.Text, "Wait, aren't you going to be friendly?")],
        [(PromptElement.Text, "To you? Bleurgh!")]
    )
    print(session.prompt([(PromptElement.Text, "Why are you being like this?")]))
    print(session.history)
