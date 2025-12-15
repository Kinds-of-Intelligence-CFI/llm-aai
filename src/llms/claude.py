# https://docs.anthropic.com/claude/docs/quickstart-guide
from typing import List, Literal, Optional, Union
from os.path import join

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
                 thinking_budget: Optional[int] = None,
                 ) -> None:
        super().__init__()
        self._client = anthropic.Anthropic(api_key=api_key)
        self._history: list[MessageParam] = []
        self._model = model
        self._system_prompt = system_prompt
        self._thinking_budget = thinking_budget

        # Additional cost tracking for caching and thinking
        self.cache_creation_costs = np.array([])  # Tokens written to cache
        self.cache_read_costs = np.array([])      # Tokens read from cache
        self.thinking_costs = np.array([])        # Thinking/reasoning tokens used

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
        # max_tokens must be greater than thinking_budget (it's the total: thinking + response)
        max_tokens = 1024
        if self._thinking_budget is not None:
            # Ensure max_tokens is greater than thinking budget plus room for response
            max_tokens += self._thinking_budget

        api_params = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": 1.0, #`temperature` may only be set to 1 when thinking is enabled
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

        # Add extended thinking if budget specified
        if self._thinking_budget is not None:
            api_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._thinking_budget
            }

        message = self._client.messages.create(**api_params)
        response_content = message.content

        # Track standard token usage
        self.input_costs = np.append(self.input_costs, message.usage.input_tokens)
        self.output_costs = np.append(self.output_costs, message.usage.output_tokens)

        # Track cache usage (if present)
        cache_creation = getattr(message.usage, 'cache_creation_input_tokens', 0)
        cache_read = getattr(message.usage, 'cache_read_input_tokens', 0)
        self.cache_creation_costs = np.append(self.cache_creation_costs, cache_creation)
        self.cache_read_costs = np.append(self.cache_read_costs, cache_read)

        # Extract thinking tokens and content
        thinking_tokens = 0
        thinking_summary = None
        text_response = None

        for block in response_content:
            if block.type == "thinking":
                thinking_tokens += len(block.thinking.split())  # Approximate token count
                thinking_summary = block.thinking
            elif block.type == "text":
                text_response = block.text

        self.thinking_costs = np.append(self.thinking_costs, thinking_tokens)

        # Append response to history (include both thinking and text if present)
        if resp_prefix is not None:
            self._history.pop()
            response_text = resp_prefix
            if thinking_summary:
                response_text += f"\n[THINKING: {thinking_summary}]\n"
            if text_response:
                response_text += text_response
            else:
                response_text += ''.join([block.text for block in response_content if block.type == "text"])
            self._history.append(MessageParam(role='assistant', content=response_text))
        else:
            self._history.append(MessageParam(role='assistant', content=response_content))

        # Return just the text response (not thinking)
        if text_response:
            return text_response
        else:
            return ''.join([block.text for block in response_content if block.type == "text"])

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

    def save_cost_arrays(self, cost_folder_path: str = "./"):
        """Override parent method to save additional metrics for caching and thinking"""
        # Save standard costs (input/output tokens)
        super().save_cost_arrays(cost_folder_path)

        # Save cache statistics
        np.save(join(cost_folder_path, "costs_cache_creation.npy"), self.cache_creation_costs)
        np.save(join(cost_folder_path, "costs_cache_read.npy"), self.cache_read_costs)

        # Save thinking token usage
        np.save(join(cost_folder_path, "costs_thinking.npy"), self.thinking_costs)

        # Create a summary text file with aggregate statistics
        with open(join(cost_folder_path, "token_usage_summary.txt"), "w") as f:
            f.write("=" * 60 + "\n")
            f.write("TOKEN USAGE SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Total Turns: {len(self.input_costs)}\n\n")

            f.write("STANDARD TOKENS:\n")
            f.write(f"  Input tokens:  {self.input_costs.sum():,} total, {self.input_costs.mean():.1f} avg/turn\n")
            f.write(f"  Output tokens: {self.output_costs.sum():,} total, {self.output_costs.mean():.1f} avg/turn\n\n")

            if self.cache_creation_costs.sum() > 0 or self.cache_read_costs.sum() > 0:
                f.write("CACHE STATISTICS:\n")
                f.write(f"  Cache writes:  {self.cache_creation_costs.sum():,} tokens\n")
                f.write(f"  Cache reads:   {self.cache_read_costs.sum():,} tokens\n")
                cache_hit_rate = (self.cache_read_costs.sum() /
                                 (self.cache_read_costs.sum() + self.input_costs.sum()) * 100
                                 if (self.cache_read_costs.sum() + self.input_costs.sum()) > 0 else 0)
                f.write(f"  Cache hit rate: {cache_hit_rate:.1f}%\n\n")

            if self.thinking_costs.sum() > 0:
                f.write("THINKING TOKENS:\n")
                f.write(f"  Total thinking: {self.thinking_costs.sum():,} tokens\n")
                f.write(f"  Avg per turn:   {self.thinking_costs.mean():.1f} tokens\n\n")

            # Cost estimates (using Sonnet 4.5 pricing as example)
            input_cost = self.input_costs.sum() * 3.0 / 1_000_000
            output_cost = self.output_costs.sum() * 15.0 / 1_000_000
            cache_write_cost = self.cache_creation_costs.sum() * 3.75 / 1_000_000
            cache_read_cost = self.cache_read_costs.sum() * 0.30 / 1_000_000
            thinking_cost = self.thinking_costs.sum() * 11.25 / 1_000_000

            total_cost = input_cost + output_cost + cache_write_cost + cache_read_cost + thinking_cost

            f.write("ESTIMATED COSTS (Sonnet 4.5 pricing):\n")
            f.write(f"  Input tokens:   ${input_cost:.4f}\n")
            f.write(f"  Output tokens:  ${output_cost:.4f}\n")
            if cache_write_cost > 0:
                f.write(f"  Cache writes:   ${cache_write_cost:.4f}\n")
            if cache_read_cost > 0:
                f.write(f"  Cache reads:    ${cache_read_cost:.4f}\n")
            if thinking_cost > 0:
                f.write(f"  Thinking:       ${thinking_cost:.4f}\n")
            f.write(f"  TOTAL:          ${total_cost:.4f}\n")

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
                 system_prompt: Optional[str] = None,
                 thinking_budget: Optional[int] = None
                 ) -> None:
        self._api_key = api_key
        self._model = model
        self._system_prompt = system_prompt
        self._thinking_budget = thinking_budget

    def start_session(self):
        return AnthropicSession(self._api_key, self._model, self._system_prompt, self._thinking_budget)


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
