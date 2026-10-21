import base64
import logging
import os
import xml.etree.ElementTree as ET
from typing import Dict, List
from openai import OpenAI, BadRequestError

import backoff
import openai
import tiktoken
from requests.exceptions import SSLError

from mm_agents.accessibility_tree_wrap.heuristic_retrieve import filter_nodes
from mm_agents.prompts import *
from mm_agents.response_parser import extract_evidence, parse_pyautogui_response

logger = logging.getLogger("desktopenv.agent")

attributes_ns_ubuntu = "https://accessibility.windows.example.org/ns/attributes"
attributes_ns_windows = "https://accessibility.windows.example.org/ns/attributes"
state_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/state"
state_ns_windows = "https://accessibility.windows.example.org/ns/state"
component_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/component"
component_ns_windows = "https://accessibility.windows.example.org/ns/component"
value_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/value"
value_ns_windows = "https://accessibility.windows.example.org/ns/value"
class_ns_windows = "https://accessibility.windows.example.org/ns/class"
# More namespaces defined in OSWorld, please check desktop_env/server/main.py

SPECIAL_ACTIONS = ("WAIT", "DONE", "FAIL")


# Function to encode the image
def encode_image(image_content):
    return base64.b64encode(image_content).decode('utf-8')


def linearize_accessibility_tree(accessibility_tree, platform="ubuntu"):

    if platform == "ubuntu":
        _attributes_ns = attributes_ns_ubuntu
        _state_ns = state_ns_ubuntu
        _component_ns = component_ns_ubuntu
        _value_ns = value_ns_ubuntu
    elif platform == "windows":
        _attributes_ns = attributes_ns_windows
        _state_ns = state_ns_windows
        _component_ns = component_ns_windows
        _value_ns = value_ns_windows
    else:
        raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")

    filtered_nodes = filter_nodes(ET.fromstring(accessibility_tree), platform)
    linearized_accessibility_tree = ["tag\tname\ttext\tclass\tdescription\tposition (top-left x&y)\tsize (w&h)"]

    # Linearize the accessibility tree nodes into a table format
    for node in filtered_nodes:
        if node.text:
            text = (
                node.text if '"' not in node.text \
                    else '"{:}"'.format(node.text.replace('"', '""'))
            )

        elif node.get("{{{:}}}class".format(class_ns_windows), "").endswith("EditWrapper") \
                and node.get("{{{:}}}value".format(_value_ns)):
            node_text = node.get("{{{:}}}value".format(_value_ns), "")
            text = (node_text if '"' not in node_text \
                        else '"{:}"'.format(node_text.replace('"', '""'))
                    )
        else:
            text = '""'

        linearized_accessibility_tree.append(
            "{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}".format(
                node.tag, node.get("name", ""),
                text,
                node.get("{{{:}}}class".format(_attributes_ns), "") if platform == "ubuntu" else node.get("{{{:}}}class".format(class_ns_windows), ""),
                node.get("{{{:}}}description".format(_attributes_ns), ""),
                node.get('{{{:}}}screencoord'.format(_component_ns), ""),
                node.get('{{{:}}}size'.format(_component_ns), "")
            )
        )

    return "\n".join(linearized_accessibility_tree)


def trim_accessibility_tree(linearized_accessibility_tree, max_tokens):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(linearized_accessibility_tree)
    if len(tokens) > max_tokens:
        linearized_accessibility_tree = enc.decode(tokens[:max_tokens])
        linearized_accessibility_tree += "[...]\n"
    return linearized_accessibility_tree


def _observation_prompt(observation_type, linearized_accessibility_tree=None):
    """Build the user-turn text describing the current observation."""
    if observation_type == "screenshot_a11y_tree":
        return (
            "Given the screenshot and info from accessibility tree as below:\n{}\n"
            "What's the next step that you will do to help with the task?"
        ).format(linearized_accessibility_tree)
    return "Given the screenshot as below. What's the next step that you will do to help with the task?"


def _observation_message(observation_type, base64_image, linearized_accessibility_tree=None):
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": _observation_prompt(observation_type, linearized_accessibility_tree),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "high"
                }
            }
        ]
    }


class PromptAgent:
    """LLM agent that observes screenshots (optionally plus the accessibility
    tree) and emits pyautogui Python code as actions."""

    def __init__(
            self,
            platform="ubuntu",
            model="gpt-4-vision-preview",
            base_url="https://api.openai.com/v1",
            max_tokens=1500,
            top_p=0.9,
            temperature=1.0,
            observation_type="screenshot_a11y_tree",
            max_trajectory_length=3,
            a11y_tree_max_tokens=10000,
            client_password="password",
            api_key=None
    ):
        if observation_type not in OBSERVATION_TYPES:
            raise ValueError(
                "Invalid observation_type: {!r}, must be one of {}".format(
                    observation_type, list(OBSERVATION_TYPES)
                )
            )

        self.platform = platform
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.a11y_tree_max_tokens = a11y_tree_max_tokens
        self.client_password = client_password
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]

        # The SDK client is created on first use so that constructing a
        # PromptAgent never requires network access or a reachable endpoint.
        self._client = None

        self.thoughts = []
        self.actions = []
        self.observations = []
        self.last_evidence = None

        # Prompt templates contain many literal JSON object braces for the
        # structured evidence schema.  ``str.format`` would treat those braces
        # as placeholders and raise ``KeyError``.  Replace only the password
        # token so all JSON braces remain literal.
        self.system_message = SYS_PROMPTS[observation_type].replace(
            "{CLIENT_PASSWORD}", self.client_password
        )

    def predict(self, instruction: str, obs: Dict) -> List:
        """
        Predict the next action(s) based on the current observation.
        """
        system_message = self.system_message + "\nYou are asked to complete the following task: {}".format(instruction)

        # Prepare the payload for the API call
        messages = []

        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                },
            ]
        })

        # Append trajectory
        assert len(self.observations) == len(self.actions) and len(self.actions) == len(self.thoughts) \
            , "The number of observations and actions should be the same."

        if len(self.observations) > self.max_trajectory_length:
            if self.max_trajectory_length == 0:
                _observations = []
                _actions = []
                _thoughts = []
            else:
                _observations = self.observations[-self.max_trajectory_length:]
                _actions = self.actions[-self.max_trajectory_length:]
                _thoughts = self.thoughts[-self.max_trajectory_length:]
        else:
            _observations = self.observations
            _actions = self.actions
            _thoughts = self.thoughts

        for previous_obs, previous_action, previous_thought in zip(_observations, _actions, _thoughts):
            messages.append(_observation_message(
                self.observation_type,
                previous_obs["screenshot"],
                previous_obs["accessibility_tree"],
            ))
            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": previous_thought.strip() if len(previous_thought) > 0 else "No valid action"
                    },
                ]
            })

        # Current observation
        base64_image = encode_image(obs["screenshot"])
        linearized_accessibility_tree = None
        if self.observation_type == "screenshot_a11y_tree":
            linearized_accessibility_tree = linearize_accessibility_tree(
                accessibility_tree=obs["accessibility_tree"], platform=self.platform
            )
            logger.debug("LINEAR AT: %s", linearized_accessibility_tree)
            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(
                    linearized_accessibility_tree, self.a11y_tree_max_tokens
                )

        self.observations.append({
            "screenshot": base64_image,
            "accessibility_tree": linearized_accessibility_tree
        })
        messages.append(_observation_message(
            self.observation_type, base64_image, linearized_accessibility_tree
        ))

        try:
            response = self.call_llm(messages)
        except Exception as e:
            logger.error("Failed to call" + self.model + ", Error: " + str(e))
            response = ""

        logger.info("RESPONSE: %s", response)

        try:
            actions = self.parse_actions(response)
            self.thoughts.append(response)
        except ValueError as e:
            print("Failed to parse action from response", e)
            actions = None
            self.thoughts.append("")

        return response, actions

    def _build_client(self):
        """Create the OpenAI SDK client, normalizing the endpoint to end in /v1/v4."""
        base_url = self.base_url
        if not (base_url.endswith("/v1") or base_url.endswith("/v4")):
            base_url = f"{base_url}/v1"
        return OpenAI(api_key=self.api_key, base_url=base_url)

    @property
    def client(self):
        if self._client is None:
            self._client = self._build_client()
        return self._client

    @backoff.on_exception(
        backoff.constant,
        (
                # General exceptions
                SSLError,

                # OpenAI exceptions
                openai.RateLimitError,
                openai.BadRequestError,
                openai.InternalServerError
        ),
        interval=30,
        max_tries=10
    )
    def call_llm(self, messages):
        extra_body = {
            "enable_thinking": True,
            "chat_template_kwargs": {"enable_thinking": True},
            # "top_k": 20,
            # "min_p": 0.0,
            # "repetition_penalty": 1.0,
        }

        logger.info("Generating content with GPT model: %s", self.model)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=1.0,
                seed=42,
                top_p=0.9,
                extra_body=extra_body,
            )
        except openai.BadRequestError as e:
            if getattr(e, "code", None) != "context_length_exceeded":
                raise
            logger.error("Context length exceeded. Retrying with a smaller context.")
            shortened = [messages[0], messages[-1]]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=shortened,
                max_tokens=self.max_tokens,
                temperature=1.0,
                top_p=0.9,
                extra_body=extra_body,
            )

        return response.choices[0].message.content

    def parse_actions(self, response: str):
        """Parse the LLM response into evidence screenshots + pyautogui code."""
        self.last_evidence = extract_evidence(response)
        actions = parse_pyautogui_response(response)
        self.actions.append(actions)
        return actions

    def reset(self, _logger=None, vm_ip=None, **kwargs):
        global logger
        logger = _logger if _logger is not None else logging.getLogger("desktopenv.agent")

        self.vm_ip = vm_ip

        self.thoughts = []
        self.actions = []
        self.observations = []
        self.last_evidence = None