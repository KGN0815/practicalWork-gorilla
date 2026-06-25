import json
import os
from typing import Any

from bfcl_eval.constants.eval_config import LOCAL_SERVER_PORT
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)
from bfcl_eval.model_handler.utils import (
    convert_to_tool,
    prepare_openai_tools_for_structured_outputs,
)
from openai import OpenAI


class OpenAICompatibleChatFCHandler(OpenAICompletionsHandler):
    """
    OpenAI-compatible Chat Completions handler for local/remote servers such as
    vLLM. This path sends native `tools`, `tool_choice`, and optional
    per-function `strict=true` instead of BFCL's prompt-formatted OSS FC mode.
    """

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)

        self.client = OpenAI(**self._build_client_kwargs())
        self.model_name = (
            kwargs.get("served_model_name")
            or os.getenv("REMOTE_OPENAI_SERVED_MODEL_NAME")
            or os.getenv("REMOTE_OPENAI_MODEL")
            or model_name
        )
        self.tool_choice = kwargs.get("tool_choice", "auto")
        self.strict_tools = kwargs.get("strict_tools", False)
        self.tool_schema_mode = kwargs.get("tool_schema_mode", "bfcl")
        self.parallel_tool_calls = kwargs.get("parallel_tool_calls", False)
        self.underscore_to_dot = True

    def _tool_call_decoding_mode(self) -> str:
        if self.tool_choice == "auto":
            return "auto-parser-no-vllm-schema-constraint"
        if self.tool_choice == "required":
            return "vllm-structured-outputs-required"
        if self.tool_choice == "none":
            return "no-tool-calls"
        return "endpoint-specific"

    def _build_client_kwargs(self):
        endpoint = os.getenv("LOCAL_SERVER_ENDPOINT", "localhost")
        port = os.getenv("LOCAL_SERVER_PORT", LOCAL_SERVER_PORT)
        base_url = os.getenv("REMOTE_OPENAI_BASE_URL", f"http://{endpoint}:{port}/v1")
        api_key = os.getenv("REMOTE_OPENAI_API_KEY", "EMPTY")
        default_headers = os.getenv("REMOTE_OPENAI_DEFAULT_HEADERS")

        client_kwargs = {"base_url": base_url, "api_key": api_key}
        if default_headers:
            client_kwargs["default_headers"] = json.loads(default_headers)
        return client_kwargs

    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]

        kwargs = {
            "messages": message,
            "model": self.model_name,
            "temperature": self.temperature,
        }

        if len(tools) > 0:
            kwargs["tools"] = tools
            if self.tool_choice:
                kwargs["tool_choice"] = self.tool_choice
            if self.parallel_tool_calls:
                kwargs["parallel_tool_calls"] = True

        inference_data["inference_input_log"] = {
            "message": repr(message),
            "tools": tools,
            "tool_choice": kwargs.get("tool_choice"),
            "parallel_tool_calls": kwargs.get("parallel_tool_calls"),
            "strict_tools": self.strict_tools,
            "tool_schema_mode": self.tool_schema_mode,
            "tool_call_decoding_mode": self._tool_call_decoding_mode(),
            "model": self.model_name,
        }

        return self.generate_with_backoff(**kwargs)

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)
        if self.strict_tools and self.tool_schema_mode == "bfcl":
            self.tool_schema_mode = "strict-compatible"
        inference_data["tools"] = prepare_openai_tools_for_structured_outputs(
            tools,
            strict_tools=self.strict_tools,
            schema_mode=self.tool_schema_mode,
        )
        return inference_data

    def _parse_query_response_FC(self, api_response: Any) -> dict:
        response_data = super()._parse_query_response_FC(api_response)
        self._add_reasoning_content_if_available_FC(api_response, response_data)
        return response_data
