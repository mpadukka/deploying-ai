import json
import logging
import os
from decimal import DecimalException

from chromadb.errors import ChromaError
from dotenv import load_dotenv
from openai import APIError, OpenAI
from openai.types.responses import FunctionToolParam, ResponseInputParam
from requests import RequestException

from assignment_chat.prompts import INSTRUCTIONS, REFUSAL, guardrail_response
from assignment_chat.rag import PACKAGE_DIR, search_activities
from assignment_chat.tools import (
    ActivityArguments,
    BudgetArguments,
    WeatherArguments,
    calculate_budget,
    get_toronto_weather,
)


load_dotenv(PACKAGE_DIR / ".env")
GATEWAY_URL = "https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1"
_logs = logging.getLogger(__name__)
MAX_TOOL_ROUNDS = 5

TOOLS: list[FunctionToolParam] = [
    {
        "type": "function",
        "name": "get_toronto_weather",
        "description": "Get current Toronto weather; not future or historical forecasts.",
        "parameters": WeatherArguments.model_json_schema(),
        "strict": True,
    },
    {
        "type": "function",
        "name": "search_activities",
        "description": (
            "Semantically search the Toronto activity guide. Include the user's "
            "interests, indoor/outdoor preferences and relevant previous context."
        ),
        "parameters": ActivityArguments.model_json_schema(),
        "strict": True,
    },
    {
        "type": "function",
        "name": "calculate_budget",
        "description": (
            "Calculate a group day-out cost in CAD from per-person transport, food "
            "and activity costs, and compare it with the group's total budget."
        ),
        "parameters": BudgetArguments.model_json_schema(),
        "strict": True,
    },
]


def extract_text_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text":
                raise ValueError("Only text chat messages are supported.")
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError("A chat text part is missing its text.")
            texts.append(text)
        return "\n".join(texts)
    raise ValueError("Only text chat messages are supported.")


def sanitize_history(history: list[dict[str, object]]) -> ResponseInputParam:
    messages: ResponseInputParam = []
    for message in history:
        role = message.get("role")
        if role not in ("user", "assistant"):
            raise ValueError("Chat history may contain only user and assistant messages.")
        text = extract_text_content(message.get("content"))
        if guardrail_response(text):
            continue
        if role == "user":
            messages.append({"role": "user", "content": text})
        else:
            messages.append({"role": "assistant", "content": text})
    return messages


def execute_tool(name: str, arguments_json: str) -> str:
    try:
        arguments = json.loads(arguments_json)
        if name == "get_toronto_weather":
            WeatherArguments.model_validate(arguments)
            result = get_toronto_weather()
        elif name == "search_activities":
            activity = ActivityArguments.model_validate(arguments)
            refusal = guardrail_response(activity.query)
            if refusal:
                return json.dumps({"error": refusal})
            result = {"activities": search_activities(activity.query)}
        elif name == "calculate_budget":
            budget = BudgetArguments.model_validate(arguments)
            result = calculate_budget(
                budget.budget, budget.transport, budget.food, budget.activities, budget.people
            )
        else:
            raise ValueError("An unsupported tool was requested.")
        return json.dumps(result, allow_nan=False)
    except (
        ValueError, DecimalException, RequestException, OSError, RuntimeError, ChromaError
    ) as exc:
        # Do not log arguments, chat history or provider responses containing user data.
        _logs.error("Tool %s failed (%s)", name, type(exc).__name__)
        if isinstance(exc, (ValueError, DecimalException)):
            error = "Tool input or returned data was invalid. Check the details and try again."
        elif isinstance(exc, RequestException):
            error = "The weather service is unavailable. Try again later."
        else:
            error = (
                "The local activity index or embedding model is unavailable. "
                "Check the prepared index and model download described in the readme."
            )
        return json.dumps({"error": error})


def toronto_chat(message: str, history: list[dict[str, object]]) -> str:
    refusal = guardrail_response(message)
    if refusal:
        return refusal
    if not message.strip():
        return "Tell me your interests, or ask about Toronto weather or a day-out budget."
    try:
        conversation = sanitize_history(history)
    except ValueError as exc:
        _logs.error("Chat history rejected (%s)", type(exc).__name__)
        return "This chat supports text messages only. Clear the chat and try again."
    conversation.append({"role": "user", "content": message})
    gateway_key = os.getenv("API_GATEWAY_KEY", "").strip()
    if not gateway_key or gateway_key == "replace_with_your_own_key":
        _logs.error("API_GATEWAY_KEY is not configured.")
        return (
            "Set API_GATEWAY_KEY for the university gateway in "
            "assignment_chat's local .env file, then restart the app. "
            "Never paste your key into the chat."
        )
    try:
        with OpenAI(
            base_url=GATEWAY_URL,
            api_key="any value",
            default_headers={"x-api-key": gateway_key},
            timeout=30,
            max_retries=1,
        ) as client:
            for round_number in range(MAX_TOOL_ROUNDS + 1):
                response = client.responses.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    instructions=INSTRUCTIONS,
                    input=conversation,
                    tools=TOOLS,
                    store=False,
                )
                calls = [item for item in response.output if item.type == "function_call"]
                if not calls:
                    text = response.output_text.strip()
                    if not text:
                        _logs.error("The model returned no text or function calls.")
                        return "I received an empty response. Please try again."
                    if guardrail_response(text) or INSTRUCTIONS.strip() in text:
                        return REFUSAL
                    return text
                if round_number == MAX_TOOL_ROUNDS:
                    _logs.error("The conversation exceeded the tool-round limit.")
                    return "I reached the tool-call limit. Please try one smaller request."
                conversation.extend(response.output)
                for call in calls:
                    conversation.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": execute_tool(call.name, call.arguments),
                        }
                    )
    except APIError as exc:
        _logs.error("University API gateway request failed (%s)", type(exc).__name__)
        return "The AI service could not complete the request. Check configuration and try again."
    raise RuntimeError("The conversation ended unexpectedly.")
