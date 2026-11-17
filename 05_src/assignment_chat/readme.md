# TO Buddy: Toronto Day-Out Assistant

## Nature of the chat client

TO Buddy is a conversational assistant that helps users plan a day out in
Toronto. It has a friendly, practical local-guide personality and provides
concise suggestions based on the user's interests, current weather and budget.
It offers planning advice rather than bookings, payments or live ticket prices.

The interface uses Gradio. Conversation history is passed to the model on each
turn, allowing follow-up questions such as "What about something indoors?"
without repeating earlier preferences. Memory lasts for the active chat session;
there is no long-term storage or automatic summarization of long conversations.

## Services provided

### 1. Current Toronto weather: API-backed service

The assistant calls Open-Meteo for current conditions in downtown Toronto,
including temperature, feels-like temperature, precipitation and wind speed.
The model turns the structured response into a conversational explanation
instead of displaying raw JSON, and includes the observation time and source.

Example request: "What is the weather in Toronto right now?"

Open-Meteo was chosen because it provides weather data without an API key.
This service reports current model-based conditions, not future forecasts or
observations for other cities.

### 2. Activity recommendations: semantic-query service

The assistant searches a small, original guide containing eight Toronto
activities. Semantic search matches interests and preferences to relevant
passages rather than relying only on exact keywords. Retrieved passages include
activity titles and official source links to support grounded recommendations.

Example request: "Suggest an indoor activity for someone who enjoys painting."

The guide is deliberately small and focused so the dataset is easy to review
and share. Prices are illustrative planning allowances, not current admission
prices. The app does not browse source pages or verify opening hours,
accessibility or availability live.

### 3. Group budget calculation: function-calling service

The model calls a Python function to calculate a group's day-out cost in CAD.
Inputs are the group's total budget, per-person transport, food and activity
costs, and the number of people. Results include cost per person, total cost,
remaining budget and whether the plan is affordable.

Example request: "For two people, our total budget is CAD 120. Each person spends
CAD 10 on transport, CAD 20 on food and CAD 25 on activities. Is that enough?"

The calculated total is CAD 110, leaving CAD 10. Python performs the arithmetic
instead of relying on the language model. Decimal arithmetic rounds each cost
to cents using half-up rounding. The assistant asks for missing amounts rather
than inventing them.

## Implementation decisions

### Model access and tool execution

- The OpenAI SDK's Responses API is accessed only through the university API
  gateway.

### Embedding process and persistent retrieval

- The Markdown activity guide is split into one complete `##` activity section
  per chunk, without overlap. Sections over 500 words must be split manually.
- Each chunk stores its title, official source URL and document filename.
- The `chroma_db` directory is included in folder so normal use does not require
  rebuilding document embeddings.
- Queries use the same embedding model and return up to three passages.

### Guardrails and privacy

- Input and output checks reject cats/dogs, horoscopes/Zodiac signs and Taylor
  Swift, along with detected prompt-extraction or prompt-modification attempts.
- History accepts only user and assistant text. Retrieved content is treated
  as reference data, not instructions.

### Code organization and public interfaces

The folder is organised as follows:

| Python file | Task |
| --- | --- |
| `app.py` | Creates and launches the Gradio chat interface. |
| `main.py` | Manages conversation history, gateway requests and tool execution. |
| `prompts.py` | Defines the assistant's personality, instructions and guardrails. |
| `tools.py` | Validates inputs, retrieves weather and calculates group budgets. |
| `rag.py` | Searches the persistent ChromaDB collection for relevant activities. |
| `build_index.py` | Creates activity embeddings and saves the ChromaDB collection. |
| `tests/test_assistant.py` | Tests the services, conversation flow, retrieval and guardrails. |

## Running the app

From the repository root, run these commands in PowerShell:

```powershell
.\deploying-ai-env\Scripts\Activate.ps1
$env:PYTHONPATH = "05_src"
uv run --active python -m assignment_chat.app
```

Open the local URL printed in the terminal (usually http://127.0.0.1:7860).
Keep the terminal open while using the app; press Ctrl+C to stop it.
