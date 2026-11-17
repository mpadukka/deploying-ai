import logging

import gradio as gr

from assignment_chat.main import toronto_chat


def create_app() -> gr.ChatInterface:
    return gr.ChatInterface(
        fn=toronto_chat,
        title="TO Buddy - Toronto Day-Out Assistant",
        description=(
            "Plan a Toronto day out with current weather, a small activity guide, "
            "and a CAD budget calculator. Guide prices are illustrative, not live "
            "admission prices. Do not enter personal information or API keys."
        ),
        examples=[
            "What is the weather in Toronto right now?",
            "Suggest an indoor activity for someone who enjoys art.",
            (
                "For two people, our total budget is $120. Each person spends "
                "$10 on transport, $20 on food and $25 on activities. Is that enough?"
            ),
        ],
        analytics_enabled=False,
        save_history=False,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_app().launch(server_name="127.0.0.1", share=False, show_error=False)
