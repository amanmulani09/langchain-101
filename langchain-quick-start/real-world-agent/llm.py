from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="qwen/qwen3-32b",
    model_provider="groq",
    temprature=0.5,
    timeout=600,
    max_tokens=25000,
    streaming=True,
)

