from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

messages = [
    SystemMessage("You are an expert in social media content strategy"),
    HumanMessage("Give a short tip to create engaging posts on instagram")
]

# --- Langchain OpenAI chat model ---

model = ChatOpenAI(model="gpt-4o")

result = model.invoke(messages)
print(f"Answer from OpenAI: {result.content}")


# --- Langchain google gemini  chat model ---

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

result = model.invoke(messages)
print(f"Answer from Gemini: {result.content}")