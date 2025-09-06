from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# --- Langchain google gemini  chat model ---
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

chat_history = [] 

# initial system message:
system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)

# chat loop:
while True:
    query= input("You : ")
    if(query.lower() == "exit") :
        break
    chat_history.append(HumanMessage(content=query))

    # GEN AI response using history:
    result  = model.invoke(chat_history);
    response = result.content

    chat_history.append(AIMessage(content=response))

    print(f"AI response: {response}")

print("---- Chat history ----")
print(chat_history)

