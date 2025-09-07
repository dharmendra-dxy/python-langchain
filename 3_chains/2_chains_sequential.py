from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# prmopt_template1:
facts_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows facts about {animal}."),
        ("human", "Tell me {count} facts."),
    ]
)

# prmopt_template2:
translation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a translator and convert the provided text into {language}."),
        ("human", "Translate the following text to {language} :{text} ."),
    ]
)

# Define additional processing steps using RunnableLambda
count_words = RunnableLambda(lambda x: f"word count: {len(x.split())}\n{x}")
prepare_for_translation = RunnableLambda(lambda output: {"text": output, "language": "hindi"})

# chain:
chain = facts_template | model | StrOutputParser() |prepare_for_translation| translation_template | model | StrOutputParser()

result = chain.invoke({"animal": "cat", "count": 2})

print(result)

