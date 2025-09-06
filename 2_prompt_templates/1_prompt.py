from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv();

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# # # Example 1: Promopt with template:
# template= "Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skill} as a key strength. Keep it 4 lines max." 

# # Prompt that lang-chain understands:
# prompt_template = ChatPromptTemplate.from_template(template)

# prompt = prompt_template.invoke({
#     "tone" : "energetic",
#     "company" : "samsung",
#     "position" : "software enginner",
#     "skill" : "frontend"
# })

# result = model.invoke(prompt)
# print(result.content)


# Example 2: Promopt with system and human message
messages = [
    ("system", "You are a comedian that tells a joke about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]

promopt_template = ChatPromptTemplate.from_messages(messages)

prompt = promopt_template.invoke({
    "topic" : "Engineering",
    "joke_count" : "4",
})

result = model.invoke(prompt)
print(result.content)

