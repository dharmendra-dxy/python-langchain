from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

#Example: Write a blog post for a movie - inception

# step1: Get the summary of the movie
    #Step2:
        # chain1: Analyze the plot
        # chain2: Analyze the character
# step3: combine both for final result

# summary:
summary_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie critic."),
        ("human", "Provide a brief summary of the movie {movie}"),
    ]
)

# function : chain1 - Analyze the plot
def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            ("human", "Analyze the plot: {plot}. What are its strength and weaknesses ?"),
        ]
    )
    return plot_template.format_prompt(plot=plot)


# function : chain2 - Analyze the character
def analyze_characters(characters):
    character_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            ("human", "Analyze the character: {characters}. What are its strength and weaknesses ?"),
        ]
    )
    return character_template.format_prompt(characters=characters)


# combine analysis into final verdict:
def combine_verdicts(plot_analysis,character_analysis):
    return f"Plot analysis:\n{plot_analysis}\n\nCharacter analysis:\n{character_analysis}"

# chain branches with LCEL:
plot_branch_chain = (
    RunnableLambda(lambda x: analyze_plot(x)) | model | StrOutputParser()
)

charcter_branch_chain = (
    RunnableLambda(lambda x: analyze_characters(x)) | model | StrOutputParser()
)


# create the combined chain using Langchain Expression Language (LCEL)
chain = (
    summary_template 
    |model 
    | StrOutputParser()
    | RunnableParallel(branches={"plot": plot_branch_chain, "characters": charcter_branch_chain})
    | RunnableLambda(lambda x: combine_verdicts(x["branches"]["plot"], x["branches"]["characters"]))
)

result = chain.invoke({"movie" : "Inception"})

print(result);