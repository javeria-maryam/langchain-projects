from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

chain = prompt | llm

history = []

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    
    response = chain.invoke({
        "question": question,
        "history": history
    })
    
    print(f"Bot: {response.content}\n")
    
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=response.content))