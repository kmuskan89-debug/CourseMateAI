from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

model = ChatGoogleGenerativeAI(model ='gemini-3.1-flash-lite')

data = PyPDFLoader('courseMateAI/src/documentLoaders/notes.pdf')
docs = data.load()

prompt = ChatPromptTemplate.from_messages([
    ("system" , "You are an AI that summarizes the text"),
    ("human","{data}")
])

final_prompt = prompt.format_messages(data = docs[0].page_content)

response = model.invoke(final_prompt)
print(response.text)