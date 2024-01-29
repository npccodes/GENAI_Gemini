import os
from dotenv import load_dotenv
load_dotenv()

Gemini_api_key = os.environ["GEMINI_API_KEY"]
ASTRA_DB_API_ENDPOINT = os.environ["ASTRA_DB_API_ENDPOINT"]
ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]

from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import AstraDB

loader = PyPDFLoader("./Brown_Churchill_Complex_Variables_and_Ap.pdf")
pages = loader.load_and_split()

GoogleGeminiEmbeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=Gemini_api_key )
text_splitter = RecursiveCharacterTextSplitter(chunk_size =1200, chunk_overlap = 400)
splits = text_splitter.split_documents(pages)

vstore = AstraDB(
    embedding=GoogleGeminiEmbeddings,
    collection_name="Complex_variables",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
)

vstore.add_documents(splits)

retriever = vstore.as_retriever()
template = """
context: {context}

question: {question}
"""
prompt = PromptTemplate.from_template(template)
genai_llm = GoogleGenerativeAI(model = "gemini-pro", temperature=0.0, google_api_key=Gemini_api_key)

def pages_adjustments(pages):
    return "\n\n".join(page.page_content for page in pages)

context_chain = retriever | pages_adjustments

main_chain = (
    {"context" : context_chain, "question": RunnablePassthrough()}
    |prompt
    | genai_llm
    | StrOutputParser()
)