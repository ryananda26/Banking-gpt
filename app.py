import os
import sys
from langchain.llms import OpenAI
import streamlit as st  

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

from langchain.agents.agent_toolkits import (create_vectorstore_agent,
                                             VectorStoreToolkit,
                                             VectorStoreInfo)

print(sys.path)
os.environ[
    'OPENAI_API_KEY'] = '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'

llm = OpenAI(temperature=0.9)

loader = PyPDFLoader('Sharia-Economic-and-Financial-Review-2022.pdf')
pages = loader.load_and_split()
store = Chroma.from_documents(pages, collection_name='annualreport')

vectorstore_info = VectorStoreInfo(
    name="annual_report",
    description="a banking annual report as a pdf",
    vectorstore=store)

toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

agent_executor = create_vectorstore_agent(llm - llm,
                                          toolkit=toolkit,
                                          verbose=True)

prompt = st.text_input('Input your prompt here..')

if prompt:
    response = llm(prompt)
    response = agent_executor.run(prompt)
    st.write(response)

with st.expander('Document Similarity Search'):
    search = store.similarity_search_with_score(prompt)
    st.write(search[0][0].page_content)