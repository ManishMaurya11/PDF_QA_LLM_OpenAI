import streamlit as st
import pickle as pkl
import os

from dotenv import load_dotenv

from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


with st.sidebar:
    st.title('LLM chat app from PDF')
    st.markdown('''
                Upload the pdf and query
                ''')
    add_vertical_space(5)
    st.write('Manish')

def main():
    st.write('Chat with document here')
    load_dotenv()

    pdf = st.file_uploader('Upload the PDF', type = 'pdf')

    if pdf:
        st.write(pdf.name)
        text = ""
        for page in PdfReader(pdf).pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embeddings)
        file_name = pdf.name[:-4]

        if os.path.exists(f"{file_name}.pkl"):
            with open(f"{file_name}.pkl", "rb") as f:
                vectorstore = pkl.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(chunks, embeddings)
            file_name = pdf.name[:-4]
            with open(f"{file_name}.pkl", "wb") as f:
                pkl.dump(vectorstore, f)

        # st.write(chunks)
        user_query = st.text_input("Question for PDF")
        st.write(user_query)
        if user_query:
            docs = vectorstore.similarity_search(query=user_query, k=3)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            
            # callback to check the cost per query

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_query)
                print(cb)
            st.write(response)



if  __name__=='__main__':
    main()
