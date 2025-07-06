import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

class RAGModel:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7, top_p=0.85)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db_dir = "chroma_db"
        os.makedirs(self.db_dir, exist_ok=True)

    def predict(self, message, filename):

        if not filename:
            raise Exception('No file to make task')
        
        persist_directory = os.path.join(self.db_dir, os.path.splitext(filename)[0])

        if os.path.exists(persist_directory):
            vector_store = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
        else:
            documents = self.get_documents_from_file(filename)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)

            vector_store = Chroma.from_documents(texts, self.embeddings, persist_directory=persist_directory)
            vector_store.persist()

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )

        result = qa_chain(message)

        return {
            'message': result['result'],
        }
    
    def get_documents_from_file(self, filename):
        filepath = os.path.join('uploads', filename)
        
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(filepath)
        elif filename.endswith('.csv'):
            loader = CSVLoader(filepath)
        else:
            loader = TextLoader(filepath)
        
        return loader.load()
