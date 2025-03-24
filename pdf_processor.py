from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
import os

class PDFProcessor:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
    def get_pdf_text(self, pdf_docs):
        """Extract text from PDF files"""
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def get_text_chunks(self, text):
        """Split text into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        return text_splitter.split_text(text)
    
    def create_vector_store(self, text_chunks):
        """Create and save FAISS vector store"""
        vector_store = FAISS.from_texts(text_chunks, embedding=self.embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    
    def extract_characters(self, text):
        """Extract character names from text"""
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
        prompt = f"""
        Extract a list of characters from the following text: {text}. 
        Return only a comma-separated list of names.
        """
        response = model.invoke(prompt)
        characters = [char.strip() for char in response.content.split(',') if char.strip()]
        return characters