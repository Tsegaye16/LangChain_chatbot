import json
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
import os
from typing import Union, List, Optional
import re

class PDFProcessor:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=1000,
            length_function=len
        )
        
    def get_pdf_text(self, input_source: Union[List[str], str]) -> str:
        """
        Extract text from either PDF files or direct text input
        
        Args:
            input_source: Either list of PDF files or text string
            
        Returns:
            Extracted text as a single string
        """
        if isinstance(input_source, list):  # PDF files
            return self._extract_pdf_text(input_source)
        return self._clean_text(input_source)  # Direct text
    
    def _extract_pdf_text(self, pdf_docs: List[str]) -> str:
        """Extract text from PDF files"""
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  # Handle None returns
        return self._clean_text(text)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace and normalize newlines
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def get_text_chunks(self, text: str) -> List[str]:
        """Split text into chunks with improved handling"""
        if not text.strip():
            raise ValueError("Empty text provided for chunking")
        return self.text_splitter.split_text(text)
    
    def create_vector_store(self, text_chunks: List[str], index_name: str = "faiss_index") -> FAISS:
        """Create and save FAISS vector store with validation"""
        if not text_chunks:
            raise ValueError("No text chunks provided for vector store creation")
            
        vector_store = FAISS.from_texts(text_chunks, embedding=self.embeddings)
        vector_store.save_local(index_name)
        return vector_store
    
    def process_input(self, text):
        """Process direct text input with better character extraction"""
        if not text.strip():
            return []
            
        text_chunks = self.get_text_chunks(text)
        self.create_vector_store(text_chunks)
        characters = self.extract_characters(text)
        
        return characters

    def extract_characters(self, text):
        """Extract character names from text with better handling of short/non-narrative text"""
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
        
        # First check if the text is too short or doesn't contain narrative content
        if len(text.split()) < 50:  # If less than 50 words
            prompt = f"""
            Analyze this text to determine if it contains any narrative content with characters:
            {text}
            
            If this appears to be a complete story or narrative with multiple characters, 
            return a comma-separated list of character names.
            Otherwise return exactly: NO_CHARACTERS_FOUND
            """
        else:
            prompt = f"""
            Extract a list of characters from the following narrative text: {text}. 
            Return only a comma-separated list of names.
            For non-narrative text, return exactly: NO_CHARACTERS_FOUND
            """
        
        response = model.invoke(prompt)
        response_text = response.content.strip()
        
        if response_text == "NO_CHARACTERS_FOUND":
            return []
        
        characters = [char.strip() for char in response_text.split(',') if char.strip()]
        return characters