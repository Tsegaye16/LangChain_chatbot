import json
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
import os
from typing import Union, List, Optional
import re

class PDFProcessor:
    """
    Handles processing of PDF and text inputs including:
    - Text extraction from PDFs
    - Text cleaning and normalization
    - Chunking for vector storage
    - Character extraction using LLMs
    - FAISS vector store creation
    
    Uses Google's Generative AI embeddings for text vectorization
    and Gemini model for character extraction.
    """
    
    def __init__(self):
        """Initialize with embeddings model and text splitter configuration"""
        # Google's text embedding model
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Configured text splitter for optimal chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,      # Optimal size for context retention
            chunk_overlap=1000,    # Maintains context between chunks
            length_function=len    # Standard length calculation
        )
        
    def get_pdf_text(self, input_source: Union[List[str], str]) -> str:
        """
        Extract and clean text from either PDF files or direct text input
        
        Args:
            input_source: Either:
                - List of PDF file paths/objects
                - Direct text string
                
        Returns:
            str: Cleaned, normalized text content
            
        Raises:
            ValueError: If input is empty or invalid
        """
        if isinstance(input_source, list):  # Handle PDF input
            return self._extract_pdf_text(input_source)
        return self._clean_text(input_source)  # Handle direct text
    
    def _extract_pdf_text(self, pdf_docs: List[str]) -> str:
        """
        Internal method to extract text from multiple PDFs
        
        Args:
            pdf_docs: List of PDF files (paths or file-like objects)
            
        Returns:
            str: Combined text from all PDF pages
            
        Note:
            Silently skips pages that return None from extract_text()
        """
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  # Handle None returns
        return self._clean_text(text)
    
    def _clean_text(self, text: str) -> str:
        """
        Normalize and clean text content
        
        Args:
            text: Raw input text
            
        Returns:
            str: Text with:
                - Excessive whitespace removed
                - Normalized newlines
                - Consistent spacing
        """
        # Remove excessive whitespace and normalize newlines
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def get_text_chunks(self, text: str) -> List[str]:
        """
        Split text into properly sized chunks for processing
        
        Args:
            text: Input text to split
            
        Returns:
            List[str]: Text chunks meeting size requirements
            
        Raises:
            ValueError: If input text is empty
        """
        if not text.strip():
            raise ValueError("Empty text provided for chunking")
        return self.text_splitter.split_text(text)
    
    def create_vector_store(self, text_chunks: List[str], index_name: str = "faiss_index") -> FAISS:
        """
        Create and persist FAISS vector store from text chunks
        
        Args:
            text_chunks: List of text segments to vectorize
            index_name: Name for the saved index (default: "faiss_index")
            
        Returns:
            FAISS: Created vector store
            
        Raises:
            ValueError: If no text chunks provided
        """
        if not text_chunks:
            raise ValueError("No text chunks provided for vector store creation")
            
        # Generate embeddings and create vector store
        vector_store = FAISS.from_texts(text_chunks, embedding=self.embeddings)
        
        # Persist to disk for later use
        vector_store.save_local(index_name)
        return vector_store
    
    def process_input(self, text: str) -> List[str]:
        """
        Complete text processing pipeline:
        1. Text cleaning
        2. Chunking
        3. Vector store creation
        4. Character extraction
        
        Args:
            text: Input text to process
            
        Returns:
            List[str]: Extracted character names
            
        Note:
            Returns empty list for empty/non-narrative text
        """
        if not text.strip():
            return []
            
        text_chunks = self.get_text_chunks(text)
        self.create_vector_store(text_chunks)
        characters = self.extract_characters(text)
        
        return characters

    def extract_characters(self, text: str) -> List[str]:
        """
        Extract character names from text using LLM analysis
        
        Args:
            text: Narrative text to analyze
            
        Returns:
            List[str]: Identified character names
            
        Special Cases:
            - Returns empty list for non-narrative/short text
            - Handles edge cases with NO_CHARACTERS_FOUND response
        """
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
        
        # Dynamic prompt based on text length
        if len(text.split()) < 50:  # Short text detection
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
        
        # Handle special case response
        if response_text == "NO_CHARACTERS_FOUND":
            return []
        
        # Process valid character list
        characters = [char.strip() for char in response_text.split(',') if char.strip()]
        return characters