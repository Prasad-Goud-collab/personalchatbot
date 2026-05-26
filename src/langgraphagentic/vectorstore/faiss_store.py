

# src/langgraphagentic/vectorstore/faiss_store.py

# src/langgraphagentic/vectorstore/faiss_store.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ updated import
from langchain_community.vectorstores import FAISS
import re


class FAISSVectorStore:
    """Handles PDF loading, chunking, embedding and FAISS vector store creation."""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"  # ✅ stronger model
        )
        self.vectorstore = None

    def _clean_text(self, text: str) -> str:
        """Cleans extracted PDF text."""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = text.strip()
        return text

    def build_from_pdf_path(self, pdf_path: str) -> None:
        """
        Loads PDF from file path, chunks it and builds FAISS vectorstore.

        Args:
            pdf_path (str): Absolute or relative path to the PDF file.
        """
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # ✅ Clean each page
        for doc in documents:
            doc.page_content = self._clean_text(doc.page_content)

        # ✅ Remove empty pages
        documents = [doc for doc in documents if len(doc.page_content.strip()) > 50]

        if not documents:
            raise RuntimeError(
                "PDF appears to be empty or image-based. "
                "Please export LinkedIn profile as text-based PDF."
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

    def retrieve(self, query: str, k: int = 6) -> str:
        """
        Retrieves top-k relevant chunks for the query.

        Args:
            query (str): User question
            k (int): Number of chunks to retrieve

        Returns:
            str: Combined context from retrieved chunks
        """
        if not self.vectorstore:
            raise RuntimeError("Vectorstore not built. Check PDF path.")

        docs = self.vectorstore.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context

    def is_ready(self) -> bool:
        """Returns True if vectorstore is built and ready."""
        return self.vectorstore is not None