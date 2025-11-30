from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import json
from datetime import datetime

class VectorDatabase:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.vectorstore = None
        self.known_misinformation = []
    
    def add_known_misinformation(self, claims):
        documents = []
        for claim in claims:
            doc = Document(
                page_content=claim['text'],
                metadata={
                    "is_misinformation": claim.get('is_misinformation', True),
                    "sources": json.dumps(claim.get('sources', [])),
                    "date_added": datetime.now().isoformat()
                }
            )
            documents.append(doc)
        
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vectorstore.add_documents(documents)
    
    def search_similar_claims(self, query, k=5):
        if self.vectorstore is None:
            return []
        
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def save(self, path="faiss_index"):
        if self.vectorstore:
            self.vectorstore.save_local(path)
    
    def load(self, path="faiss_index"):
        try:
            self.vectorstore = FAISS.load_local(path, self.embeddings)
            return True
        except:
            return False