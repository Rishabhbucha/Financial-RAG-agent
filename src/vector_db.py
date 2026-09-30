import os
from typing import List, Dict, Any
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

class VectorDBManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        # Initialize the Sentence Transformer embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Create or get the collection
        self.collection_name = "finrag_documents"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        Adds chunked documents to ChromaDB.
        """
        if not chunks:
            print("No chunks to add.")
            return

        documents = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Create unique IDs based on source and chunk_id
        ids = [f"{m['source']}_{m['chunk_id']}" for m in metadatas]

        # Embed the documents
        embeddings = self.embeddings.embed_documents(documents)

        # Store in ChromaDB
        self.collection.upsert(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(chunks)} chunks to Vector DB.")

    def search(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the Vector DB for the most relevant chunks.
        """
        # Embed the search query
        query_embedding = self.embeddings.embed_query(query)

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        search_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                search_results.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results and results['distances'] else None
                })
        
        return search_results

if __name__ == "__main__":
    db = VectorDBManager()
    print("VectorDBManager initialized successfully.")
