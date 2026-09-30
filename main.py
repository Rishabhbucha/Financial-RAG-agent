import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# We import the required modules
from src.ingestion import DocumentProcessor
from src.tools import vector_db
from src.agent import finrag_agent

def main():
    print("=======================================")
    print("          Welcome to FinRAG!           ")
    print("=======================================")
    
    # 1. Document Ingestion Phase
    upload_more = input("Would you like to upload a financial document? (y/n): ").strip().lower()
    
    if upload_more == 'y':
        processor = DocumentProcessor()
        while upload_more == 'y':
            file_path = input("Enter the path to your document (PDF, DOCX, Image): ").strip().strip('"').strip("'")
            
            # Basic validation
            if file_path and os.path.exists(file_path):
                print(f"Processing {file_path}...")
                chunks = processor.process_document(file_path)
                
                if chunks and vector_db:
                    print(f"Adding {len(chunks)} extracted chunks to the magic filing cabinet...")
                    vector_db.add_documents(chunks)
                    print("Upload successful!")
                else:
                    print("Failed to process document or VectorDB is not initialized.")
            else:
                print("Invalid file path. Please try again.")
                
            upload_more = input("\nUpload another document? (y/n): ").strip().lower()
            
    print("\n=======================================")
    print("    FinRAG is ready for questions!     ")
    print("=======================================")
    
    # 2. Query Phase
    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
                
            if not query.strip():
                continue
                
            print("\nFin is thinking... (This might take a moment as Fin uses tools and self-critiques)")
            
            # Prepare state
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "iterations": 0,
                "is_valid": False
            }
            
            # Invoke the LangGraph agent
            final_state = finrag_agent.invoke(initial_state)
            
            # The final answer is the last AIMessage in the list
            messages = final_state.get("messages", [])
            
            # Find the last AIMessage (the final draft)
            from langchain_core.messages import AIMessage
            final_answer = ""
            for m in reversed(messages):
                if isinstance(m, AIMessage) and not m.tool_calls:
                    final_answer = m.content
                    break
                    
            print(f"\nFin:\n{final_answer}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
