"""
Interactive RAG Assistant Tester
=================================
Test the RAG system with custom queries using the updated knowledge base.
"""

import os
import sys

# Add parent directory to path so we can import from RAG_Assistant
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RAG_Assistant.vector_store_builder import load_vector_store


def test_queries_with_chromadb():
    """Test the RAG system with ChromaDB vector store."""
    
    print("=" * 70)
    print("  RAG Knowledge Base Tester (ChromaDB)")
    print("=" * 70)
    print("\nLoading vector store...")
    
    try:
        collection = load_vector_store()
        print(f"✓ Loaded {collection.count()} documents\n")
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        print("Run vector_store_builder.py first to build the vector store.")
        return
    
    print("=" * 70)
    print("Enter your questions (or 'quit' to exit)")
    print("=" * 70)
    
    while True:
        print("\n" + "-" * 70)
        query = input("\n💬 Your question: ").strip()
        
        if not query:
            continue
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        # Query the vector store
        try:
            results = collection.query(
                query_texts=[query],
                n_results=2,  # Get top 2 most relevant documents
                include=['documents', 'metadatas', 'distances']
            )
            
            print(f"\n📚 Retrieved Knowledge:")
            print("=" * 70)
            
            for i, (doc_id, doc_text, metadata, distance) in enumerate(
                zip(
                    results['ids'][0],
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ),
                start=1
            ):
                title = metadata.get('title', 'Unknown')
                keywords = metadata.get('keywords', '')
                
                print(f"\n[{i}] {title}")
                print(f"    Keywords: {keywords}")
                print(f"    Relevance Score: {distance:.4f} (lower = more relevant)")
                print(f"\n    Content:")
                print("    " + "-" * 66)
                
                # Print the content with proper formatting
                lines = doc_text.split('\n')
                for line in lines[:20]:  # Show first 20 lines
                    print(f"    {line}")
                
                if len(lines) > 20:
                    print(f"    ... ({len(lines) - 20} more lines)")
                print()
            
        except Exception as e:
            print(f"\n❌ Error during query: {e}")


def test_sample_questions():
    """Test with predefined sample questions."""
    
    print("\n" + "=" * 70)
    print("  Testing with Sample Questions")
    print("=" * 70)
    
    collection = load_vector_store()
    
    sample_questions = [
        "What is enhancing tumor?",
        "Explain peritumoral edema",
        "What does midline shift indicate?",
        "How are MRI sequences used?",
        "What is non-enhancing tumor core?",
        "Tell me about T1 and T2 sequences",
        "How is tumor volume measured?",
        "What is vasogenic edema?",
        "Explain blood-brain barrier disruption",
        "What are the BraTS sub-regions?",
    ]
    
    for i, question in enumerate(sample_questions, 1):
        print(f"\n[{i}] Q: {question}")
        
        results = collection.query(
            query_texts=[question],
            n_results=1,
            include=['metadatas', 'distances']
        )
        
        if results['ids'][0]:
            title = results['metadatas'][0][0].get('title', 'Unknown')
            distance = results['distances'][0][0]
            print(f"    → Best Match: {title} (score: {distance:.4f})")
        print()


if __name__ == "__main__":
    # Show menu
    print("\n" + "=" * 70)
    print("  RAG Knowledge Base Testing Menu")
    print("=" * 70)
    print("\n1. Interactive mode (ask your own questions)")
    print("2. Test with sample questions")
    print("3. Both")
    print()
    
    choice = input("Select option [1-3]: ").strip()
    
    if choice == "1":
        test_queries_with_chromadb()
    elif choice == "2":
        test_sample_questions()
    elif choice == "3":
        test_sample_questions()
        print("\n\nSwitching to interactive mode...\n")
        test_queries_with_chromadb()
    else:
        print("Invalid choice. Running interactive mode...")
        test_queries_with_chromadb()
