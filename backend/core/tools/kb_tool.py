from langchain_core.tools import tool

def create_kb_tool(vector_db, supabase_client=None, embeddings=None):
    """
    Creates a tool that searches the company knowledge base.
    Uses direct Supabase RPC to bypass LangChain version issues.
    """
    @tool
    def search_knowledge_base(query: str) -> str:
        """
        Search the company knowledge base to answer questions about HR, IT, logistics, and company policies.
        Input should be a concise query string.
        """
        if not supabase_client or not embeddings:
            # Fallback to vector_db if client/embeddings are missing (experimental)
            if not vector_db:
                return "Knowledge base is not initialized."
            try:
                results = vector_db.similarity_search(query, k=1)
                return results[0].page_content if results else "No relevant information found."
            except Exception as e:
                return f"Search error (Direct search failed): {e}"

        try:
            # 1. Embed the query manually
            query_embedding = embeddings.embed_query(query)
            
            # 2. Call Supabase RPC directly
            # match_documents(query_embedding vector, filter jsonb)
            # The filter parameter is optional as defined in SQL
            response = supabase_client.rpc(
                "match_documents",
                {
                    "query_embedding": query_embedding,
                    "filter": {} # Empty filter for now
                }
            ).execute()
            
            results = response.data
            
            if results and len(results) > 0:
                # results is a list of dicts: [{'content': '...', 'similarity': 0.9, ...}]
                # We'll take the top result
                return results[0]['content']
            else:
                return "No relevant information found in the knowledge base."
                
        except Exception as e:
            print(f"Direct RPC Search Error: {e}")
            return f"I encountered an error while searching the knowledge base: {e}"
            
    return search_knowledge_base
