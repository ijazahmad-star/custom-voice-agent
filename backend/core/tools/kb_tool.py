from langchain_core.tools import tool

def create_kb_tool(vector_db):
    """
    Creates a tool that searches the company knowledge base.
    """
    @tool
    def search_knowledge_base(query: str) -> str:
        """
        Search the company knowledge base to answer questions about HR, IT, logistics, and company policies.
        Input should be a concise query string.
        """
        if not vector_db:
            return "Knowledge base is not initialized."
            
        results = vector_db.similarity_search(query, k=1)
        if results:
            return results[0].page_content
        else:
            return "No relevant information found in the knowledge base."
            
    return search_knowledge_base
