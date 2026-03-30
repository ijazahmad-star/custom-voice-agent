import os
from langchain_groq import ChatGroq
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.chat_models import ChatHuggingFace
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from core.tools.info_tools import save_user_info, retrieve_user_info, list_all_user_info
from core.tools.kb_tool import create_kb_tool
from core.prompts import SYSTEM_PROMPT

def create_agent(config, vector_db, llm_pipeline=None):
    """
    Creates a React agent with the specified configuration and tools.
    
    Args:
        config: VoiceAgentConfig object.
        vector_db: FAISS vector database for RAG.
        llm_pipeline: Optional HuggingFace pipeline for local execution.
    """
    print("Setting up Agent Mode...")
    
    if getattr(config, 'USE_GROQ', False):
        chat_model = ChatGroq(
            model_name=config.LLM_MODEL,
            groq_api_key=os.getenv("groq_api_key")
        )
    else:
        if llm_pipeline is None:
            raise ValueError("LLM pipeline must be provided for local execution.")
        langchain_llm = HuggingFacePipeline(pipeline=llm_pipeline)
        chat_model = ChatHuggingFace(llm=langchain_llm)
    
    # Initialize tools
    kb_tool = create_kb_tool(vector_db)
    tools = [
        kb_tool,
        save_user_info,
        retrieve_user_info,
        list_all_user_info
    ]
    
    # Memory and Agent creation
    memory = MemorySaver()
    agent = create_react_agent(
        chat_model, 
        tools=tools, 
        checkpointer=memory,
        prompt=SYSTEM_PROMPT
    )
    
    return agent
