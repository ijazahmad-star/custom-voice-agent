import sqlite3
import os
from langchain_core.tools import tool

DB_PATH = "data/user_agent.db"

def _get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS user_info (key TEXT PRIMARY KEY, value TEXT)")
    return conn

@tool
def save_user_info(key: str, value: str) -> str:
    """
    Saves a specific piece of information about the user in a SQLite database.
    Use this when the user tells you something they want you to remember (e.g., their name, preferences, or notes).
    
    Args:
        key: The name of the information (e.g., 'user_name', 'favorite_color').
        value: The information to save.
    """
    try:
        with _get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO user_info (key, value) VALUES (?, ?)",
                (key, value)
            )
            conn.commit()
        return f"Successfully saved {key}: {value} (SQLite)"
    except Exception as e:
        return f"Error saving to SQLite: {e}"

@tool
def retrieve_user_info(key: str) -> str:
    """
    Retrieves a specific piece of information about the user from the SQLite database.
    Use this when you need to recall something the user told you before.
    
    Args:
        key: The name of the information to retrieve.
    """
    try:
        with _get_connection() as conn:
            cursor = conn.execute("SELECT value FROM user_info WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return f"The stored value for '{key}' is: {row[0]}"
            else:
                return f"No information found for '{key}'."
    except Exception as e:
        return f"Error retrieving from SQLite: {e}"

@tool
def list_all_user_info() -> str:
    """
    Lists all information currently stored about the user in the SQLite database.
    """
    try:
        with _get_connection() as conn:
            cursor = conn.execute("SELECT key, value FROM user_info")
            rows = cursor.fetchall()
            if not rows:
                return "No user information is currently stored."
            
            info_list = "\n".join([f"- {row[0]}: {row[1]}" for row in rows])
            return f"Here is the stored information from SQLite:\n{info_list}"
    except Exception as e:
        return f"Error listing from SQLite: {e}"
