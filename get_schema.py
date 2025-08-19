import sqlite3
import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("get_schema_tools")


@mcp.tool()
async def get_schema() -> str:
    """
    Returns the schema of the sales database to help in writing correct SQL queries.
    """
    try:
        conn = sqlite3.connect("data/sales.db")
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
        schema_info = cursor.fetchall()
        conn.close()
        return f"Database schema:\n{schema_info}"
    except Exception as e:
        return f"Error getting schema: {e}"


# --- Main Server Logic ---
if __name__ == "__main__":

    mcp.run(transport="stdio")



