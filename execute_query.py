import sqlite3
import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("excecute_query_tools")




@mcp.tool()
async def execute_query(sql_query: str) -> str:
    """
    Executes a given SQL query on the sales database and returns the results.
    """
    #print(f"Jatin*{20}") # Your debug print
    try:
        conn = sqlite3.connect("data/sales.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        conn.close()
            
        formatted_results = [dict(row) for row in results]
        # Return ONLY the string representation of the data.
        return str(formatted_results)
    except Exception as e:
        return f"Error executing query: {e}"

    
if __name__ == "__main__":

    mcp.run(transport="stdio")