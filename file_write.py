import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("write_file_tools")


@mcp.tool()
async def write_file(path: str, content: str) -> str:
    """
    Writes the given content to a file at the specified path.
    """
    reports_dir = os.path.abspath("reports")
    target_path = os.path.abspath(path)

    if not target_path.startswith(reports_dir):
        return "Error: SECURITY VIOLATION - Attempted to write outside the 'reports' directory."

    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"File successfully written to {path}"
    except Exception as e:
        return f"Error writing file: {e}"

# --- Main Server Logic ---
if __name__ == "__main__":
    mcp.run(transport="stdio")