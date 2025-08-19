import asyncio
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.callbacks.stdout import StdOutCallbackHandler

load_dotenv()


async def main():
    #  Spin up your MCP tools
    client = MultiServerMCPClient({
        "get_schema": {
            "command": "python",
            "args": ["get_schema.py"],
            "transport": "stdio",
        },
        "execute_query": {
            "command": "python",
            "args": ["execute_query.py"],
            "transport": "stdio",
        },
        "write_file": {
            "command": "python",
            "args": ["file_write.py"],
            "transport": "stdio",
        },
    })
    tools = await client.get_tools()
    print("🔧 Available tools:", [t.name for t in tools])

    #  Initialize your LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)


    translator_agent = create_react_agent(
        llm,
        tools=[t for t in tools if t.name == "get_schema"],
        # FIX: Make the prompt stricter to contain the agent's scope
        prompt=(
            "You are an expert SQL writer. Your sole purpose is to create a SQL query.\n"
            "1. First, you MUST use the 'get_schema' tool to see the database structure.\n"
            "2. Then, based on the schema and the user's question, construct the correct SQL query.\n"
            "3. Your final output MUST be only the SQL query string, inside a single code block, and nothing else. Do not attempt to execute it or explain it."
        ),
        verbose=True,
        callbacks=[StdOutCallbackHandler()],
    )

# ...

    executor_agent = create_react_agent(
        llm,
        tools=[t for t in tools if t.name == "execute_query"],
        # Make the prompt more explicit about when to stop.
        prompt=(
            "You are a SQL execution machine. Your sole purpose is to execute a SQL query you are given. "
            "1. You will receive a single SQL query.\n"
            "2. You MUST immediately call the `execute_query` tool with the exact query string.\n"
            "3. After the tool returns the result, you MUST immediately output that result as your final answer. Do not add any commentary, explanation, or further tool calls."
        ),
        verbose=True,
        callbacks=[StdOutCallbackHandler()],
    )

    reporter_agent = create_react_agent(
        llm,
        tools=[t for t in tools if t.name == "write_file"],
        prompt=(
            "You are a senior business analyst. You will receive raw query results.\n"
            "Analyze them to answer the user's original question in concise Markdown.\n"
            "You MUST then call the 'write_file' tool with:\n"
            "  path='reports/sales_brief.md'\n"
            "  content=<your markdown report>\n"
            "Return only via the tool call."
        ),
        verbose=True,
        callbacks=[StdOutCallbackHandler()],
    )




    async def translator_node(state: MessagesState) -> dict:
        # This node is working correctly, but for consistency and best practice,
        # let's adjust it to only append its *new* messages to the state.
        print("\n--- TRANSLATOR NODE INPUT ---")
        print(state["messages"][-1].content)
        result = await translator_agent.ainvoke({"messages": state["messages"]})
        print("--- TRANSLATOR NODE OUTPUT ---")
        for m in result["messages"]:
            print(m.type, ":", m.content)
        # Return only the messages this agent added
        return {"messages": result["messages"][len(state["messages"]):]}

    async def executor_node(state: MessagesState) -> dict:
        print("\n--- EXECUTOR NODE INPUT ---")
        # FIX: Extract ONLY the SQL query from the last message
        sql_query = state["messages"][-1].content
        print(sql_query)

        # Invoke the agent with a fresh history containing only the SQL query
        result = await executor_agent.ainvoke({"messages": [HumanMessage(content=sql_query)]})

        print("--- EXECUTOR NODE OUTPUT ---")
        for m in result["messages"]:
            print(m.type, ":", m.content)
        
        # Append only the new messages from this agent's work (its tool calls and final answer)
        # We slice off the first message, which was the input we just provided.
        return {"messages": result["messages"][1:]}

    async def reporter_node(state: MessagesState) -> dict:
        print("\n--- REPORTER NODE INPUT ---")
        # Get the original question from the start and the DB results from the end
        original_question = state["messages"][0].content
        query_results = state["messages"][-1].content

        # Create a clear, specific task for the reporter agent
        reporter_task = (
            f"The user's original question was: '{original_question}'.\n\n"
            f"Analyze the following data and generate a concise markdown report:\n\n{query_results}"
        )
        print(reporter_task)

        # Invoke the agent with a fresh history containing only this task
        result = await reporter_agent.ainvoke({"messages": [HumanMessage(content=reporter_task)]})

        print("--- REPORTER NODE OUTPUT ---")
        for m in result["messages"]:
            print(m.type, ":", m.content)

       
        final_md = ""
        # The agent should call the 'write_file' tool. The result is in the ToolMessage.
        for msg in result["messages"]:
            if msg.type == "tool":
                final_md = msg.content
                break # Found the tool output

        if final_md:
            try:
                # We no longer need this debug write, as the agent tool call handles it.
                # The agent's output confirms success.
                print("💾 Reporter agent successfully called the 'write_file' tool.")
                print(f"Tool output: {final_md}")
            except Exception as e:
                print("❌ DEBUG: Error processing reporter output:", e)

        # FIX: Append only the new messages from the reporter agent
        return {"messages": result["messages"][1:]}




    # Build a static, sequential StateGraph over MessagesState
    workflow = StateGraph(MessagesState)
    workflow.add_node("translator", translator_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("reporter", reporter_node)

    workflow.add_edge(START,"translator")
    workflow.add_edge("translator","executor")
    workflow.add_edge("executor", "reporter")
    workflow.add_edge("reporter", END)

    app = workflow.compile()

    # 7️⃣ Invoke the workflow and inspect output
    print("\n" + "=" * 50)
    print("--- RUNNING AI SALES ASSISTANT ---")
    initial_state = {
        "messages": [
            HumanMessage(content="What were our top 3 best-selling products by quantity in Q2 2025?")
        ]
    }
    final_state = await app.ainvoke(initial_state)

    print("\n✅ --- Workflow Complete ---")
    print("Final Agent Output:", final_state["messages"][-1].content)
    #print("If the reporter tool was called correctly, you should now have:")
    #print("  reports/sales_brief.md")

if __name__ == "__main__":
    asyncio.run(main())













# import os
# from dotenv import load_dotenv
# from langgraph.graph import StateGraph, START, END, MessagesState
# from langchain_core.messages import HumanMessage
# from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.prebuilt import create_react_agent
# from langchain_core.callbacks.stdout import StdOutCallbackHandler

# load_dotenv()


# def main():
#     # 2️⃣ Spin up your MCP tools
#     client = MultiServerMCPClient({
#         "get_schema": {
#             "command": "python",
#             "args": ["get_schema.py"],
#             "transport": "stdio",
#         },
#         "execute_query": {
#             "command": "python",
#             "args": ["execute_query.py"],
#             "transport": "stdio",
#         },
#         "write_file": {
#             "command": "python",
#             "args": ["file_write.py"],
#             "transport": "stdio",
#         },
#     })

#     tools = client.get_tools()
#     #print("🔧 Available tools:", [t.name for t in tools])

#     # 3️⃣ Initialize your LLM
#     llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.7)

#     # Translator Agent
#     translator_agent = create_react_agent(
#         llm,
#         tools=[t for t in tools if t.name == "get_schema"],
#         prompt=(
#             "You are an expert SQL writer. First call the 'get_schema' tool to retrieve table schemas, "
#             "then construct the exact SQL query the user needs."
#         ),
#         verbose=True,
#         callbacks=[StdOutCallbackHandler()],
#     )

#     # Executor Agent
#     executor_agent = create_react_agent(
#         llm,
#         tools=[t for t in tools if t.name == "execute_query"],
#         prompt=(
#             "You are an expert SQL executor. You will receive a SQL query in plain text. "
#             "Call the 'execute_query' tool with:\n"
#             "  sql_query=<the SQL query string>\n"
#             "Do not rephrase the SQL, do not explain it — just run it and return the results."
#         ),
#         verbose=True,
#         callbacks=[StdOutCallbackHandler()],
#     )

#     # Reporter Agent
#     reporter_agent = create_react_agent(
#         llm,
#         tools=[t for t in tools if t.name == "write_file"],
#         prompt=(
#             "You are a senior business analyst. You will receive raw query results. "
#             "Analyze them to answer the user's original question in concise Markdown. "
#             "Then, call the 'write_file' tool with:\n"
#             "  path='reports/sales_brief.md'\n"
#             "  content=<your markdown report>"
#         ),
#         verbose=True,
#         callbacks=[StdOutCallbackHandler()],
#     )

#     # Translator Node
#     def translator_node(state: MessagesState) -> dict:
#         print("\n--- TRANSLATOR NODE INPUT ---")
#         print(state["messages"][-1].content)
#         result = translator_agent.invoke({"messages": state["messages"]})
#         print("--- TRANSLATOR NODE OUTPUT ---")
#         for m in result["messages"]:
#             print(m.type, ":", m.content)
#         return {"messages": result["messages"]}

#     # Executor Node
#     def executor_node(state: MessagesState) -> dict:
#         print("\n--- EXECUTOR NODE INPUT ---")
#         print(state["messages"][-1].content)
#         result = executor_agent.invoke({"messages": state["messages"]})
#         print("--- EXECUTOR NODE OUTPUT ---")
#         for m in result["messages"]:
#             print(m.type, ":", m.content)
#         return {"messages": result["messages"]}

#     # Reporter Node
#     def reporter_node(state: MessagesState) -> dict:
#         print("\n--- REPORTER NODE INPUT ---")
#         print(state["messages"][-1].content)
#         result = reporter_agent.invoke({"messages": state["messages"]})
#         print("--- REPORTER NODE OUTPUT ---")
#         for m in result["messages"]:
#             print(m.type, ":", m.content)

#         # Save debug Markdown
#         final_md = result["messages"][-1].content
#         if final_md.strip():
#             try:
#                 os.makedirs("reports", exist_ok=True)
#                 with open("reports/debug_sales_brief.md", "w", encoding="utf-8") as f:
#                     f.write(final_md)
#                 print("💾 DEBUG: Markdown saved to reports/debug_sales_brief.md")
#             except Exception as e:
#                 print("❌ DEBUG: Failed to save debug file:", e)

#         return {"messages": result["messages"]}

#     # Build workflow
#     workflow = StateGraph(MessagesState)
#     workflow.add_node("translator", translator_node)
#     workflow.add_node("executor", executor_node)
#     workflow.add_node("reporter", reporter_node)

#     workflow.add_edge(START, "translator")
#     workflow.add_edge("translator", "executor")
#     workflow.add_edge("executor", "reporter")
#     workflow.add_edge("reporter", END)

#     app = workflow.compile()

#     # Run
#     print("\n" + "=" * 50)
#     print("--- RUNNING AI SALES ASSISTANT ---")
#     initial_state = {
#         "messages": [
#             HumanMessage(content="What were our top 3 best-selling products by quantity in Q2 2025?")
#         ]
#     }
#     final_state = app.invoke(initial_state)

#     print("\n✅ --- Workflow Complete ---")
#     print("Final Agent Output:", final_state["messages"][-1].content)
#     print("If the reporter tool was called correctly, you should now have:")
#     print("  reports/sales_brief.md")


# if __name__ == "__main__":
#     main()




# import asyncio
# from langgraph.graph import StateGraph, START, END, MessagesState
# from langchain_core.messages import HumanMessage, ToolMessage
# from langchain_mistralai.chat_models import ChatMistralAI
# from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.prebuilt import create_react_agent
# import getpass
# from dotenv import load_dotenv
# import os
# load_dotenv()
# # if "MISTRAL_API_KEY" not in os.environ:
# #     os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter your Mistral API key: ")

# async def main():

#     # 1. Spin up your MCP tools
#     client = MultiServerMCPClient({
#         "get_schema": {
#             "command": "python",
#             "args": ["get_schema.py"],
#             "transport": "stdio",
#         },
#         "execute_sql_query": {
#             "command": "python",
#             "args": ["execute_query.py"],
#             "transport": "stdio",
#         },
#         "write_file": {
#             "command": "python",
#             "args": ["file_write.py"],
#             "transport": "stdio",
#         },
#     })
#     tools = await client.get_tools()
#     print("🔧 Available tools:", [t.name for t in tools])


#     # 2. Initialize your LLM
#     #llm = ChatMistralAI(model="mistral-large-latest", temperature=0.7)
#     llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.7)
#     # 3. Create three ReAct agents with simple string prompts
#     translator_agent = create_react_agent(
#         llm,
#         tools=[t for t in tools if t.name == "get_schema"],
#         prompt=(
#             "You are an expert SQL writer. First call the 'get_schema' tool to retrieve table schemas, "
#             "then construct the exact SQL query the user needs."
#         )
#     )

#     executor_agent = create_react_agent(
#         llm,
#         tools=[t for t in tools if t.name == "execute_sql_query"],
#         prompt=(
#             "You are a database operator. You will receive exactly one SQL query string. "
#             "Use the 'execute_sql_query' tool to run it and return the raw query results."
#         )
#     )

#     reporter_agent = create_react_agent(
#         llm,
#         tools=[t for t in tools if t.name == "file_write"],
#         prompt=(
#             "You are a senior business analyst. You will receive the raw query results. "
#             "Analyze the data to answer the user's original question in a concise Markdown report, "
#             "then save it by calling the 'write_file' tool with path 'reports/sales_brief.md'."
#         )
#     )

#     # 4. Define each node: they take the full message history and return updated history.
#     def translator_node(state: MessagesState) -> dict:
#         resp = translator_agent.invoke({"messages": state["messages"]})
#         return {"messages": resp["messages"]}

#     def executor_node(state: MessagesState) -> dict:
#         resp = executor_agent.invoke({"messages": state["messages"]})
#         return {"messages": resp["messages"]}

#     def reporter_node(state: MessagesState) -> dict:
#         resp = reporter_agent.invoke({"messages": state["messages"]})
#         return {"messages": resp["messages"]}

#     # 5. Build a sequential StateGraph over MessagesState
#     workflow = StateGraph(MessagesState)
#     workflow.add_node("translator", translator_node)
#     workflow.add_node("executor", executor_node)
#     workflow.add_node("reporter", reporter_node)

#     workflow.add_edge(START, "translator")
#     workflow.add_edge("translator", "executor")
#     workflow.add_edge("executor", "reporter")
#     workflow.add_edge("reporter", END)

#     app = workflow.compile()

#     # 6. Kick off the pipeline
#     print("\n" + "=" * 50)
#     print("--- RUNNING AI SALES ASSISTANT ---")
#     initial_state = {
#         "messages": [
#             HumanMessage(content="What were our top 3 best-selling products by quantity in Q2 2025?")
#         ]
#     }

#     # 🔑 Remove the await here
#     final_state = app.invoke(initial_state)

#     print("\n✅ --- Workflow Complete ---")
#     print("Final Agent Output:", final_state["messages"][-1].content)
#     print("Report should be saved in 'reports/sales_brief.md'.")
# if __name__ == "__main__":
#     asyncio.run(main())




# import operator
# from typing import Literal, TypedDict, Annotated, List
# import asyncio
# from langgraph.graph import MessagesState
# from langgraph.graph import StateGraph, END
# from langchain.agents import AgentExecutor
# from langchain_core.messages import HumanMessage, BaseMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_mistralai.chat_models import ChatMistralAI
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph import MessagesState
# from langgraph.types import Command
# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.prebuilt import create_react_agent

# from dotenv import load_dotenv


# # class AgentState(TypedDict):
# #     messages: Annotated[List[BaseMessage], operator.add]


# # def create_agent_executor(llm, tools, system_prompt):
# #     """
# #     Factory function to create a ReAct agent executor.
# #     """
# #     # --- KEY CHANGE HERE ---
# #     # We are renaming the variable from 'input' to 'question' to avoid conflicts.
# #     prompt = ChatPromptTemplate.from_messages([
# #         ("system", system_prompt),
# #         ("human", "{question}"),
# #         ("placeholder", "{agent_scratchpad}"),
# #     ])

# #     agent = create_react_agent(llm, tools, prompt=prompt)
# #     return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)



# async def main():
#     load_dotenv()

#     client = MultiServerMCPClient({
#         "execute_sql_query": {
#             "command": "python",
#             "args": ["execute_query.py"],
#             "transport": "stdio",
#         },
#         "get_schema": {
#             "command": "python",
#             "args": ["get_schema.py"],
#             "transport": "stdio",
#         },
#         "file_write": {
#             "command": "python",
#             "args": ["file_write.py"],
#             "transport": "stdio",
#         },
#     })

#     tools = await client.get_tools()

#     llm = ChatMistralAI(model="mistral-large-latest", temperature=0.7)

#     translator_agent = create_react_agent(
#     llm,
#     tools=[tool for tool in tools if tool.name == "get_schema"],
#     prompt="""You are an expert SQL writer. Your goal is to write a SQL query to answer the user's question.
#         To do this, you MUST first use the 'get_schema' tool to understand the database tables and columns.
#         Then, based on the schema you receive, construct the final SQL query."""
#     )

#     executor_agent = create_react_agent(
#         llm,
#         [tool for tool in tools if tool.name == "execute_sql_query"],
#         prompt="""You are a database execution agent. You will be given a single SQL query.
#         Your job is to use the 'execute_sql_query' tool to run the query.
#         Pass the exact SQL query you receive as input to the tool."""
#     )


#     reporter_agent = create_react_agent(
#         llm,
#         [tool for tool in tools if tool.name == "file_write"],
#         prompt="""You are a senior business analyst. You will receive data from a database query.
#         1. Analyze this data to answer the user's original question.
#         2. Write a concise summary report in markdown format.
#         3. Use the 'file_write' tool to save your markdown report to a file. The path should be 'reports/sales_brief.md'."""
#     )   
        


#     def get_next_node(last_message, goto, role):
#         return goto
        


#     def translator_node(state: MessagesState) -> Command[Literal["executor_agent"]]:
#         # Using 'question' as the key in the invoke call.
#         result = translator_agent.invoke(state)
#         result["messages"][-1] = HumanMessage(content=result["messages"][-1].content, name="Translator")
#         return Command(update={"messages": state["messages"] + result["messages"]}, goto=get_next_node(result["messages"][-1], "executor_agent", "translator_agent"))

#     def executor_node(state: MessagesState) -> Command[Literal["file_writer_agent"]]:
#         # Using 'question' as the key in the invoke call.
#         result = executor_agent.invoke(state)
#         result["messages"][-1] = HumanMessage(content=result["messages"][-1].content, name="Executor")
#         return Command(update={"messages": state["messages"] + result["messages"]}, goto=get_next_node(result["messages"][-1], "reporter_agent", "executor_agent"))
    
#     def reporter_node(state: MessagesState) -> Command[Literal[END]]:
#         # Using 'question' as the key in the invoke call.
#         result = reporter_agent.invoke(state)
#         result["messages"][-1] = HumanMessage(content=result["messages"][-1].content, name="Reporter")
#         return Command(update={"messages": state["messages"] + result["messages"]}, goto=get_next_node(result["messages"][-1], END, "reporter_agent"))


#     workflow = StateGraph(MessagesState)
#     workflow.add_node("translator", translator_node)
#     workflow.add_node("executor", executor_node)
#     workflow.add_node("reporter", reporter_node)
    
#     workflow.add_edge(START, "translator")
#     # workflow.set_entry_point("translator")
#     # workflow.add_edge("translator", "executor")
#     # workflow.add_edge("executor", "reporter")
#     # workflow.add_edge("reporter", END)

#     app = workflow.compile()

#     print("\n" + "=" * 50)
#     print("--- RUNNING AI SALES ASSISTANT (with Mistral AI) ---")
#     user_query = "What were our top 3 best-selling products by quantity in Q2 2025?"
#     initial_state = {"messages": [HumanMessage(content=user_query)]}

#     final_state = await app.invoke(initial_state)

#     print("\n\n✅ --- Workflow Complete ---")
#     print("Final Status:", final_state['messages'][-1].content)
#     print("Check the 'reports/sales_brief.md' file for the output.")


# if __name__ == "__main__":
#     asyncio.run(main())


# import operator
# from typing import TypedDict, Annotated, List
# import asyncio
# from langgraph.graph import StateGraph, END
# from langchain.agents import AgentExecutor
# from langchain_core.messages import HumanMessage, BaseMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI

# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.prebuilt import create_react_agent

# #from langchain_openai import ChatOpenAI
# #from langchain_mcp_adapters.server import StdioMcpToolServer # Correct import from the adapter library

# from dotenv import load_dotenv


# client = MultiServerMCPClient(
#     {
#         "execute_sql_query":{
#             "command": "python",
#             # Make sure to update to the full absolute path to your math_server.py file
#             "args": ["execute_query.py"],
#             "transport": "stdio",
#         },
#         "get_schema":{
#             "command": "python",
#             # Make sure to update to the full absolute path to your math_server.py file
#             "args": ["get_schema.py"],
#             "transport": "stdio",
#         },
#         "file_write":{
#             "command": "python",
#             # Make sure to update to the full absolute path to your math_server.py file
#             "args": ["file_write.py"],
#             "transport": "stdio",
#         },
#     }        
# )

# tools = await client.get_tools()
# # --- 1. Define the Graph State ---
# class AgentState(TypedDict):
#     """Represents the shared state of our workflow."""
#     messages: Annotated[List[BaseMessage], operator.add]
    
# # --- 2. Agent and Node Creation ---
# def create_agent_executor(llm, tools, system_prompt):
#     """Factory function to create a ReAct agent executor."""
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         ("human", "{input}"),
#         ("placeholder", "{agent_scratchpad}"),
#     ])
#     agent = create_react_agent(llm, tools, prompt)
#     return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# # --- 3. Main Application Logic ---
# def main():
#     load_dotenv()
    
#     # --- Correctly start the server and get tools using the adapter ---
#     # print("Initializing MCP Tool Server Adapter...")
#     # tool_server = StdioMcpToolServer(
#     #     "SalesTools",
#     #     command=["python", "-u", "server.py"] # The '-u' flag is important for unbuffered output
#     # )
#     # The adapter handles starting the server and gives us the tools
#     # tools = tool_server.get_tools()
#     # print(f"Tools loaded from server: {[tool.name for tool in tools]}")
    
#     try:
#         # Initialize the LLM
#         #llm = ChatOpenAI(model="gpt-4o")
#         #llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
#         llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
#         # Create Agent Executors with specific tools
#         translator_agent = create_agent_executor(
#             llm,
#             [tool for tool in tools if tool.name == "get_schema"],
#             "You are a master SQL writer. Your job is to translate a user's question into a precise SQL query based on the provided schema. Only output the final SQL query and nothing else."
#         )
#         executor_agent = create_agent_executor(
#             llm,
#             [tool for tool in tools if tool.name == "execute_sql_query"],
#             "You are a database operator. Your job is to execute the provided SQL query and return the raw results. Do not analyze them."
#         )
#         reporter_agent = create_agent_executor(
#             llm,
#             [tool for tool in tools if tool.name == "file_write"],
#             "You are a senior business analyst. Analyze the provided data and write a concise summary report in markdown. Then, save this report to a file named 'reports/sales_brief.md'."
#         )

#         # Define the graph nodes
#         def translator_node(state: AgentState):
#             result = translator_agent.invoke({"input": state['messages'][-1].content})
#             return {"messages": [HumanMessage(content=result["output"], name="Translator")]}

#         def executor_node(state: AgentState):
#             # The input for this agent is the output from the previous one
#             result = executor_agent.invoke({"input": state['messages'][-1].content})
#             return {"messages": [HumanMessage(content=result["output"], name="Executor")]}
        
#         def reporter_node(state: AgentState):
#             result = reporter_agent.invoke({"input": state['messages'][-1].content})
#             return {"messages": [HumanMessage(content=result["output"], name="Reporter")]}

#         # Define the graph
#         workflow = StateGraph(AgentState)
#         workflow.add_node("translator", translator_node)
#         workflow.add_node("executor", executor_node)
#         workflow.add_node("reporter", reporter_node)

#         # Define the workflow logic
#         workflow.set_entry_point("translator")
#         workflow.add_edge("translator", "executor")
#         workflow.add_edge("executor", "reporter")
#         workflow.add_edge("reporter", END)

#         # Compile the graph
#         app = workflow.compile()

#         # Run the workflow
#         print("\n" + "="*50)
#         print("--- RUNNING AI SALES ASSISTANT ---")
#         user_query = "What were our top 3 best-selling products by quantity in Q2 2025?"
#         initial_state = {"messages": [HumanMessage(content=user_query)]}
        
#         final_state = await app.invoke(initial_state)
        
#         print("\n\n✅ --- Workflow Complete ---")
#         print("Final Status:", final_state['messages'][-1].content)
#         print("Check the 'reports/sales_brief.md' file for the output.")

#     finally:
#         # The adapter handles shutting down the server process
#         print("\n--- Shutting down MCP server ---")
#         #  tool_server.stop()

# if __name__ == "__main__":
#     main()



# import operator
# from typing import TypedDict, Annotated, List
# import asyncio
# from langgraph.graph import StateGraph, END
# from langchain.agents import AgentExecutor
# from langchain_core.messages import HumanMessage, BaseMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_mistralai.chat_models import ChatMistralAI
# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.prebuilt import create_react_agent

# from dotenv import load_dotenv


# class AgentState(TypedDict):
#     messages: Annotated[List[BaseMessage], operator.add]


# def create_agent_executor(llm, tools, system_prompt):
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         ("human", "{input}"),
#         ("placeholder", "{agent_scratchpad}"),
#     ])

#     agent = create_react_agent(llm, tools, prompt=system_prompt)
#     #return agent
#     return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    


# async def main():
#     load_dotenv()

#     client = MultiServerMCPClient({
#         "execute_sql_query": {
#             "command": "python",
#             "args": ["execute_query.py"],
#             "transport": "stdio",
#         },
#         "get_schema": {
#             "command": "python",
#             "args": ["get_schema.py"],
#             "transport": "stdio",
#         },
#         "file_write": {
#             "command": "python",
#             "args": ["file_write.py"],
#             "transport": "stdio",
#         },
#     })

#     tools = await client.get_tools()

#     #llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
#     llm = ChatMistralAI(model="mistral-large-latest", temperature=0.7)
#     translator_agent = create_agent_executor(
#         llm,
#         [tool for tool in tools if tool.name == "get_schema"],
#         "You are a master SQL writer. Your job is to translate a user's question into a precise SQL query based on the provided schema. Only output the final SQL query and nothing else."
#     )
#     executor_agent = create_agent_executor(
#         llm,
#         [tool for tool in tools if tool.name == "execute_sql_query"],
#         "You are a database operator. Your job is to execute the given SQL query and return the results. Only output the final results and nothing else."
#     )
#     reporter_agent = create_agent_executor(
#         llm,
#         [tool for tool in tools if tool.name == "file_write"],
#         "You are a senior business analyst. Your job is to write the final report based on the provided data. Only output the final report and nothing else."
#     )


#     def translator_node(state: AgentState):
#         result = translator_agent.invoke({"input": state['messages'][-1].content})
#         return {"messages": [HumanMessage(content=result["output"], name="Translator")]}

#     def executor_node(state: AgentState):
#         result = executor_agent.invoke({"input": state['messages'][-1].content})
#         return {"messages": [HumanMessage(content=result["output"], name="Executor")]}

#     def reporter_node(state: AgentState):
#         result = reporter_agent.invoke({"input": state['messages'][-1].content})
#         return {"messages": [HumanMessage(content=result["output"], name="Reporter")]}

#     workflow = StateGraph(AgentState)
#     workflow.add_node("translator", translator_node)
#     workflow.add_node("executor", executor_node)
#     workflow.add_node("reporter", reporter_node)

#     workflow.set_entry_point("translator")
#     workflow.add_edge("translator", "executor")
#     workflow.add_edge("executor", "reporter")
#     workflow.add_edge("reporter", END)

#     app = workflow.compile()

#     print("\n" + "=" * 50)
#     print("--- RUNNING AI SALES ASSISTANT ---")
#     user_query = "What were our top 3 best-selling products by quantity in Q2 2025?"
#     initial_state = {"messages": [HumanMessage(content=user_query)]}

#     final_state = await app.invoke(initial_state)

#     print("\n\n✅ --- Workflow Complete ---")
#     print("Final Status:", final_state['messages'][-1].content)
#     print("Check the 'reports/sales_brief.md' file for the output.")


# if __name__ == "__main__":
#     asyncio.run(main())





















# import asyncio
# import operator
# from typing import Annotated
# from langgraph.graph import StateGraph, START, END, MessagesState
# from langchain_core.messages import HumanMessage
# from langchain_mistralai.chat_models import ChatMistralAI
# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.prebuilt import create_react_agent
# from dotenv import load_dotenv

# async def main():
#     load_dotenv()

#     # --- 1. Set up Tools & LLM ---
#     client = MultiServerMCPClient({
#         "execute_sql_query": {
#             "command": "python", "args": ["execute_query.py"], "transport": "stdio",
#         },
#         "get_schema": {
#             "command": "python", "args": ["get_schema.py"], "transport": "stdio",
#         },
#         "file_write": {
#             "command": "python", "args": ["file_write.py"], "transport": "stdio",
#         },
#     })
#     tools = await client.get_tools()
#     llm = ChatMistralAI(model="mistral-large-latest", temperature=0.1)

#     # --- 2. Create Agent Runnables ---
#     # These are the core agent "brains" that will be called by our node functions.
#     translator_runnable = create_react_agent(
#         llm,
#         tools=[tool for tool in tools if tool.name == "get_schema"],
#         messages_modifier="""You are an expert SQL writer. To answer the user's question, you MUST first use the 'get_schema' tool to see the database structure. Then, you must output a final, complete SQL query.""",
#     )
#     executor_runnable = create_react_agent(
#         llm,
#         tools=[tool for tool in tools if tool.name == "execute_sql_query"],
#         messages_modifier="""You are a database query executor. You will be given a SQL query. You MUST use the 'execute_sql_query' tool to run it. Return only the raw result from the tool.""",
#     )
#     reporter_runnable = create_react_agent(
#         llm,
#         tools=[tool for tool in tools if tool.name == "file_write"],
#         messages_modifier="""You are a business analyst. You will receive data and the original user question. Analyze the data, write a markdown report answering the question, then use the 'file_write' tool to save the report to 'reports/sales_brief.md'.""",
#     )

#     # --- 3. Define Node Functions ---
#     # These functions correctly manage the state between agent runs.
#     def translator_node(state: MessagesState):
#         """Generates the SQL query."""
#         # The input to the agent is a fresh conversation starting with the user's query.
#         result = translator_runnable.invoke(state)
#         # We only return the messages generated by this agent run.
#         return {"messages": result["messages"]}

#     def executor_node(state: MessagesState):
#         """Executes the SQL query."""
#         # Extract the SQL query from the previous step.
#         sql_query = state["messages"][-1].content
#         # The input to the agent is a fresh conversation starting with the SQL query.
#         result = executor_runnable.invoke({"messages": [HumanMessage(content=sql_query)]})
#         # We return only the new messages generated by this run.
#         return {"messages": result["messages"][1:]} # Slice to exclude the input HumanMessage we just made

#     def reporter_node(state: MessagesState):
#         """Analyzes results and writes the final report file."""
#         # Get the original question and the data from the database.
#         original_question = state["messages"][0].content
#         db_data = state["messages"][-1].content
#         # Format a clear task for the final agent.
#         task = f"Original Question: {original_question}\n\nData to analyze:\n{db_data}"
#         # The input is a fresh conversation starting with the formatted task.
#         result = reporter_runnable.invoke({"messages": [HumanMessage(content=task)]})
#         # Return only the new messages.
#         return {"messages": result["messages"][1:]}

#     # --- 4. Build the Graph ---
#     workflow = StateGraph(MessagesState)
#     workflow.add_node("translator", translator_node)
#     workflow.add_node("executor", executor_node)
#     workflow.add_node("reporter", reporter_node)

#     workflow.add_edge(START, "translator")
#     workflow.add_edge("translator", "executor")
#     workflow.add_edge("executor", "reporter")
#     workflow.add_edge("reporter", END)

#     app = workflow.compile()

#     # --- 5. Run the Graph ---
#     print("\n" + "=" * 50)
#     print("--- RUNNING AI SALES ASSISTANT (with Mistral AI) ---")
#     user_query = "What were our top 3 best-selling products by quantity in Q2 2025?"
#     initial_state = {"messages": [HumanMessage(content=user_query)]}

#     final_state = None
#     async for event in app.astream(initial_state, stream_mode="values"):
#         final_state = event
    
#     print("\n\n✅ --- Workflow Complete ---")
#     if final_state:
#         print("Final Status:", final_state['messages'][-1].content)
#         print("Check the 'reports/sales_brief.md' file for the output.")
#     else:
#         print("Workflow did not complete successfully.")

# if __name__ == "__main__":
#     asyncio.run(main())