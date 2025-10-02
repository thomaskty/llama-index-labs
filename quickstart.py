import asyncio
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context

# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b

# Create an agent workflow with our calculator tool
agent = FunctionAgent(
    tools=[multiply],
    llm=Ollama(
        model="llama3.2:latest",
        request_timeout=100,
        # Manually set the context window to limit memory usage
        context_window=200,
    ),
    system_prompt="You are a helpful assistant",
)

async def main():
    ctx = Context(agent)
    await agent.run("What is 100 - 70?")
    await agent.run("My name is Logan", ctx=ctx)
    await agent.run('I like ice cream and my friend merin likes cakes', ctx=ctx)
    response = await agent.run("What is my name? and what merin likes?", ctx=ctx)
    print(str(response))

# Run the agent
if __name__ == "__main__":
    asyncio.run(main())


