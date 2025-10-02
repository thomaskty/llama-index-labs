from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import asyncio
import os

# Settings control global defaults
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(
    model="llama3.2:latest",
    request_timeout=100.0,
    # Manually set the context window to limit memory usage
    context_window=100,
)

# Create a RAG tool using LlamaIndex
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(
    documents,
    # we can optionally override the embed_model here
    # embed_model=Settings.embed_model,
)
query_engine = index.as_query_engine(
    # we can optionally override the llm here
    # llm=Settings.llm,
)


def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about a personal essay written by Paul Graham."""
    response = await query_engine.aquery(query)
    return str(response)


# Create an enhanced workflow with both tools
agent = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=[multiply, search_documents],
    llm=Settings.llm,
    system_prompt="""You are a helpful assistant that can perform calculations
    and search through documents to answer questions.""",
)


# Now we can ask questions about the documents or do calculations
async def main():
    response = await agent.run("What did the author do in college? Also, what's 7 * 8?")
    print(response)


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())


# # Save the index
# index.storage_context.persist("storage")
#
# # Later, load the index
# from llama_index.core import StorageContext, load_index_from_storage
#
# storage_context = StorageContext.from_defaults(persist_dir="storage")
# index = load_index_from_storage(
#     storage_context,
#     # we can optionally override the embed_model here
#     # it's important to use the same embed_model as the one used to build the index
#     # embed_model=Settings.embed_model,
# )
# query_engine = index.as_query_engine(
#     # we can optionally override the llm here
#     # llm=Settings.llm,
# )
