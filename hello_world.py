import os
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex,Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.web import SimpleWebPageReader

# Settings control global defaults
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(
    model="gemma:2b",
    request_timeout=360.0,
    # Manually set the context window to limit memory usage
    context_window=1000,
)

def main(webpage_url: str):
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls=[webpage_url])
    index = VectorStoreIndex.from_documents(documents=documents)
    print(index)
    query_engine = index.as_query_engine()
    response = query_engine.query('can you explain what is a longest substring means?')
    print(response)


if __name__ == "__main__":
    load_dotenv()
    url = 'https://leetcode.com/problems/longest-substring-without-repeating-characters/'
    main(url)
