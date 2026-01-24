from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain.agents import create_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
# Retrieval-Augmented Generation
class RagAgent:
  def __init__(self, model_name, directory):
    self.model = init_chat_model(model_name)
    self.directory = directory
    self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    self.vector_store = InMemoryVectorStore(self.embeddings)
    self.memory = InMemoryChatMessageHistory()

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
      """Retrieve relevant context from PDF documents using similarity search

      This tool search the vector store containing the loaded PDF documents to
      find the most relevant information based on the query.

      Use this tool to find answers about the Nissan Frontier vehicle.

      Args:
        query: The search query to find relevant documents.

      Returns:
        A tuple containing the serialized string and the retrieved documents.
      """
      retrieved_docs = self.vector_store.similarity_search(query, k=2)
      serialized = "\n\n".join(
          (f"Source: {doc.metadata}\nContent: {doc.page_content}")
          for doc in retrieved_docs
      )
      return serialized, retrieved_docs

    tools = [retrieve_context]
    system_prompt = self._get_system_prompt()
    self.agent = create_agent(self.model, tools, system_prompt=system_prompt)
    self.load_documents()

  def load_documents(self):
    loader = DirectoryLoader(self.directory, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=200,
      add_start_index=True,
    )
    all_splits = text_splitter.split_documents(documents)
    self.vector_store.add_documents(all_splits)

  def _get_system_prompt(self):
    return f"""
You are an assistant specialized in answering questions about the Nissan Frontier vehicle
You have access to a tool that retrieves context from PDF Documents stored in the filesystem

IMPORTANT RULES:
1. you MUST ONLY answer questions that are directly related to the information in the documents before answering any question.
2. ALWAYS use the retrieve_context tool to search for relevant information in the documents before answering any question.
"""

  def query(self, question):
    self.memory.add_user_message(question)
    input_messages = self._prepare_messages()
    last_messages = self._stream_agent_response(input_messages)
    self._save_assistant_response(last_messages)

  def _save_assistant_response(self, messages):
    if not messages:
      return
    for msg in reversed(messages):
      if msg.type == "ai":
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        self.memory.add_ai_message(content)

  def _stream_agent_response(self, input_messages):
    last_messages = None
    for step in self.agent.stream({"messages": input_messages}, stream_mode="values"):
      step["messages"][-1].pretty_print()
      last_messages = step["messages"]
    return last_messages

  def _prepare_messages(self):
    role_map = {
      "human": "user",
      "ai": "assistant",
      "system": "system",
      "tool": "tool"
    }
    return [
      {"role": role_map.get(m.type, "user"), "content": m.content or ""}
      for m in self.memory.messages
    ]
