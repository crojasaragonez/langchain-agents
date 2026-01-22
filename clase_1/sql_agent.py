from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langchain_core.chat_history import InMemoryChatMessageHistory

class SQLAgent:

  def __init__(self, db_uri, model_name, top_k=5):
    self.db_uri     = db_uri
    self.model_name = model_name
    self.top_k      = top_k

    self.model = None
    self.db = None
    self.toolkit = None
    self.tools = None
    self.agent = None
    self.memory = InMemoryChatMessageHistory()
    self._setup()

  def _setup(self):
    self.model = init_chat_model(self.model_name)
    self.db = SQLDatabase.from_uri(self.db_uri)
    self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.model)
    self.tools = self.toolkit.get_tools()
    system_prompt = self._get_system_prompt()
    self.agent = create_agent(self.model, self.tools, system_prompt=system_prompt)

  def _get_system_prompt(self):
    return f"""
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {self.db.dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {self.top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
"""
  def query(self, question):
    self.memory.add_user_message(question)
    input_messages = self._prepare_messages()
    last_messages = self._stream_agent_response(input_messages)
    self._save_assistant_response(last_messages)

  def _save_assistant_response(self, messages):
    for msg in reversed(messages):
      if msg.type == 'ai':
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
    # { role: user, content: "cuantas facturas hay?" }
    # { role: ai, content: "hay 150 facturas"}
    return [
      { "role": role_map.get(m.type, "user"), "content": m.content or "" }
      for m in self.memory.messages
    ]

