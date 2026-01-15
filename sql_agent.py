from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent

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
    for step in self.agent.stream(
      {"messages": [{"role": "user", "content": question}]},
      stream_mode="values"
    ):
      step["messages"][-1].pretty_print()
