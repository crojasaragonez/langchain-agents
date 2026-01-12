from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
import os


class SQLAgent:
    """
    A SQL Agent class that provides an object-oriented interface for interacting
    with SQL databases using LangChain agents.

    This class encapsulates the setup and usage of a SQL database agent,
    making it easier to reuse and maintain for software engineering applications.
    """

    def __init__(self, db_uri, model_name, top_k=5):
        """
        Initialize the SQL Agent.

        Args:
            db_uri (str): Database URI for connecting to the SQL database
            model_name (str): Name of the LLM model to use
            top_k (int, optional): Maximum number of results to return in queries (default: 5)
        """
        # Setup configuration
        self.db_uri = db_uri
        self.model_name = model_name
        self.top_k = top_k

        # Initialize components
        self.model = None
        self.db = None
        self.toolkit = None
        self.tools = None
        self.agent = None

        # Setup the agent
        self._setup_agent()

    def _setup_agent(self):
        """Private method to setup the agent and all its components."""
        self._setup_model()
        self._setup_database()
        self._create_toolkit()
        self._create_agent()

    def _setup_model(self):
        """Setup the language model."""
        self.model = init_chat_model(self.model_name)

    def _setup_database(self):
        """Setup the database connection."""
        if not self.db_uri:
            raise ValueError("Database URI not provided and not found in environment variables")
        self.db = SQLDatabase.from_uri(self.db_uri)

    def _create_toolkit(self):
        """Create the SQL database toolkit."""
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.model)
        self.tools = self.toolkit.get_tools()

    def _create_agent(self):
        """Create the agent with the system prompt."""
        system_prompt = self._get_system_prompt()
        self.agent = create_agent(
            self.model,
            self.tools,
            system_prompt=system_prompt,
        )

    def _get_system_prompt(self):
        """Generate the system prompt for the agent."""
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

    def inspect_database(self):
        """Inspect and print database information including dialect, tables, and tools."""
        print(f"Dialect: {self.db.dialect}")
        print(f"Available tables: {self.db.get_usable_table_names()}")
        print(f'Sample output: {self.db.run("SELECT * FROM Artist LIMIT 5;")}')

        print("\nAvailable tools:")
        for tool in self.tools:
            print(f"{tool.name}: {tool.description}\n")

    def query(self, question, stream_mode="values"):
        """
        Run a query using the agent.

        Args:
            question (str): The question to ask the agent
            stream_mode (str): Streaming mode for the agent response (default: "values")
        """
        for step in self.agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode=stream_mode,
        ):
            step["messages"][-1].pretty_print()


