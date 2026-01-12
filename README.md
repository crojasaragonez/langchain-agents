# SQL Agent with LangChain

An interactive SQL database agent built with LangChain that allows users to query databases using natural language.

## Installation

```bash
pip install -r requirements.txt
```

## Setup

1. Create a `.env` file in the project root with required configuration:
   ```
   DB_URI=sqlite:///Chinook.db
   MODEL_NAME=gpt-4o-mini  # Optional, defaults to gpt-4o-mini
   ```

2. Make sure your database file exists (Chinook.db is included in this repo)

**Note**: `DB_URI` is required. `MODEL_NAME` is optional and defaults to `gpt-4o-mini`.

## Usage

Run the interactive CLI:

```bash
python main.py
```

The CLI provides:
- Interactive questioning of your database
- Database inspection capabilities
- Natural language queries converted to SQL

### Available Commands

- **Ask questions**: Type any natural language question about your data
- **inspect**: Show database information and available tables
- **help**: Display help message
- **quit/exit/q**: Exit the application

### Example Questions

- "Which genre on average has the longest tracks?"
- "Show me the top 5 customers by total purchase amount"
- "How many albums does each artist have?"

## Architecture

- `sql_agent.py`: Contains the `SQLAgent` class with all business logic (requires `db_uri` and `model_name` parameters)
- `main.py`: Interactive CLI that imports and uses the SQLAgent class
- Environment variables are loaded in `main.py`, keeping business logic separate from application setup
