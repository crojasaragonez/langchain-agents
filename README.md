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

