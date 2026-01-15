import os
from dotenv import load_dotenv
from sql_agent import SQLAgent

def main():
  load_dotenv()
  db_uri = os.getenv('DB_URI')
  model_name = os.getenv('MODEL_NAME')

  sql_agent = SQLAgent(db_uri=db_uri, model_name=model_name)

  while True:
    user_input = input("Ask something to the DB: \n").strip()
    if user_input.lower() == 'quit':
      break

    sql_agent.query(user_input)
    print("-" * 40)
    print("\n")

main()
