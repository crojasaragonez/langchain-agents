import os
from dotenv import load_dotenv
from rag_agent import RagAgent

def main():
  load_dotenv()
  directory = os.getenv('DIRECTORY_TO_SCAN')
  model_name = os.getenv('MODEL_NAME')

  rag_agent = RagAgent(model_name=model_name, directory=directory)

  while True:
    user_input = input("Ask something to the Manual: \n").strip()
    if user_input.lower() == 'quit':
      break

    rag_agent.query(user_input)
    print("-" * 40)
    print("\n")

main()
