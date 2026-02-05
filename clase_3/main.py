import os
from dotenv import load_dotenv
from image_agent import ImageAgent

def main():
  load_dotenv()
  model_name = os.getenv('MODEL_NAME', 'gpt-4o-mini')
  image_model = os.getenv('IMAGE_MODEL', 'dall-e-3')
  output_dir = os.getenv('OUTPUT_DIR', '.')

  image_agent = ImageAgent(model_name=model_name, image_model=image_model, output_dir=output_dir)

  print("Welcome to the Album Cover Generator!")
  print("This agent will create an album cover for you.")
  print("Type 'quit' to exit.\n")

  while True:
    artist = input("Enter artist name (or 'quit' to exit): ").strip()
    if artist.lower() == 'quit':
      break

    if not artist:
      print("Artist name cannot be empty. Please try again.\n")
      continue

    album = input("Enter album name: ").strip()
    if album.lower() == 'quit':
      break

    if not album:
      print("Album name cannot be empty. Please try again.\n")
      continue

    style = input("Enter style (original/alternative) [default: alternative]: ").strip().lower()
    if style == 'quit':
      break

    if style not in ['original', 'alternative', '']:
      print("Invalid style. Using 'alternative' as default.\n")
      style = 'alternative'

    if not style:
      style = 'alternative'

    print(f"\nGenerating {style} album cover for '{album}' by {artist}...\n")
    image_agent.generate_cover(artist, album, style)
    print("-" * 40)
    print("\n")

if __name__ == "__main__":
  main()
