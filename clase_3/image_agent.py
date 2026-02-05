from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.tools import tool
import requests
import base64
import os
from pathlib import Path
from datetime import datetime

class ImageAgent:
  def __init__(self, model_name, image_model="dall-e-3", output_dir=None):
    self.model = init_chat_model(model_name)
    self.image_model = image_model
    self.output_dir = Path(output_dir) if output_dir else Path.cwd()
    self.memory = InMemoryChatMessageHistory()

    @tool
    def generate_album_cover(artist: str, album: str, style: str = "original") -> str:
      """Generate an album cover image for a given artist and album.

      This tool creates album cover artwork using AI image generation.
      The style parameter can be 'original' or 'alternative' to create
      different variations of the album cover.

      Args:
        artist: The name of the artist or band
        album: The name of the album
        style: The style of the cover ('original' or 'alternative')

      Returns:
        A message indicating where the image was saved
      """
      # Create a prompt based on the style
      if style.lower() == "alternative":
        prompt = f"""Design a creative and alternative interpretation of an album cover
for '{album}' by {artist}. Make it highly artistic, visually striking, and unconventional.
Use bold colors, abstract elements, surreal imagery, or experimental composition.
The design should be modern, eye-catching, and push creative boundaries.
Do NOT recreate any existing album cover - create something entirely new and imaginative."""
      else:
        prompt = f"""Design a new original-style album cover for '{album}' by {artist}.
The design should be professional, elegant, and timeless with a classic aesthetic.
Use clean composition, sophisticated color palette, and traditional design principles.
Make it appropriate for the music genre while being fresh and contemporary.
Do NOT recreate any existing album cover - create an original new design."""

      try:
        # Use OpenAI's DALL-E for image generation
        from openai import OpenAI
        client = OpenAI()

        image_response = client.images.generate(
          model=self.image_model,
          prompt=prompt,
          size="auto",
          quality="low",
          n=1,
        )


        image_base64 = image_response.data[0].b64_json

        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_artist = "".join(c for c in artist if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_album = "".join(c for c in album if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"{safe_artist}_{safe_album}_{style}_{timestamp}.png"
        filepath = self.output_dir / filename

        # Download and save the image
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(image_base64))

        return f"Album cover saved successfully to: {filepath}"

      except Exception as e:
        return f"Error generating image: {str(e)}"

    tools = [generate_album_cover]
    system_prompt = self._get_system_prompt()
    self.agent = create_agent(self.model, tools, system_prompt=system_prompt)

  def _get_system_prompt(self):
    return """
You are an assistant specialized in generating album cover artwork.

When a user provides an artist name, album name, and style preference, you should:
1. Generate ONE album cover using the generate_album_cover tool with the specified style

The image will be saved to the current working directory.

IMPORTANT RULES:
1. Generate exactly ONE cover with the style specified by the user
2. Use the exact artist and album names provided by the user
3. Report back to the user where the image was saved
"""

  def generate_cover(self, artist, album, style="alternative"):
    """Generate one album cover with the specified style

    Returns:
        str: Path to the generated image file
    """
    question = f"Generate a {style} album cover for the album '{album}' by {artist}"
    self.memory.add_user_message(question)
    input_messages = self._prepare_messages()
    last_messages = self._stream_agent_response(input_messages)
    self._save_assistant_response(last_messages)

    # Extract file path from the response
    if last_messages:
      for msg in reversed(last_messages):
        if msg.type == "ai" and msg.content:
          content = msg.content if isinstance(msg.content, str) else str(msg.content)
          # Look for the file path in the response
          if "Album cover saved successfully to:" in content:
            import re
            match = re.search(r'Album cover saved successfully to: (.+?)(?:\n|$)', content)
            if match:
              return match.group(1).strip()

    # If we can't extract the path, construct it based on the pattern
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_artist = "".join(c for c in artist if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_album = "".join(c for c in album if c.isalnum() or c in (' ', '-', '_')).strip()
    filename = f"{safe_artist}_{safe_album}_{style}_{timestamp}.png"
    return str(self.output_dir / filename)

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
