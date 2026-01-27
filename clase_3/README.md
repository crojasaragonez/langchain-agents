# Image Agent with LangChain

An interactive album cover generator agent built with LangChain that creates original and alternative album cover artwork using AI image generation (DALL-E).

## Features

- Generates two variations of album covers (original and alternative styles)
- Uses OpenAI's DALL-E 3 for high-quality image generation
- Saves images to the current working directory with descriptive filenames
- Interactive CLI interface

## Installation

```bash
pip install -r requirements.txt
```

## Setup

1. Create a `.env` file in the project root with required configuration:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   MODEL_NAME=gpt-4o-mini  # Optional, defaults to gpt-4o-mini
   IMAGE_MODEL=dall-e-3    # Optional, defaults to dall-e-3
   OUTPUT_DIR=.            # Optional, defaults to current directory
   ```

2. Make sure you have a valid OpenAI API key with access to DALL-E

**Note**: `OPENAI_API_KEY` is required. Other variables are optional with sensible defaults.

## Usage

Run the interactive CLI:

```bash
python main.py
```

The agent will prompt you for:
1. Artist name
2. Album name

It will then generate two album cover variations:
- **Original**: A professional, classic cover appropriate for the music genre
- **Alternative**: A creative, artistic, and visually striking variation

Both images are saved as PNG files in the output directory with timestamps.

## Example

```
Enter artist name: Pink Floyd
Enter album name: Dark Side of the Moon

Generating album covers for 'Dark Side of the Moon' by Pink Floyd...
```

This will create two files:
- `Pink_Floyd_Dark_Side_of_the_Moon_original_20260126_143022.png`
- `Pink_Floyd_Dark_Side_of_the_Moon_alternative_20260126_143045.png`

## Image Generation Details

- **Size**: 1024x1024 pixels
- **Format**: PNG
- **Model**: DALL-E 3 (configurable)
- **Quality**: Standard (cost-effective)

## Requirements

- Python 3.8+
- OpenAI API key with DALL-E access
- Internet connection for API calls
