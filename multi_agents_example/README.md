# Clase 4: Multi-Agent System with LangGraph

This example demonstrates a simple multi-agent system using LangGraph that orchestrates two agents:
- **SQLAgent**: Queries a database to find albums by an artist
- **ImageAgent**: Generates album cover images using AI

## Workflow

The multi-agent system follows this simple workflow:

```
1. User Input → 2. SQLAgent → 3. ImageAgent → 4. Done
   (Artist)      (Get Albums)   (Generate Covers)
```

### Steps:
1. **User provides artist name** - The system asks for an artist's name
2. **SQLAgent queries database** - Retrieves all album titles for that artist from the Chinook database
3. **ImageAgent generates covers** - Creates album cover artwork for each album found

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```

3. Add your OpenAI API key to the `.env` file:
```
OPENAI_API_KEY=your_key_here
```

## Usage

Run the multi-agent system:
```bash
python main.py
```

Then enter an artist name when prompted. The system will:
1. Search for albums by that artist in the database
2. Generate album cover images for each album
3. Save the images to the `album_covers/` directory

## Example Artists in Database

The Chinook database includes albums from various artists such as:
- AC/DC
- Iron Maiden
- Led Zeppelin
- Deep Purple
- Metallica
- And many more!

## Architecture

The system uses **LangGraph** to orchestrate the workflow:

- **State**: Maintains artist name, album list, and current step
- **Nodes**: 
  - `get_albums`: Uses SQLAgent to query database
  - `generate_covers`: Uses ImageAgent to create artwork
- **Edges**: Defines the flow from one step to the next

## Output

Generated album covers are saved to the `album_covers/` directory with filenames like:
```
ArtistName_AlbumName_alternative_20260126_123456.png
```

## Key Features

- ✅ Simple linear workflow (easy to understand)
- ✅ Automatic album parsing from SQL responses
- ✅ Progress tracking with visual feedback
- ✅ Error handling for each step
- ✅ Clean separation of concerns between agents
