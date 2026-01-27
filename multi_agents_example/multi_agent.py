import sys
from pathlib import Path
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

# Add parent directories to path to import agents
sys.path.append(str(Path(__file__).parent.parent / "clase_1"))
sys.path.append(str(Path(__file__).parent.parent / "clase_3"))

from sql_agent import SQLAgent
from image_agent import ImageAgent


class AgentState(TypedDict):
    """State for the multi-agent system."""
    artist_name: str
    albums: Annotated[list[str], operator.add]
    current_step: str
    messages: Annotated[list, operator.add]


class MultiAgent:
    """Multi-agent system that coordinates SQLAgent and ImageAgent."""

    def __init__(self, db_uri, model_name, image_model="dall-e-3", output_dir=None):
        self.sql_agent = SQLAgent(
            db_uri=db_uri,
            model_name=model_name,
            top_k=50,
            history_path=Path(__file__).parent / "sql_history.json"
        )
        self.image_agent = ImageAgent(
            model_name=model_name,
            image_model=image_model,
            output_dir=output_dir or Path(__file__).parent / "album_covers"
        )

        # Ensure output directory exists
        self.image_agent.output_dir.mkdir(exist_ok=True)

        # Build the workflow graph
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()

    def _build_workflow(self):
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("get_albums", self._get_albums_node)
        workflow.add_node("generate_covers", self._generate_covers_node)

        # Define edges
        workflow.set_entry_point("get_albums")
        workflow.add_edge("get_albums", "generate_covers")
        workflow.add_edge("generate_covers", END)

        return workflow

    def _get_albums_node(self, state: AgentState) -> AgentState:
        """Node to get albums using SQLAgent."""
        artist_name = state["artist_name"]
        print(f"\n{'='*60}")
        print(f"Step 1: Querying database for albums by {artist_name}")
        print(f"{'='*60}\n")

        # Query the SQL database for albums
        question = f"Get all album titles by the artist '{artist_name}'. Return only the album titles, one per line."

        # Capture the response
        self.sql_agent.memory.add_user_message(question)
        input_messages = self.sql_agent._prepare_messages()

        albums = []
        last_messages = None
        for step in self.sql_agent.agent.stream({"messages": input_messages}, stream_mode="values"):
            step["messages"][-1].pretty_print()
            last_messages = step["messages"]

        # Save response to memory
        self.sql_agent._save_assistant_response(last_messages)

        # Extract album names from the last AI message
        if last_messages:
            for msg in reversed(last_messages):
                if msg.type == 'ai' and msg.content:
                    # Parse album names from response
                    albums = self._parse_albums_from_response(msg.content)
                    break

        return {
            "albums": albums,
            "current_step": "albums_retrieved",
            "messages": [f"Found {len(albums)} albums"]
        }

    def _parse_albums_from_response(self, response: str) -> list[str]:
        """Parse album names from the SQL agent response."""
        albums = []
        lines = response.split('\n')

        for line in lines:
            line = line.strip()
            # Skip empty lines and common non-album text
            if not line or line.lower().startswith(('here', 'the', 'i found', 'albums:', 'album titles')):
                continue
            # Remove common prefixes like "- ", "* ", numbers, etc.
            cleaned = line.lstrip('-*•► ').strip()
            # Remove numbering like "1. ", "2. ", etc.
            import re
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', cleaned)
            if cleaned and len(cleaned) > 1:
                albums.append(cleaned)

        return albums

    def _generate_covers_node(self, state: AgentState) -> AgentState:
        """Node to generate album covers using ImageAgent."""
        artist_name = state["artist_name"]
        albums = state.get("albums", [])

        print(f"\n{'='*60}")
        print(f"Step 2: Generating album covers")
        print(f"{'='*60}\n")

        if not albums:
            print("⚠️  No albums found!")
            return {
                "current_step": "completed",
                "messages": ["No albums to generate covers for"]
            }

        print(f"Found {len(albums)} albums. Generating covers...\n")

        for i, album in enumerate(albums, 1):
            print(f"\n[{i}/{len(albums)}] Generating cover for: {album}")
            print("-" * 60)
            try:
                self.image_agent.generate_cover(
                    artist=artist_name,
                    album=album,
                    style="alternative"
                )
                print(f"✓ Cover generated successfully")
            except Exception as e:
                print(f"✗ Error generating cover: {e}")

        return {
            "current_step": "completed",
            "messages": [f"Generated covers for {len(albums)} albums"]
        }

    def run(self, artist_name: str):
        """Run the multi-agent workflow."""
        print(f"\n{'*'*60}")
        print(f"Multi-Agent Workflow Started")
        print(f"Artist: {artist_name}")
        print(f"{'*'*60}\n")

        initial_state = {
            "artist_name": artist_name,
            "albums": [],
            "current_step": "start",
            "messages": []
        }

        # Run the workflow
        final_state = self.app.invoke(initial_state)

        print(f"\n{'*'*60}")
        print(f"Multi-Agent Workflow Completed")
        print(f"Albums processed: {len(final_state.get('albums', []))}")
        print(f"Output directory: {self.image_agent.output_dir}")
        print(f"{'*'*60}\n")

        return final_state
