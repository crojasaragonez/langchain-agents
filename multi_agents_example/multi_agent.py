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
from email_agent import EmailAgent


class AgentState(TypedDict):
    """State for the multi-agent system."""
    artist_name: str
    albums: Annotated[list[str], operator.add]
    current_step: str
    messages: Annotated[list, operator.add]
    generated_images: Annotated[list[Path], operator.add]
    email_approved: bool
    recipient_email: str


class MultiAgent:
    """Multi-agent system that coordinates SQLAgent, ImageAgent, and EmailAgent."""

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

        # Initialize EmailAgent (will raise error if env vars not set)
        try:
            self.email_agent = EmailAgent()
        except ValueError as e:
            print(f"⚠️  Warning: {e}")
            print("Email functionality will be disabled.")
            self.email_agent = None

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
        workflow.add_node("send_email", self._send_email_node)

        # Define edges
        workflow.set_entry_point("get_albums")
        workflow.add_edge("get_albums", "generate_covers")

        # Add conditional edge based on email agent availability
        if self.email_agent:
            workflow.add_edge("generate_covers", "send_email")
            workflow.add_edge("send_email", END)
        else:
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
                "current_step": "covers_generated",
                "messages": ["No albums to generate covers for"],
                "generated_images": []
            }

        print(f"Found {len(albums)} albums. Generating covers...\n")

        generated_images = []
        for i, album in enumerate(albums, 1):
            print(f"\n[{i}/{len(albums)}] Generating cover for: {album}")
            print("-" * 60)
            try:
                image_path = self.image_agent.generate_cover(
                    artist=artist_name,
                    album=album,
                    style="alternative"
                )
                generated_images.append(Path(image_path))
                print(f"✓ Cover generated successfully")
            except Exception as e:
                print(f"✗ Error generating cover: {e}")

        return {
            "current_step": "covers_generated",
            "messages": [f"Generated covers for {len(albums)} albums"],
            "generated_images": generated_images
        }

    def _send_email_node(self, state: AgentState) -> AgentState:
        """Node to send email with generated images (with human-in-the-loop approval)."""
        artist_name = state["artist_name"]
        albums = state.get("albums", [])
        generated_images = state.get("generated_images", [])
        recipient_email = state.get("recipient_email", "")

        print(f"\n{'='*60}")
        print(f"Step 3: Sending email with album covers")
        print(f"{'='*60}\n")

        if not generated_images:
            print("⚠️  No images to send!")
            return {
                "current_step": "completed",
                "messages": ["No images to send via email"]
            }

        # Prepare email content
        subject = f"Album Covers for {artist_name}"
        body = f"""Hello!

I've generated album covers for {artist_name}.

This email contains {len(generated_images)} album cover(s) for the following albums:
{chr(10).join(f"  • {album}" for album in albums)}

Best regards,
Multi-Agent System
"""

        # Preview email
        print("📧 Email Preview:")
        print("-" * 60)
        preview = self.email_agent.preview_email(
            to_email=recipient_email,
            body=body
        )

        print(f"From: {preview['from']}")
        print(f"To: {preview['to']}")
        print(f"Subject: {preview['subject']}")
        print(f"\nBody:\n{preview['body']}")
        print("-" * 60)

        # Human-in-the-loop: Ask for approval
        print("\n🤔 Do you want to send this email?")
        approval = input("Enter 'yes' to send or 'no' to cancel: ").strip().lower()

        if approval in ['yes', 'y']:
            print("\n✓ Email approved! Sending...")
            success = self.email_agent.send_email(
                to_email=recipient_email,
                body=body,
            )

            if success:
                return {
                    "current_step": "completed",
                    "messages": [f"Email sent successfully to {recipient_email}"],
                    "email_approved": True
                }
            else:
                return {
                    "current_step": "completed",
                    "messages": ["Failed to send email"],
                    "email_approved": False
                }
        else:
            print("\n✗ Email cancelled by user.")
            return {
                "current_step": "completed",
                "messages": ["Email sending cancelled by user"],
                "email_approved": False
            }

    def run(self, artist_name: str, recipient_email: str = ""):
        """Run the multi-agent workflow.

        Args:
            artist_name: Name of the artist to search for
            recipient_email: Email address to send the album covers to (optional)
        """
        print(f"\n{'*'*60}")
        print(f"Multi-Agent Workflow Started")
        print(f"Artist: {artist_name}")
        if recipient_email and self.email_agent:
            print(f"Recipient: {recipient_email}")
        print(f"{'*'*60}\n")

        initial_state = {
            "artist_name": artist_name,
            "albums": [],
            "current_step": "start",
            "messages": [],
            "generated_images": [],
            "email_approved": False,
            "recipient_email": recipient_email
        }

        # Run the workflow
        final_state = self.app.invoke(initial_state)

        print(f"\n{'*'*60}")
        print(f"Multi-Agent Workflow Completed")
        print(f"Albums processed: {len(final_state.get('albums', []))}")
        print(f"Images generated: {len(final_state.get('generated_images', []))}")
        print(f"Output directory: {self.image_agent.output_dir}")
        if self.email_agent and recipient_email:
            email_status = "Sent" if final_state.get('email_approved', False) else "Not sent"
            print(f"Email status: {email_status}")
        print(f"{'*'*60}\n")

        return final_state
