from multi_agent import MultiAgent
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def main():
    # Configuration
    db_path = Path(__file__).parent.parent / "clase_1" / "Chinook.db"
    db_uri = f"sqlite:///{db_path}"
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    image_model = os.getenv("IMAGE_MODEL", "dall-e-3")

    # Create output directory for album covers
    output_dir = Path(__file__).parent / "album_covers"
    output_dir.mkdir(exist_ok=True)

    # Initialize multi-agent system
    print("Initializing Multi-Agent System...")
    multi_agent = MultiAgent(
        db_uri=db_uri,
        model_name=model_name,
        image_model=image_model,
        output_dir=output_dir
    )

    # Get artist name from user
    print("\n" + "="*60)
    artist_name = input("Enter the name of an artist: ").strip()
    print("="*60)

    if not artist_name:
        print("❌ No artist name provided. Exiting.")
        return

    # Run the workflow
    try:
        final_state = multi_agent.run(artist_name)

        # Display results
        albums = final_state.get("albums", [])
        if albums:
            print("\n📋 Albums found:")
            for i, album in enumerate(albums, 1):
                print(f"  {i}. {album}")
        else:
            print("\n⚠️  No albums found for this artist.")
            print("💡 Tip: Try artists like 'AC/DC', 'Iron Maiden', 'Led Zeppelin', or 'Deep Purple'")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
