import sys
from pathlib import Path

# Add current directory to path so src module is discoverable
sys.path.insert(0, str(Path(__file__).parent))

from src.langgraphagentic.main import load_langgraph_agenticai_app


if __name__ == "__main__":
    load_langgraph_agenticai_app()
