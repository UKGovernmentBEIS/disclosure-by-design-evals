import sys
import os

# Set required hooks to empty to allow running without hooks
os.environ["INSPECT_REQUIRED_HOOKS"] = ""

import argparse
import yaml
from pathlib import Path
from datetime import datetime
from itertools import product
from inspect_ai import eval_set

# Add the script directory to Python path so we can import task_voice
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from task_voice import disclosure_voice_eval

def load_variants(config_path: str = None):
    """Load variant configurations."""
    if config_path is None:
        # Use config.yaml in the same directory as this script
        config_path = script_dir / "config.yaml"
    
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)['variants']

def run_disclosure_eval_set(epochs: int = 1, model: str = "openai/gpt-4o-audio-preview", log_dir: str = None):
    """Run evaluation across all variants."""
    variants = load_variants()
    
    # Create all combinations
    tasks = []
    for system_msg, tts_voice in product(
        variants['system_messages'], 
        variants['tts_voices']
    ):
        # Pass the config path relative to the script directory
        config_path = str(script_dir / "config.yaml")
        task = disclosure_voice_eval(
            system_msg=system_msg,
            tts_voice=tts_voice,
            config_path=config_path
        )
        tasks.append(task)

    # Use provided log_dir or create timestamped one
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/results_{timestamp}"
    
    # Run eval set
    logs = eval_set(
        tasks,
        model=model,
        epochs=epochs,
        log_dir=log_dir
    )
    
    print(f"Results saved to: {log_dir}")
    return logs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--model", default="openai/gpt-4o-audio-preview")
    parser.add_argument("--log-dir", help="Custom log directory")
    
    args = parser.parse_args()
    logs = run_disclosure_eval_set(
        epochs=args.epochs,
        model=args.model,
        log_dir=args.log_dir
    )
    print(f"Completed {len(logs)} evaluations with {args.epochs} epochs each")