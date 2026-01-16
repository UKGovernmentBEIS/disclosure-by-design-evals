# run_text_baseline.py
import sys
import os

# Set required hooks to empty to allow running without hooks
os.environ["INSPECT_REQUIRED_HOOKS"] = ""

import argparse
import yaml
from pathlib import Path
from datetime import datetime

from inspect_ai import eval_set

# Add the script directory to Python path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from disclosure_text_eval import disclosure_text_eval

def load_variants(config_path: str = None):
    """Load variant configurations."""
    if config_path is None:
        config_path = script_dir / "config.yaml"
    
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)['variants']

def run_text_baseline(epochs: int = 1, model: str = "openai/gpt-4o", log_dir: str = None):
    """Run text-only baseline evaluation across system message variants."""
    variants = load_variants()
    
    # Create tasks for each system message variant
    tasks = []
    for system_msg in variants['system_messages']:
        config_path = str(script_dir / "config.yaml")
        task = disclosure_text_eval(
            system_msg=system_msg,
            config_path=config_path
        )
        tasks.append(task)

    # Use provided log_dir or create timestamped one
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/text_baseline_{timestamp}"
    
    # Run eval set
    logs = eval_set(
        tasks,
        model=model,
        epochs=epochs,
        log_dir=log_dir
    )
    
    print(f"Text baseline results saved to: {log_dir}")
    return logs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--model", default="openai/gpt-4o")
    parser.add_argument("--log-dir", help="Custom log directory")
    
    args = parser.parse_args()
    logs = run_text_baseline(
        epochs=args.epochs,
        model=args.model,
        log_dir=args.log_dir
    )
    print(f"Completed {len(logs)} text baseline evaluations with {args.epochs} epochs each")