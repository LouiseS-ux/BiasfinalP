"""
Main orchestrator for the applications pipeline.
Runs all five stages in sequence. Each stage checks whether its
output already exists and skips if so.
"""

# Makes type hints work consistently across Python versions
from __future__ import annotations

import logging

# Goes into each stage file, imports its main function, and gives it clear local name
from src.stage1_probe_bank import main as run_stage1
from src.stage2_query_llm import main as run_stage2
from src.stage3_hearts import main as run_stage3_hearts
from src.stage3b_divergence import main as run_stage3b_divergence
from src.stage4_shap import main as run_stage4

# creates info logger for this file and adds this files name
logger = logging.getLogger(__name__)


# Defines the orchestrator function
def main() -> None:
    """Entry point — runs all pipeline stages in sequence."""

    # Logs a message to the console
    logger.info("Starting pipeline run.")

    # Logs a message to the console
    logger.info("Stage 1: Building probe bank...")

    # calls and runs Stage 1's main() function
    run_stage1()

    logger.info("Stage 2: Querying LLMs...")
    run_stage2()

    logger.info("Stage 3: Running classifier...")
    run_stage3_hearts()

    logger.info("Stage 3b: Running candidate choice divergence scoring...")
    run_stage3b_divergence()

    logger.info("Stage 4: Running SHAP explainability...")
    run_stage4()

    logger.info(
        "Pipeline complete. Launch dashboard with: streamlit run src/dashboard/app.py"
    )


if __name__ == "__main__":  # python sets file name to main and runs file contents
    logging.basicConfig(
        level=logging.INFO
    )  # switches the logger on and sets it to show msgs
    main()  # executes the function defined above
