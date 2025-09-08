import logging
from pathlib import Path

def loggerSetup():
    """
        Set up the logger for logs.
        
        This function configures the logging settings for the module,
        including the log file name, logging level, format, and date format.
    """
    logs_path = Path("logs")
    logs_path.mkdir(parents=True, exist_ok=True)  # Create folder if not exists

    logging.basicConfig(
        filename=logs_path/"parser_logs.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True
    )

logger = logging.getLogger(__name__)