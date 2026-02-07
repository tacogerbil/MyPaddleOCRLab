import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Central configuration adapter for the PaddleOCR workflow.
    Loads settings from environment variables and provides typed properties.
    """

    # --- Paths ---
    @property
    def INPUT_DIR(self) -> Path:
        """Directory to watch for new PDF/image files."""
        return Path(os.getenv("INPUT_DIR", "./data"))

    @property
    def OUTPUT_DIR(self) -> Path:
        """Directory to store raw OCR output (JSON/Text)."""
        return Path(os.getenv("OUTPUT_DIR", "./output"))

    @property
    def CLEANED_DIR(self) -> Path:
        """Directory to store LLM-cleaned text pages."""
        return Path(os.getenv("CLEANED_DIR", "./cleaned"))

    # --- LLM Settings ---
    @property
    def LLM_API_URL(self) -> str:
        """URL for the Local LLM API (e.g., Ollama or LocalAI)."""
        return os.getenv("LLM_API_URL", "http://localhost:11434/api/generate")

    @property
    def LLM_MODEL_NAME(self) -> str:
        """Name of the model to use (specific to the API provider)."""
        return os.getenv("LLM_MODEL_NAME", "llama3.1")

    # --- PaddleOCR Settings ---
    @property
    def IS_DOCKER(self) -> bool:
        """
        True: Use 'docker run' to execute PaddleOCR.
        False: Use local 'paddleocr' command.
        """
        return os.getenv("IS_DOCKER", "false").lower() == "true"
    
    @property
    def PADDLE_DOCKER_IMAGE(self) -> str:
        """Name of the PaddleOCR docker image to run."""
        return os.getenv("PADDLE_DOCKER_IMAGE", "paddlepaddle/paddleocr:latest")

    @property
    def USE_GPU(self) -> bool:
        """Enable GPU acceleration for PaddleOCR."""
        return os.getenv("USE_GPU", "true").lower() == "true"

    @property
    def USE_ANGLE_CLS(self) -> bool:
        """Enable angle classifier for document rotation."""
        return os.getenv("USE_ANGLE_CLS", "true").lower() == "true"

    @property
    def LANG(self) -> str:
        """Language code for OCR (default 'en')."""
        return os.getenv("LANG", "en")

    @property
    def SHOW_LOG(self) -> bool:
        """Show internal PaddleOCR logs."""
        return os.getenv("SHOW_LOG", "false").lower() == "true"

    def validate(self):
        """Ensure all critical paths exist."""
        self.INPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CLEANED_DIR.mkdir(parents=True, exist_ok=True)

# Global instance
config = Config()
