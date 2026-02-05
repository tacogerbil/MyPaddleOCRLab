import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from .config import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCRExecutionError(Exception):
    """Custom exception for errors during OCR execution."""
    pass

class PaddleOCRProcessor:
    """
    Adapter class for PaddleOCR.
    Responsible for executing the OCR engine on a given image/PDF and returning the result.
    """

    def __init__(self):
        """Initialize the processor using the global config."""
        self.output_dir = config.OUTPUT_DIR
        self.is_docker = config.IS_DOCKER
        self.docker_image = config.PADDLE_DOCKER_IMAGE

    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Run PaddleOCR on the specified file.

        Args:
            file_path (Path): Absolute path to the input image or PDF.

        Returns:
            Dict[str, Any]: The raw structured output from the OCR engine.
        
        Raises:
            OCRExecutionError: If the external command fails or output is invalid.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        logger.info(f"Processing file: {file_path}")
        
        # Construct command
        command = self._build_command(file_path)
        logger.debug(f"Executing command: {' '.join(command)}")

        try:
            # Run the command
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                check=False # We check return code manually for better error messages
            )

            if result.returncode != 0:
                logger.error(f"OCR execution failed: {result.stderr}")
                raise OCRExecutionError(f"PaddleOCR command failed with code {result.returncode}: {result.stderr}")

            # For now, we assume standard PaddleOCR CLI structure.
            # Real execution output parsing depends heavily on the specific paddleocr version/flags.
            # This is a baseline implementation.
            logger.info("OCR execution successful.")
            return self._parse_output(result.stdout, file_path)

        except Exception as e:
            logger.exception("Unexpected error during OCR processing")
            raise OCRExecutionError(f"Failed to process {file_path}: {str(e)}") from e

    def _build_command(self, file_path: Path) -> List[str]:
        """Construct the subprocess command based on config."""
        if self.is_docker:
            # Docker execution logic (simplified)
            # This requires binding volumes, which complicates things. 
            # For this initial MCCC pass, we will sketch the Docker command but rely on local for the user's specific request.
            input_dir_mount = file_path.parent
            docker_cmd = [
                "docker", "run", "--rm",
                "-v", f"{input_dir_mount}:/data",
                self.docker_image,
                "paddleocr", "--image_dir", f"/data/{file_path.name}",
                "--use_angle_cls", "true",
                "--lang", "en",
                "--use_gpu", "false" # Default to false for compatibility, user can tune
            ]
            return docker_cmd
        else:
            # Local execution execution
            return [
                "paddleocr",
                "--image_dir", str(file_path),
                "--use_angle_cls", "true",
                "--lang", "en",
                "--use_gpu", "false" 
            ]

    def _parse_output(self, raw_output: str, source_file: Path) -> Dict[str, Any]:
        """
        Parse the raw CLI stdout from PaddleOCR.
        
        Note: PaddleOCR CLI prints results line by line. 
        We capture this and wrap it in a structured dictionary.
        """
        # Save raw output for debugging
        raw_output_path = self.output_dir / f"{source_file.stem}_raw.txt"
        with open(raw_output_path, "w", encoding="utf-8") as f:
            f.write(raw_output)
        
        logger.info(f"Raw output saved to {raw_output_path}")

        # Basic parsing (can be enhanced to parse actual JSON if --output_json is used)
        return {
            "source_file": str(source_file),
            "raw_text_content": raw_output
        }

if __name__ == "__main__":
    # Internal Manual Test
    # This block allows running this specific file to test it in isolation
    try:
        config.validate()
        processor = PaddleOCRProcessor()
        print("PaddleOCRProcessor initialized. Ready to process.")
        # To test: processor.process_document(Path("path/to/test.pdf"))
    except Exception as e:
        print(f"Initialization failed: {e}")
