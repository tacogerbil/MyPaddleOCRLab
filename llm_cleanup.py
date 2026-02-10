import requests
import logging
import json
from enum import Enum
from typing import Optional, Dict, Any
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    OLLAMA = "ollama"
    OPEN_WEBUI = "open_webui"
    # Add others as needed

class LLMConnectionError(Exception):
    """Custom exception for errors connecting to the LLM API."""
    pass

class LLMCorrector:
    """
    Adapter class for Local LLM interaction.
    Responsible for sending dirty OCR text to the LLM and retrieving the cleaned version.
    """

    def __init__(self, llm_url: str = None, llm_api_key: str = None, llm_model: str = None):
        """Initialize using global config or provided overrides."""
        self.api_url = llm_url if llm_url else config.LLM_API_URL
        self.api_key = llm_api_key  # Optional API key for OpenWebUI/OpenAI
        self.model_name = llm_model if llm_model else config.LLM_MODEL_NAME
        
        # Auto-adjust URL mainly for OpenWebUI which often omits /chat/completions in the base URL
        # If user provides a base URL like http://192.168.1.101:3000, we might need to append /api/chat/completions
        # But we'll trust the input for now unless it's clearly missing the endpoint.
        
        # self.provider could be inferred from URL or set explicitly if api schemas differ wildly.
        # For now, we assume a standard Ollama/OpenAI-like "generate" or "chat" endpoint.

    def cleanup_text(self, raw_text: str) -> str:
        """
        Send raw OCR text to the LLM for correction.
        """
        if not raw_text.strip():
            return ""

        prompt = self._construct_prompt(raw_text)
        
        try:
            response = self._send_request(prompt)
            cleaned_text = self._extract_content(response)
            return cleaned_text
        except LLMConnectionError as e:
            logger.error(f"LLM Cleanup failed: {e}")
            return raw_text  # Fallback to raw text on failure
        except Exception as e:
            logger.exception("Unexpected error during LLM cleanup")
            return raw_text

    def _construct_prompt(self, text: str) -> str:
        """create the system/user prompt payload."""
        return f"""
Input: 
{text}

Task: Correct OCR errors (e.g., '1' vs 'l', '0' vs 'O'), fix broken line breaks, and normalize paragraphs. 
Do NOT summarize. Do NOT add commentary. Return ONLY the cleaned text.

Output:
"""

    def _send_request(self, prompt: str) -> Dict[str, Any]:
        """Execute the HTTP request."""
        # This payload structure assumes the configured endpoint supports "prompt" or "messages".
        # Adjust based on specific API (Ollama /generate vs OpenWebUI /chat/completions).
        
        # Defaulting to Ollama /api/generate style for simplicity based on "llama.cpp" mention
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        # If the user is using Open WebUI/OpenAI compatible endpoint, payload would need 'messages'.
        # We can detect this config flag later if needed.

        try:
            resp = requests.post(self.api_url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            raise LLMConnectionError(f"HTTP Request failed: {e}")

    def _extract_content(self, response_json: Dict[str, Any]) -> str:
        """Parse strict response from JSON."""
        # Support Ollama 'response' field
        if "response" in response_json:
            return response_json["response"]
        
        # Support OpenAI-style 'choices' field (for Open WebUI)
        if "choices" in response_json and len(response_json["choices"]) > 0:
            return response_json["choices"][0].get("message", {}).get("content", "")

        logger.error(f"Unknown JSON response format: {response_json.keys()}")
        raise LLMConnectionError("Could not parse LLM response.")

if __name__ == "__main__":
    # Test Block
    try:
        corrector = LLMCorrector()
        print("LLMCorrector initialized.")
        # test_text = "Th1s is a t3st."
        # print(corrector.cleanup_text(test_text))
    except Exception as e:
        print(f"Init failed: {e}")
