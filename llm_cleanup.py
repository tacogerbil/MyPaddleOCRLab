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
        """Create the system/user prompt payload with strict OCR correction rules."""
        return text  # Just return the raw text - system prompt handles instructions

    def _send_request(self, prompt: str) -> Dict[str, Any]:
        """
        Send HTTP POST request to the LLM API.
        Supports both Ollama (/api/generate) and OpenWebUI/OpenAI (/v1/chat/completions) formats.
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add Bearer token if API key is present
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Determine payload format based on endpoint or presence of API key
        # OpenWebUI/OpenAI uses /chat/completions and "messages" array
        is_chat_api = "chat/completions" in self.api_url or self.api_key is not None
        
        if is_chat_api:
             # Chat Completion format (OpenAI/OpenWebUI)
             # OpenWebUI requires /api/chat/completions or /v1/chat/completions
             
             # Auto-fix URL if needed: if user gave base URL "http://host:port" and it's OpenWebUI (has key),
             # append the chat endpoint if safe to do so.
             if self.api_key and not self.api_url.endswith("/chat/completions"):
                 # Simplistic check to avoid double-appending. 
                 # If url ends in slash, remove it
                 base = self.api_url.rstrip("/")
                 # OpenWebUI usually listens on /api/chat/completions or /v1/chat/completions
                 # We'll default to the one in the screenshot if user just gave base URL
                 if not "api" in base and not "v1" in base:
                      self.api_url = f"{base}/api/chat/completions"
                      # logger.info(f"Auto-corrected URL to: {self.api_url}")

             payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": """You are a strict OCR text corrector. Your task is to output corrected plain text only.
Do not summarize, rephrase, paraphrase, reorganize, or add content.
Do not be creative.
The output must be identical to the input except for fixing clear OCR errors, spelling errors and broken line issues defined below.

Allowed corrections:

Character-level OCR fixes (context required):
- Fix common OCR confusions only when context makes the correction obvious:
  - Numeric context: O/o→0, l/I→1, S→5, B→8, Z→2
  - Word context: 0→O/o, 1→l, 5→S only if it forms a valid word
- Fix ligatures and OCR artifacts (ﬁ→fi, ﬂ→fl, stray symbols)
- If ambiguous, do not change

Spelling Errors:
- Correct obvious spelling mistakes

Dot-leader + page number line repair (critical):
- If a line appears to be a table of contents entry (title text)
- AND the following line contains only:
  - dots + a number (e.g. .2, ..ix, ....17)
  - or a number/roman numeral alone
- THEN merge them into a single line:
  - Preserve the dots
  - Add exactly one space before the page number
- Examples of lines to merge:
  - Title\n.2
  - Title\n..ix
  - Title .....\n17

Line-break fixes:
- Remove line breaks that split:
  - a word
  - a dot-leader/page-number pair
- Do not remove real paragraph breaks
- If unsure, keep the line break

Output rules:
- Output text only (no markup, no explanations)
- Preserve original wording, order, casing, and punctuation except for OCR fixes
- Make the minimum number of changes required
- Be conservative. Only fix errors that are clearly caused by OCR."""},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "temperature": 0  # Deterministic output
            }
        else:
            # Legacy Ollama /api/generate format
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1
                }
            }

        
        # DIAGNOSTIC: Log the payload being sent
        logger.info(f"DIAGNOSTIC LLM: Sending request to {self.api_url}")
        logger.info(f"DIAGNOSTIC LLM: Model: {self.model_name}")
        logger.info(f"DIAGNOSTIC LLM: Payload keys: {list(payload.keys())}")
        if 'messages' in payload:
            logger.info(f"DIAGNOSTIC LLM: System prompt length: {len(payload['messages'][0]['content'])} chars")
            logger.info(f"DIAGNOSTIC LLM: User content length: {len(payload['messages'][1]['content'])} chars")
            logger.info(f"DIAGNOSTIC LLM: First 100 chars of system prompt: {payload['messages'][0]['content'][:100]}")
        
        try:
            # logger.info(f"Sending request to LLM: {self.api_url}")
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=120)
            
            # DEBUG: unexpected 404 might mean wrong endpoint
            if response.status_code == 404:
                 logger.error(f"LLM 404 Error. URL used: {self.api_url}")
                 
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
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
