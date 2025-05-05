import os
import requests
import logging
import base64
from datetime import datetime
from pathlib import Path 
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from stepfun_mcp.const import (
    ENV_STEPFUN_API_KEY,
    ENV_STEPFUN_API_HOST,
)
from stepfun_mcp.exceptions import StepFunAPIError, StepFunAuthError, StepFunRequestError

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# --- Request/Response Models (Updated based on curl examples) ---

# --- Text/Vision Models ---
class TextMessage(BaseModel):
    role: str # "system", "user", "assistant"
    content: str

class VisionContentPartText(BaseModel):
    type: str = "text"
    text: str

class VisionContentPartImage(BaseModel):
    type: str = "image_url"
    image_url: Dict[str, str] # Expects {"url": "data:image/...;base64,..."}

class VisionMessage(BaseModel):
    role: str # "system", "user", "assistant"
    # Content can be simple string (for text-only) or list (for vision)
    content: Union[str, List[Union[VisionContentPartText, VisionContentPartImage]]]

class ChatCompletionRequest(BaseModel):
    model: str # e.g., "step-1-8k", "step-1v-8k"
    messages: List[Union[TextMessage, VisionMessage]]
    # Add other optional parameters based on API docs if needed (e.g., temperature, max_tokens)
    stream: Optional[bool] = None # Example optional parameter

class ChatCompletionChoice(BaseModel):
    index: int
    message: TextMessage # Assuming response message structure is consistent
    finish_reason: Optional[str] = None

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str # e.g., "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage

# --- Image Generation Models ---
class ImageGenerationRequest(BaseModel):
    model: str = "step-1x-medium" # Default from example
    prompt: str
    n: Optional[int] = None # Example shows it's optional
    size: Optional[str] = None # Example doesn't specify, API likely has default
    seed: Optional[int] = None # From example
    response_format: str = "b64_json" # From example, or "url"
    # Add other optional parameters like negative_prompt, style_raw, etc.

class ImageGenerationDataB64(BaseModel):
    b64_json: str
    # Add other fields if present, like 'revised_prompt'

class ImageGenerationDataUrl(BaseModel):
    url: str
    # Add other fields if present

class ImageGenerationResponse(BaseModel):
    created: int
    # Data format depends on response_format
    data: List[Union[ImageGenerationDataB64, ImageGenerationDataUrl]]

# --- Speech Generation Models ---
class SpeechGenerationRequest(BaseModel):
    model: str = "step-tts-mini" # Default from example
    input: str
    voice: str = "cixingnansheng" # Default from example
    # Add other optional parameters like speed, format (e.g., "mp3", "wav")

# Response for speech is raw audio bytes, no Pydantic model needed for response body

# --- StepFun Client ---

class StepFunClient:
    """
    Client for interacting with the StepFun API.
    """

    def __init__(self):
        self.api_key = os.getenv(ENV_STEPFUN_API_KEY)
        self.api_host = os.getenv(ENV_STEPFUN_API_HOST, "https://api.stepfun.com") # Default host

        if not self.api_key:
            raise StepFunAuthError(
                f"API key not found. Please set the {ENV_STEPFUN_API_KEY} environment variable."
            )

        self.base_url = f"{self.api_host}/v1" # Confirmed from examples
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        logger.info(f"StepFunClient initialized. Base URL: {self.base_url}")

    def _request(self, method: str, endpoint: str, expect_json: bool = True, **kwargs) -> Union[requests.Response, Dict[str, Any]]:
        """Makes an HTTP request to the StepFun API."""
        url = f"{self.base_url}{endpoint}"
        # Ensure data is passed as 'json' for POST/PUT/PATCH with JSON body
        if method.upper() in ["POST", "PUT", "PATCH"] and 'json' in kwargs:
             # Pydantic models are passed directly to requests' json parameter
             pass # Already handled if kwargs['json'] is a Pydantic model or dict
        elif method.upper() in ["POST", "PUT", "PATCH"] and 'data' in kwargs:
             # If data is provided differently (e.g., form data), handle here
             pass


        logger.debug(f"Requesting {method} {url} with headers: {self.headers} and kwargs: {kwargs}")
        try:
            response = requests.request(method, url, headers=self.headers, **kwargs)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            logger.debug(f"Response status: {response.status_code}")
            if expect_json:
                json_response = response.json()
                logger.debug(f"Response JSON (truncated): {str(json_response)[:500]}...")
                return json_response
            else:
                logger.debug(f"Response Content (binary/text, truncated): {response.content[:500]}...")
                return response # Return the full response object for non-JSON cases (like audio)
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            # Improved error message extraction
            error_message = str(e)
            response_content = ""
            if hasattr(e, 'response') and e.response is not None:
                 status_code = e.response.status_code
                 try:
                     # Try to get JSON error details
                     error_data = e.response.json()
                     # Extract message based on observed StepFun error structure
                     message = error_data.get("error", {}).get("message", e.response.text)
                     error_type = error_data.get("error", {}).get("type", "unknown_error")
                     error_message = f"API request error ({status_code} - {error_type}): {message}"
                 except ValueError: # Handle cases where response is not JSON
                     response_content = e.response.text
                     error_message = f"API request error ({status_code}): {response_content[:200]}" # Truncate non-JSON response

                 # Raise specific exceptions based on status code or type
                 # REMOVED <mcreference></mcreference> tags from the end of the next line
                 if status_code == 401 or error_type == "invalid_api_key":
                     raise StepFunAuthError(error_message) from e
                 else:
                     raise StepFunRequestError(error_message) from e
            else:
                 # Network errors or other RequestExceptions
                 raise StepFunAPIError(f"Network or request error: {error_message}") from e

    def create_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Calls the chat completion API (text or vision)."""
        endpoint = "/chat/completions"
        # Use model_dump to convert Pydantic model to dict.
        # Change from exclude_unset=True to exclude_none=True.
        # This ensures default fields like 'type' in vision content parts are included,
        # which is likely required by the API based on the curl examples and the error message.
        payload = request.model_dump(exclude_none=True) # Changed from exclude_unset=True

        response_json = self._request("POST", endpoint, json=payload, expect_json=True)
        return ChatCompletionResponse(**response_json)

    # Keep separate methods for clarity, even if endpoint is the same
    def create_text_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
         """Convenience method for text-only chat completion."""
         # Ensure messages content is string if using this method? Or rely on caller.
         # For now, just call the main chat completion method.
         return self.create_chat_completion(request)

    def create_vision_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
         """Convenience method for vision chat completion."""
         # Ensure messages content includes image parts if using this method? Or rely on caller.
         # For now, just call the main chat completion method.
         return self.create_chat_completion(request)


    def create_image_generation(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """Calls the image generation API."""
        endpoint = "/images/generations"
        response_json = self._request("POST", endpoint, json=request.model_dump(exclude_unset=True), expect_json=True)
        return ImageGenerationResponse(**response_json)

    def create_speech_generation(self, request: SpeechGenerationRequest) -> bytes:
        """Calls the speech generation (TTS) API."""
        endpoint = "/audio/speech"
        # Make the request, expecting non-JSON response
        response = self._request("POST", endpoint, json=request.model_dump(exclude_unset=True), expect_json=False)
        # Return the raw audio content
        if isinstance(response, requests.Response):
             return response.content
        else:
             # Should not happen if expect_json=False, but handle defensively
             logger.error("Speech generation did not return a Response object as expected.")
             raise StepFunAPIError("Unexpected response type for speech generation.")


# Example Usage (for testing purposes)
if __name__ == "__main__":
    # --- Setup Logging ---
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_handler_console = logging.StreamHandler()
    log_handler_console.setFormatter(log_formatter)

    # Create a directory for test outputs and logs
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    log_file_path = output_dir / "client_test.log"

    log_handler_file = logging.FileHandler(log_file_path)
    log_handler_file.setFormatter(log_formatter)

    # Get the root logger and add handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # Set root logger level
    root_logger.addHandler(log_handler_console)
    root_logger.addHandler(log_handler_file)

    logger.info(f"Logging configured. Log file: {log_file_path}")


    # Ensure .env file exists and STEPFUN_API_KEY is set for testing
    if not os.getenv(ENV_STEPFUN_API_KEY):
        print(f"Warning: {ENV_STEPFUN_API_KEY} not set in environment. API calls will likely fail.")
        # Optionally create a dummy client for basic checks
        # client = StepFunClient() # This would raise StepFunAuthError
        print("Skipping API call tests.")
        
    else:
        try:
            client = StepFunClient()
            print("StepFunClient initialized successfully.")
            logger.info("StepFunClient initialized successfully for testing.")

            # --- Test Text Completion ---
            print("=" * 20)
            print("\nTesting Text Completion...")
            text_req = ChatCompletionRequest(
                model="step-1-8k", # From example
                messages=[
                    TextMessage(role="system", content="你是由阶跃星辰提供的AI聊天助手。"),
                    TextMessage(role="user", content="你好！阶跃星辰有自己的MCP吗？")
                ]
            )
            try:
                text_res = client.create_text_completion(text_req)
                print("Text Completion Response:", text_res.choices[0].message.content)
                logger.info(f"Text Completion Success. Response: {text_res.choices[0].message.content}")
            except StepFunAPIError as e:
                print(f"Text Completion Failed: {e}")
                logger.error(f"Text Completion Failed: {e}", exc_info=True)


            # --- Test Image Generation ---
            print("=" * 20)
            print("\nTesting Image Generation...")
            logger.info("Testing Image Generation...")
            img_req = ImageGenerationRequest(
                model="step-1x-medium", # From example
                prompt="一只可爱的猫咪在赛博朋克风格的街道上", # More descriptive prompt
                seed=12345,
                response_format="b64_json" # Ensure b64 for saving
            )
            try:
                img_res = client.create_image_generation(img_req)
                print(f"Image Generation Response: Received {len(img_res.data)} image(s).")
                logger.info(f"Image Generation Success: Received {len(img_res.data)} image(s).")
                # Save the image(s)
                for i, img_data in enumerate(img_res.data):
                    if isinstance(img_data, ImageGenerationDataB64):
                        try:
                            img_bytes = base64.b64decode(img_data.b64_json)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            # Create a safer filename from the prompt
                            safe_prompt = "".join(c if c.isalnum() else "_" for c in img_req.prompt[:30])
                            img_filename = output_dir / f"img_{timestamp}_{safe_prompt}_{i}.png"
                            with open(img_filename, "wb") as f:
                                f.write(img_bytes)
                            print(f"Saved image {i+1} to {img_filename}")
                            logger.info(f"Saved generated image {i+1} to {img_filename}")
                        except (base64.binascii.Error, IOError) as save_err:
                             print(f"Error saving image {i+1}: {save_err}")
                             logger.error(f"Error saving image {i+1}: {save_err}", exc_info=True)
                    elif isinstance(img_data, ImageGenerationDataUrl):
                         print(f"Received image URL (cannot save directly): {img_data.url}")
                         logger.warning(f"Received image URL instead of b64_json: {img_data.url}")
                    else:
                         print(f"Received unexpected image data format: {type(img_data)}")
                         logger.warning(f"Received unexpected image data format: {type(img_data)}")

            except StepFunAPIError as e:
                print(f"Image Generation Failed: {e}")
                logger.error(f"Image Generation Failed: {e}", exc_info=True)


            # --- Test Speech Generation ---
            print("=" * 20)
            print("\nTesting Speech Generation...")
            logger.info("Testing Speech Generation...")
            speech_req = SpeechGenerationRequest(
                model="step-tts-mini", # From example
                input="你好，这是一个来自阶跃星辰的语音合成测试。",
                voice="cixingnansheng" # From example
            )
            try:
                speech_res_bytes = client.create_speech_generation(speech_req)
                print(f"Speech Generation Response: Received {len(speech_res_bytes)} bytes of audio data.")
                logger.info(f"Speech Generation Success: Received {len(speech_res_bytes)} bytes of audio data.")
                # Save the audio to a file
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_input = "".join(c if c.isalnum() else "_" for c in speech_req.input[:30])
                    audio_filename = output_dir / f"speech_{timestamp}_{safe_input}.mp3" # Assume mp3 format
                    with open(audio_filename, "wb") as f:
                        f.write(speech_res_bytes)
                    print(f"Saved audio to {audio_filename}")
                    logger.info(f"Saved generated audio to {audio_filename}")
                except IOError as save_err:
                    print(f"Error saving audio file: {save_err}")
                    logger.error(f"Error saving audio file: {save_err}", exc_info=True)

            except StepFunAPIError as e:
                print(f"Speech Generation Failed: {e}")
                logger.error(f"Speech Generation Failed: {e}", exc_info=True)


            # --- Test Vision Completion (Requires a valid image data URI) ---
            print("=" * 20)
            print("\nTesting Vision Completion...")
            # NOTE: Replace with a real image data URI for actual testing
            # You might need a helper function to load an image file and convert to base64 data URI
            dummy_image_data_uri = "https://pic3.zhimg.com/v2-aad0f6de13653215f265e72f51b236ba_1440w.jpg"
            vision_req = ChatCompletionRequest(
                 model="step-1v-8k", # From example
                 messages=[
                     VisionMessage(role="user", content=[
                         VisionContentPartText(text="详细介绍一下这张图片"),
                         VisionContentPartImage(image_url={"url": dummy_image_data_uri})
                     ])
                 ]
            )
            try:
                 vision_res = client.create_vision_completion(vision_req)
                 print("Vision Completion Response:", vision_res.choices[0].message.content)
                 logger.info(f"Vision Completion Success. Response: {vision_res.choices[0].message.content}")
            except StepFunAPIError as e:
                 print(f"Vision Completion Failed: {e}")
                 logger.error(f"Vision Completion Failed: {e}", exc_info=True)


        except StepFunAuthError as e:
             print(f"Authentication Error: {e}")
             logger.critical(f"Authentication Error during testing: {e}", exc_info=True)
        except StepFunAPIError as e:
            print(f"An unexpected API error occurred during testing: {e}")
            logger.error(f"An unexpected API error occurred during testing: {e}", exc_info=True)
        except Exception as e: # Catch any other unexpected errors
             print(f"An unexpected error occurred: {e}")
             logger.exception(f"An unexpected error occurred during testing: {e}")