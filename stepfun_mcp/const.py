"""Constants for the StepFun MCP server."""

# Environment variable names
ENV_STEPFUN_API_KEY = "STEPFUN_API_KEY"
ENV_STEPFUN_API_HOST = "STEPFUN_API_HOST" # Or specific endpoint base URLs
ENV_STEPFUN_MCP_BASE_PATH = "STEPFUN_MCP_BASE_PATH"
ENV_RESOURCE_MODE = "STEPFUN_API_RESOURCE_MODE" # url or local
ENV_FASTMCP_LOG_LEVEL = "FASTMCP_LOG_LEVEL"

# Resource modes
RESOURCE_MODE_LOCAL = "local"
RESOURCE_MODE_URL = "url"

# Default values (adjust based on StepFun API defaults or desired behavior)
DEFAULT_TEXT_MODEL = "step-1-8k" 
DEFAULT_VISION_MODEL = "cogvlm-chat" 
DEFAULT_IMAGE_MODEL = "step-1x-medium" # Example
DEFAULT_SPEECH_MODEL = "step-tts-mini" # Example
DEFAULT_VOICE_ID = "cixingnansheng" # Example

# Add other constants as needed