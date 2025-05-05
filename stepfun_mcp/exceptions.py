"""Custom exceptions for the StepFun MCP server."""

class StepFunAPIError(Exception):
    """Base exception for StepFun API errors."""
    pass

class StepFunAuthError(StepFunAPIError):
    """Exception for authentication errors."""
    pass

class StepFunRequestError(StepFunAPIError):
    """Exception for general request errors."""
    pass

# Add more specific exceptions as needed