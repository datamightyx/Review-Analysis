"""Custom exceptions for Returns Analysis."""

from typing import Any, Optional


class ReturnsAnalysisError(Exception):
    """Base exception for Returns Analysis errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(ReturnsAnalysisError):
    """Raised when configuration is invalid or missing."""
    pass


class AuthenticationError(ReturnsAnalysisError):
    """Raised when SP-API authentication fails."""
    pass


class APIError(ReturnsAnalysisError):
    """Raised when SP-API request fails."""
    
    def __init__(
        self, 
        message: str, 
        status_code: Optional[int] = None,
        response_body: Optional[Any] = None,
        details: Optional[dict] = None
    ):
        super().__init__(message, details)
        self.status_code = status_code
        self.response_body = response_body


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded (429)."""
    pass


class ReportError(ReturnsAnalysisError):
    """Raised when report generation fails."""
    
    def __init__(
        self, 
        message: str, 
        report_id: Optional[str] = None,
        status: Optional[str] = None,
        details: Optional[dict] = None
    ):
        super().__init__(message, details)
        self.report_id = report_id
        self.status = status


class DataError(ReturnsAnalysisError):
    """Raised when data parsing/validation fails."""
    pass


class ValidationError(ReturnsAnalysisError):
    """Raised when input validation fails."""
    pass


class ExportError(ReturnsAnalysisError):
    """Raised when export generation fails."""
    pass