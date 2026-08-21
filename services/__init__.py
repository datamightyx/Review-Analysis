"""Services package for Returns Analysis."""

from services.constants import *
from services.exceptions import *
from services.credentials import CredentialsManager, get_credentials_panel
from services.data_loader import (
    get_access_token,
    fetch_customer_returns,
    parse_customer_returns,
    normalize_returns_frame,
    fetch_refund_events,
    parse_transaction_refunds,
    refunds_to_return_rows,
    load_refund_only_rows,
    validate_date_range,
)
from services.excel_builder import (
    build_workbook,
    save_workbook,
    workbook_to_bytes,
    classify_comment,
    get_status,
    get_true_reason,
)
from services.analysis_service import ReturnsAnalysisService, create_service, AnalysisConfig, AnalysisResult

__all__ = [
    # Constants
    "REQUIRED_ENV",
    "ENDPOINTS",
    "MARKETPLACES",
    "CUSTOM_MP",
    "FIELD_LABEL",
    "SECRET_PATHS",
    "MAX_DATE_RANGE_DAYS",
    "DEFAULT_DATE_RANGE_DAYS",
    "DEFAULT_RELEASE_LAG_DAYS",
    "MAX_RELEASE_LAG_DAYS",
    "FINANCES_API_RATE_LIMIT_DELAY",
    "REPORT_POLL_INTERVAL",
    "REPORT_POLL_TIMEOUT",
    "CACHE_TTL_SECONDS",
    "REFUND_NO_RETURN",
    "STATUS_REFUND_ONLY",
    "KW",
    "REASON_OK",
    "TRUE_REASON_LABEL",
    "TRIVIAL",
    "TOPIC_DISPLAY",
    "TOPIC_ROW_COLOR",
    "C_HEADER",
    "C_HEADER_FG",
    "C_MISMATCH",
    "C_MATCH",
    "C_UNCLEAR",
    "C_NOCOMMENT",
    "C_SUMMARY_H",
    "C_ALT_ROW",
    "C_REFUNDONLY",
    "STATUS_FILL_COLORS",
    "DATE_PRESETS",
    # Exceptions
    "ReturnsAnalysisError",
    "ConfigurationError",
    "AuthenticationError",
    "APIError",
    "RateLimitError",
    "ReportError",
    "DataError",
    "ValidationError",
    "ExportError",
    # Credentials
    "CredentialsManager",
    "get_credentials_panel",
    # Data Loader
    "get_access_token",
    "fetch_customer_returns",
    "parse_customer_returns",
    "normalize_returns_frame",
    "fetch_refund_events",
    "parse_transaction_refunds",
    "refunds_to_return_rows",
    "load_refund_only_rows",
    "validate_date_range",
    # Excel Builder
    "build_workbook",
    "save_workbook",
    "workbook_to_bytes",
    "classify_comment",
    "get_status",
    "get_true_reason",
    # Analysis Service
    "ReturnsAnalysisService",
    "create_service",
    "AnalysisConfig",
    "AnalysisResult",
]