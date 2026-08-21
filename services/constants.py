"""Constants and configuration for Returns Analysis."""

from datetime import timedelta

# API Configuration
LWA_TOKEN_URL = "https://api.amazon.com/auth/o2/token"
RETURNS_REPORT_TYPE = "GET_FBA_FULFILLMENT_CUSTOMER_RETURNS_DATA"
FINANCES_PATH = "/finances/v0/financialEvents"

# Required environment variables
REQUIRED_ENV = [
    "SP_API_LWA_CLIENT_ID",
    "SP_API_LWA_CLIENT_SECRET",
    "SP_API_REFRESH_TOKEN",
    "SP_API_MARKETPLACE_ID",
    "SP_API_ENDPOINT",
]

# SP-API Endpoints
ENDPOINTS = {
    "North America (NA)": "https://sellingpartnerapi-na.amazon.com",
    "Europe (EU)": "https://sellingpartnerapi-eu.amazon.com",
    "Far East (FE)": "https://sellingpartnerapi-fe.amazon.com",
}

# Marketplace IDs
MARKETPLACES = {
    "ATVPDKIKX0DER": "US — amazon.com",
    "A2EUQ1WTGCTBG2": "CA — amazon.ca",
    "A1AM78C64UM0Y8": "MX — amazon.com.mx",
    "A2Q3Y263D00KWC": "BR — amazon.com.br",
    "A1F83G8C2ARO7P": "UK — amazon.co.uk",
    "A1PA6795UKMFR9": "DE — amazon.de",
    "A13V1IB3VIYZZH": "FR — amazon.fr",
    "APJ6JRA9NG5V4": "IT — amazon.it",
    "A1RKKUPIHCS9HS": "ES — amazon.es",
    "A1805IZSGTT6HS": "NL — amazon.nl",
    "A21TJRUUN4KGV": "IN — amazon.in",
    "A1VC38T7YXB528": "JP — amazon.co.jp",
    "A39IBJ37TRP1C6": "AU — amazon.com.au",
}
CUSTOM_MP = "Інший (ввести вручну)"

# Field labels for UI
FIELD_LABEL = {
    "SP_API_LWA_CLIENT_ID": "LWA Client ID",
    "SP_API_LWA_CLIENT_SECRET": "LWA Client Secret",
    "SP_API_REFRESH_TOKEN": "Refresh Token",
    "SP_API_MARKETPLACE_ID": "Marketplace ID",
    "SP_API_ENDPOINT": "Endpoint",
}

# Secret paths for Streamlit secrets
SECRET_PATHS = {
    k: (("sp_api", k.replace("SP_API_", "").lower()), ("sp_api", k), (None, k))
    for k in REQUIRED_ENV
}

# Date range constraints
MAX_DATE_RANGE_DAYS = 180
DEFAULT_DATE_RANGE_DAYS = 90
DEFAULT_RELEASE_LAG_DAYS = 7
MAX_RELEASE_LAG_DAYS = 14

# Rate limiting
FINANCES_API_RATE_LIMIT_DELAY = 2.2  # seconds between requests (0.5 req/sec)
REPORT_POLL_INTERVAL = 20  # seconds
REPORT_POLL_TIMEOUT = 1800  # seconds (30 minutes)

# Cache TTL
CACHE_TTL_SECONDS = 3600

# Refund reconciliation
REFUND_NO_RETURN = "REFUND_NO_RETURN"
STATUS_REFUND_ONLY = "Refund - No Return"

# Keyword dictionaries for comment classification
KW = {
    'SIZE_TOO_LARGE': [
        'too large', 'too big', 'too wide', 'too long', 'too tall', 'runs large',
        'incorrect size', 'wrong size', 'size is wrong', 'too bulky', 'size too big',
        'bigger than', 'larger than', 'size larger', 'bit too big',
    ],
    'SIZE_TOO_SMALL': [
        'too small', 'too tight', 'too narrow', 'too short', 'runs small', 'size too small',
        'smaller than', 'too tiny', 'not big enough', 'too little', 'size smaller',
    ],
    'DEFECTIVE': [
        'defective', 'broken', "doesn't work", 'does not work', 'not work', 'broke',
        'malfunction', 'stopped working', 'falling off', 'falls off', 'cracked',
        'leaking', ' leak ', 'torn', 'ripped', 'damaged', 'doesnt function',
        "doesn't function", 'no longer work',
    ],
    'QUALITY_ISSUE': [
        'poor quality', 'bad quality', 'too thin', 'very thin', 'flimsy', 'not durable',
        'low quality', 'inferior', 'terrible quality', 'bad material',
        'quality is bad', 'not good quality', 'not the same quality', 'cheap quality',
        'quality is poor', 'material is thin',
    ],
    'NOT_AS_DESCRIBED': [
        'not as described', 'not as expected', 'not what i expected',
        'misleading', 'misrepresented', 'inaccurate description',
    ],
    'WRONG_ITEM': [
        'wrong item', 'wrong product', 'wrong color', 'ordered wrong',
    ],
    'DELIVERY_ISSUE': [
        'never arrived', 'did not arrive', 'not delivered', 'never received',
        'late delivery', 'delayed', 'shipping was', 'weeks delayed', 'not arrive',
        'took too long', 'not received',
    ],
    'CHANGED_MIND': [
        'changed mind', "don't need", 'do not need', 'no longer need',
        'no longer want', 'changed my mind', "don't want", 'dont need', 'dont want',
    ],
    'BETTER_PRICE': [
        'found cheaper', 'better price', 'found better price', 'cheaper elsewhere',
        'cheaper on', 'lower price',
    ],
    'SIZE_ISSUE': [
        ' size ', ' fit ', ' fits ', 'fitting', 'incorrect size', 'wrong size',
        "doesn't fit", "does not fit", "wont fit", "won't fit",
    ],
}

REASON_OK = {
    'APPAREL_TOO_LARGE': ['SIZE_TOO_LARGE', 'SIZE_ISSUE'],
    'APPAREL_TOO_SMALL': ['SIZE_TOO_SMALL', 'SIZE_ISSUE'],
    'POOR_FIT': ['SIZE_TOO_LARGE', 'SIZE_TOO_SMALL', 'SIZE_ISSUE'],
    'DEFECTIVE': ['DEFECTIVE'],
    'NOT_AS_DESCRIBED': ['NOT_AS_DESCRIBED', 'QUALITY_ISSUE', 'SIZE_TOO_LARGE', 'SIZE_TOO_SMALL'],
    'QUALITY_UNACCEPTABLE': ['QUALITY_ISSUE', 'DEFECTIVE'],
    'ORDERED_WRONG_ITEM': ['WRONG_ITEM'],
    'UNWANTED_ITEM': ['CHANGED_MIND', 'BETTER_PRICE'],
    'FOUND_BETTER_PRICE': ['BETTER_PRICE', 'CHANGED_MIND'],
    'MISSING_PARTS': ['DEFECTIVE'],
    'DAMAGED_BY_FC': ['DEFECTIVE'],
    'DAMAGED_BY_CARRIER': ['DEFECTIVE'],
    'NEVER_ARRIVED': ['DELIVERY_ISSUE'],
    'MISSED_ESTIMATED_DELIVERY': ['DELIVERY_ISSUE'],
    'UNDELIVERABLE_UNKNOWN': ['DELIVERY_ISSUE'],
    'UNDELIVERABLE_REFUSED': ['DELIVERY_ISSUE'],
    'NO_REASON_GIVEN': [],
    'SWITCHEROO': ['WRONG_ITEM'],
    'NOT_COMPATIBLE': ['NOT_AS_DESCRIBED', 'SIZE_ISSUE'],
    'PART_NOT_COMPATIBLE': ['NOT_AS_DESCRIBED', 'SIZE_ISSUE'],
    'MISORDERED': ['WRONG_ITEM'],
    'EXTRA_ITEM': [],
    'EXCESSIVE_INSTALLATION': [],
    'UNAUTHORIZED_PURCHASE': [],
}

TRUE_REASON_LABEL = {
    'SIZE_TOO_LARGE': 'SIZE - Too Large',
    'SIZE_TOO_SMALL': 'SIZE - Too Small',
    'SIZE_ISSUE': 'SIZE - Fit Issue',
    'DEFECTIVE': 'DEFECTIVE',
    'QUALITY_ISSUE': 'QUALITY ISSUE',
    'NOT_AS_DESCRIBED': 'NOT AS DESCRIBED',
    'WRONG_ITEM': 'WRONG ITEM',
    'DELIVERY_ISSUE': 'DELIVERY ISSUE',
    'CHANGED_MIND': 'CHANGED MIND',
    'BETTER_PRICE': 'FOUND BETTER PRICE',
}

TRIVIAL = {'', 'na', 'n/a', 'no', 'yes', 'z', 'return', 'ok', 'none', '.', '-', 'n'}

TOPIC_DISPLAY = {
    'SIZE_TOO_LARGE': 'SIZE — Too Large',
    'SIZE_TOO_SMALL': 'SIZE — Too Small',
    'SIZE_ISSUE': 'SIZE — Fit Issue',
    'DEFECTIVE': 'Defective / Not Working',
    'QUALITY_ISSUE': 'Quality Issue',
    'NOT_AS_DESCRIBED': 'Not As Described',
    'WRONG_ITEM': 'Wrong Item Sent',
    'DELIVERY_ISSUE': 'Delivery Issue',
    'CHANGED_MIND': 'Changed Mind',
    'BETTER_PRICE': 'Found Better Price',
    'OTHER': 'Other / Unclear',
    'NO_COMMENT': 'No Comment',
}

TOPIC_ROW_COLOR = {
    'SIZE_TOO_LARGE': 'FDE9D9',
    'SIZE_TOO_SMALL': 'FDE9D9',
    'SIZE_ISSUE': 'FDE9D9',
    'DEFECTIVE': 'FFD7D7',
    'QUALITY_ISSUE': 'EAD1DC',
    'NOT_AS_DESCRIBED': 'D9E1F2',
    'WRONG_ITEM': 'FCE4D6',
    'DELIVERY_ISSUE': 'DDEBF7',
    'CHANGED_MIND': 'E2EFDA',
    'BETTER_PRICE': 'D9F2F2',
    'OTHER': 'F2F2F2',
}

# Excel styling constants
C_HEADER = '1F3864'
C_HEADER_FG = 'FFFFFF'
C_MISMATCH = 'FFD7D7'
C_MATCH = 'D7F0D7'
C_UNCLEAR = 'FFF3CC'
C_NOCOMMENT = 'F2F2F2'
C_SUMMARY_H = '2E75B6'
C_ALT_ROW = 'EEF3FA'
C_REFUNDONLY = 'E4DFEC'

STATUS_FILL_COLORS = {
    'Mismatch': C_MISMATCH,
    'Match': C_MATCH,
    'Unclear': C_UNCLEAR,
    'No Comment': C_NOCOMMENT,
    STATUS_REFUND_ONLY: C_REFUNDONLY,
}

# Date presets for UI
DATE_PRESETS = {
    "Last 7 days": 7,
    "Last 30 days": 30,
    "Last 90 days": 90,
    "Last 180 days": 180,
}