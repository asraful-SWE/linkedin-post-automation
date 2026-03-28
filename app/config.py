"""
config.py - Application settings for LinkedIn AI Poster.

All configuration is driven by environment variables (or a `.env` file).
Values are validated and typed by Pydantic v2's BaseSettings.

Typical usage
-------------
    from linkedin_ai_poster.app.config import get_settings

    settings = get_settings()
    print(settings.openai_model)

Environment file
----------------
Copy `.env.example` to `.env` and fill in the required secrets.  The file is
loaded automatically; any variable set in the real environment takes priority
over the file.

Singleton
---------
`get_settings()` is decorated with `@lru_cache()` so the heavy validation step
(env parsing, type coercion) only runs once per process.  Call
`get_settings.cache_clear()` in tests before patching environment variables.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Unified application settings.

    All fields map 1-to-1 to an environment variable of the same name
    (case-insensitive).  Pydantic handles type coercion and validation.

    Grouping (for readability only – there is a single flat Settings object):
        * LinkedIn
        * Database
        * OpenAI
        * Email
        * Scheduler
        * Security
        * Image
        * Celery
        * Content
        * Observability
        * Base / Server
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # silently ignore unknown env vars
        case_sensitive=False,  # LINKEDIN_ACCESS_TOKEN == linkedin_access_token
    )

    # ------------------------------------------------------------------
    # LinkedIn
    # ------------------------------------------------------------------

    linkedin_access_token: Optional[str] = Field(
        default=None,
        description="OAuth 2.0 access token for the LinkedIn API.",
    )
    linkedin_person_id: Optional[str] = Field(
        default=None,
        description="LinkedIn member URN (e.g. 'urn:li:person:XXXXXXXX').",
    )
    mock_linkedin_posting: bool = Field(
        default=False,
        description=(
            "When True, skip real LinkedIn API calls and log the payload "
            "instead.  Useful for development / CI."
        ),
    )

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    database_url: str = Field(
        default="sqlite:///linkedin_poster.db",
        description="SQLAlchemy-compatible database URL.",
    )
    db_path: str = Field(
        default="linkedin_poster.db",
        description=(
            "Filesystem path used directly by lightweight DB helpers "
            "(e.g. aiosqlite).  Ignored when database_url points at a "
            "non-SQLite engine."
        ),
    )

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------

    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI secret key (sk-…).",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI chat-completion model identifier.",
    )

    # ------------------------------------------------------------------
    # Email  (supports both SMTP and Resend transports)
    # ------------------------------------------------------------------

    # --- SMTP ---
    smtp_host: Optional[str] = Field(
        default=None,
        description="SMTP server hostname or IP address.",
    )
    smtp_port: int = Field(
        default=587,
        ge=1,
        le=65535,
        description="SMTP server port (587 = STARTTLS, 465 = SSL, 25 = plain).",
    )
    smtp_user: Optional[str] = Field(
        default=None,
        description="SMTP authentication username.",
    )
    smtp_password: Optional[str] = Field(
        default=None,
        description="SMTP authentication password.",
    )
    smtp_use_tls: bool = Field(
        default=True,
        description="Upgrade the connection to TLS via STARTTLS.",
    )
    smtp_use_ssl: bool = Field(
        default=False,
        description=(
            "Use implicit SSL/TLS from the start (port 465).  "
            "Mutually exclusive with smtp_use_tls."
        ),
    )

    # --- Resend ---
    resend_api_key: Optional[str] = Field(
        default=None,
        description="Resend.com API key (re_…).",
    )
    resend_from: Optional[str] = Field(
        default=None,
        description="Sender address used by the Resend transport.",
    )

    # --- Generic email ---
    email_from: Optional[str] = Field(
        default=None,
        description=(
            "Default 'From' address for outgoing emails (used by both "
            "SMTP and Resend transports if the transport-specific field "
            "is not set)."
        ),
    )
    approval_email_to: Optional[str] = Field(
        default=None,
        description="Recipient address for post-approval notification emails.",
    )

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    timezone: str = Field(
        default="Asia/Dhaka",
        description="IANA timezone string used by APScheduler / Celery beat.",
    )
    max_posts_per_day: int = Field(
        default=2,
        ge=1,
        description="Hard cap on the number of LinkedIn posts per calendar day.",
    )
    min_hours_between_posts: float = Field(
        default=4.0,
        ge=0.0,
        description="Minimum gap (hours) enforced between two consecutive posts.",
    )
    auto_schedule_enabled: bool = Field(
        default=True,
        description="Whether the scheduler should automatically queue new posts.",
    )
    test_mode: bool = Field(
        default=False,
        description=(
            "When True, reduce delays / intervals to seconds instead of hours "
            "so that the full pipeline can be exercised quickly in tests."
        ),
    )

    # ------------------------------------------------------------------
    # Security
    # ------------------------------------------------------------------

    approval_secret: str = Field(
        default="change-this-secret",
        description=(
            "HMAC secret used to sign / verify approval tokens.  "
            "MUST be changed to a long random value in production."
        ),
    )
    admin_api_key: Optional[str] = Field(
        default=None,
        description=(
            "Static API key for the admin endpoints.  "
            "Sent via the X-Admin-Key request header."
        ),
    )
    approval_token_expires_hours: int = Field(
        default=24,
        ge=1,
        description="Lifetime of one-time approval tokens in hours.",
    )

    # ------------------------------------------------------------------
    # Image
    # ------------------------------------------------------------------

    unsplash_access_key: Optional[str] = Field(
        default=None,
        description="Unsplash API access key for royalty-free image search.",
    )
    pexels_api_key: Optional[str] = Field(
        default=None,
        description="Pexels API key for royalty-free image search.",
    )
    enable_images: bool = Field(
        default=False,
        description="Attach a header image to generated LinkedIn posts.",
    )
    image_max_size_mb: int = Field(
        default=5,
        ge=1,
        description="Maximum allowed image size in megabytes before rejection.",
    )

    # ------------------------------------------------------------------
    # Celery
    # ------------------------------------------------------------------

    celery_broker_url: str = Field(
        default="redis://localhost:6379/0",
        description="Celery broker URL (Redis, RabbitMQ, …).",
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/1",
        description="Celery result backend URL.",
    )
    use_celery: bool = Field(
        default=False,
        description=(
            "Route background tasks through Celery instead of running them "
            "in-process with asyncio.  Requires a running broker."
        ),
    )

    # ------------------------------------------------------------------
    # Content
    # ------------------------------------------------------------------

    content_score_threshold: float = Field(
        default=6.0,
        ge=0.0,
        le=10.0,
        description=(
            "Minimum quality score (0–10) a generated post must achieve "
            "before it is queued for approval."
        ),
    )
    max_regeneration_attempts: int = Field(
        default=3,
        ge=1,
        description=(
            "How many times the system will ask the LLM to regenerate a "
            "post that scored below content_score_threshold."
        ),
    )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    log_level: str = Field(
        default="INFO",
        description="Root logger level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    debug: bool = Field(
        default=False,
        description="Enable FastAPI debug mode and verbose exception output.",
    )
    enable_json_logs: bool = Field(
        default=False,
        description="Emit structured JSON log lines instead of plain text.",
    )

    # ------------------------------------------------------------------
    # Base / Server
    # ------------------------------------------------------------------

    base_url: str = Field(
        default="http://localhost:8000",
        description=(
            "Publicly reachable base URL of this service.  "
            "Used when constructing absolute callback / approval URLs."
        ),
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="TCP port uvicorn will bind to.",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Network interface uvicorn will listen on.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, v: str) -> str:
        """Accept lowercase variants and normalise to uppercase."""
        normalised = v.strip().upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalised not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got '{v}'.")
        return normalised

    @field_validator("base_url", mode="before")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        """Ensure base_url never ends with a slash for consistent URL joining."""
        return v.rstrip("/")

    @model_validator(mode="after")
    def _check_smtp_tls_ssl_mutual_exclusion(self) -> "Settings":
        """smtp_use_tls and smtp_use_ssl must not both be True."""
        if self.smtp_use_tls and self.smtp_use_ssl:
            raise ValueError(
                "smtp_use_tls and smtp_use_ssl are mutually exclusive. "
                "Set only one of them to True."
            )
        return self

    @model_validator(mode="after")
    def _warn_default_approval_secret(self) -> "Settings":
        """Emit a warning (not an error) when the default secret is used."""
        import logging

        if self.approval_secret == "change-this-secret" and not self.debug:
            logging.getLogger(__name__).warning(
                "approval_secret is set to the default value. "
                "This MUST be changed to a long random secret in production."
            )
        return self

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def is_linkedin_configured(self) -> bool:
        """True when both LinkedIn credentials are present."""
        return bool(self.linkedin_access_token and self.linkedin_person_id)

    @property
    def is_openai_configured(self) -> bool:
        """True when the OpenAI API key is set."""
        return bool(self.openai_api_key)

    @property
    def is_smtp_configured(self) -> bool:
        """True when the minimum SMTP fields are present."""
        return bool(self.smtp_host and self.smtp_user and self.smtp_password)

    @property
    def is_resend_configured(self) -> bool:
        """True when the Resend API key is set."""
        return bool(self.resend_api_key)

    @property
    def effective_email_from(self) -> Optional[str]:
        """Return the most specific 'From' address available."""
        return self.resend_from or self.email_from

    @property
    def image_max_size_bytes(self) -> int:
        """image_max_size_mb converted to bytes for direct comparison."""
        return self.image_max_size_mb * 1024 * 1024


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


@lru_cache()
def get_settings() -> Settings:
    """Return the application-wide :class:`Settings` singleton.

    The instance is created on the first call and then cached for the lifetime
    of the process.  This means ``.env`` is read only once, which is both
    efficient and predictable.

    Testing
    -------
    Patch the environment *before* the first call, or call
    ``get_settings.cache_clear()`` between test cases::

        from linkedin_ai_poster.app.config import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        settings = get_settings()
        assert settings.openai_api_key == "test-key"
    """
    return Settings()
