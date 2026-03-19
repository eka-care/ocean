class OceanError(Exception):
    """Base exception for Ocean."""


class ProviderError(OceanError):
    """Error from a provider SDK."""


class UnsupportedModalityError(OceanError):
    """Provider does not support the requested modality."""


class ModelNotFoundError(OceanError):
    """Model name could not be resolved to a provider."""


class MissingDependencyError(OceanError):
    """A required provider SDK is not installed."""

    def __init__(self, provider: str, package: str):
        super().__init__(
            f"{provider} requires '{package}'. "
            f"Install it with: uv pip install -e \".[{provider.lower()}]\""
        )
