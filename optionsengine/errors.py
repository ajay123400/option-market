"""Exception types for the options engine. All derive from ValueError so
callers can catch the whole family with `except ValueError` if they wish."""


class OptionsEngineError(ValueError):
    """Base class for every error raised deliberately by this package."""


class InvalidInputError(OptionsEngineError):
    """An input is non-finite, out of domain (e.g. S <= 0, sigma < 0) or of
    the wrong kind. The caller passed something that can never be priced."""


class DegenerateInputError(OptionsEngineError):
    """Inputs are individually valid but the requested quantity is
    undefined or singular there (Greeks at T == 0 or sigma == 0). The
    function refuses rather than returning a misleading finite number."""
