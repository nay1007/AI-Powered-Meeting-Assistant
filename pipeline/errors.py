"""Errors that are safe to show to the user as-is (no stack traces)."""


class PipelineError(Exception):
    """A problem with the input or a pipeline stage, phrased for end users."""
