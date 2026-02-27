"""Tests for smart retry filtering logic."""

from __future__ import annotations

from adele_runner.utils.retry import _get_status_code, _is_connection_or_timeout, is_retryable

# ---------------------------------------------------------------------------
# Non-retryable logic errors
# ---------------------------------------------------------------------------


def test_value_error_not_retryable():
    assert is_retryable(ValueError("bad value")) is False


def test_type_error_not_retryable():
    assert is_retryable(TypeError("bad type")) is False


def test_key_error_not_retryable():
    assert is_retryable(KeyError("missing")) is False


def test_attribute_error_not_retryable():
    assert is_retryable(AttributeError("no attr")) is False


# ---------------------------------------------------------------------------
# Generic exceptions → retryable (fail open)
# ---------------------------------------------------------------------------


def test_generic_exception_retryable():
    assert is_retryable(Exception("something went wrong")) is True


def test_runtime_error_retryable():
    assert is_retryable(RuntimeError("unexpected")) is True


# ---------------------------------------------------------------------------
# Connection / timeout (string heuristic)
# ---------------------------------------------------------------------------


class FakeTimeoutError(Exception):
    pass


class FakeConnectionError(Exception):
    pass


def test_timeout_in_type_name():
    assert _is_connection_or_timeout(FakeTimeoutError("")) is True


def test_connection_in_type_name():
    assert _is_connection_or_timeout(FakeConnectionError("")) is True


def test_timeout_in_message():
    assert _is_connection_or_timeout(Exception("Request timed out after 30s")) is True


def test_connection_reset_in_message():
    assert _is_connection_or_timeout(Exception("Connection reset by peer")) is True


def test_normal_exception_not_connection():
    assert _is_connection_or_timeout(Exception("Something else")) is False


# ---------------------------------------------------------------------------
# HTTP status codes via custom exception
# ---------------------------------------------------------------------------


class FakeHTTPError(Exception):
    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


def test_429_retryable():
    assert is_retryable(FakeHTTPError(429)) is True


def test_500_retryable():
    assert is_retryable(FakeHTTPError(500)) is True


def test_502_retryable():
    assert is_retryable(FakeHTTPError(502)) is True


def test_503_retryable():
    assert is_retryable(FakeHTTPError(503)) is True


def test_504_retryable():
    assert is_retryable(FakeHTTPError(504)) is True


def test_400_not_retryable():
    assert is_retryable(FakeHTTPError(400)) is False


def test_401_not_retryable():
    assert is_retryable(FakeHTTPError(401)) is False


def test_403_not_retryable():
    assert is_retryable(FakeHTTPError(403)) is False


def test_404_not_retryable():
    assert is_retryable(FakeHTTPError(404)) is False


def test_422_not_retryable():
    assert is_retryable(FakeHTTPError(422)) is False


# ---------------------------------------------------------------------------
# _get_status_code
# ---------------------------------------------------------------------------


def test_get_status_code_from_status_code_attr():
    assert _get_status_code(FakeHTTPError(429)) == 429


def test_get_status_code_none_for_generic():
    assert _get_status_code(Exception("no status")) is None


class FakeWithCode(Exception):
    def __init__(self, code: int):
        self.code = code
        super().__init__()


def test_get_status_code_from_code_attr():
    assert _get_status_code(FakeWithCode(503)) == 503
