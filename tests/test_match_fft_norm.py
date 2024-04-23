import pytest
from pyfar.classes.audio import _match_fft_norm


# handling invalid FFT norms
def test_input_fft_norm_valid():
    with pytest.warns(
        UserWarning, match="Invalid FFT norm provided. Defaulting to 'none'."
    ):
        assert _match_fft_norm("invalid", "none", "multiply") == "none"
        assert _match_fft_norm("none", "invalid", "multiply") == "none"


# matching FFT norms with different operations
@pytest.mark.parametrize(
    "fft_norm1, fft_norm2, operation, expected",
    [
        ("none", "none", "multiply", "none"),
        ("unitary", "unitary", "multiply", "unitary"),
        ("amplitude", "amplitude", "multiply", "amplitude"),
        ("rms", "rms", "multiply", "rms"),
        ("power", "power", "multiply", "power"),
        ("psd", "psd", "multiply", "psd"),
        ("none", "unitary", "multiply", "none"),
        ("unitary", "none", "multiply", "none"),
        ("amplitude", "none", "multiply", "none"),
        ("none", "amplitude", "multiply", "none"),
        ("unitary", "unitary", "divide", "unitary"),
        ("amplitude", "amplitude", "divide", "amplitude"),
        ("rms", "none", "divide", "none"),
        ("none", "rms", "divide", "none"),
    ],
)
def test_match_fft_norm(fft_norm1, fft_norm2, operation, expected):
    if fft_norm1 != fft_norm2 and operation != "divide":
        with pytest.warns(
            UserWarning, match="Mismatched norms without override"
        ):
            assert _match_fft_norm(fft_norm1, fft_norm2, operation) == expected
    else:
        assert _match_fft_norm(fft_norm1, fft_norm2, operation) == expected


# warning when dividing by 'none'
def test_division_warning():
    with pytest.warns(
        UserWarning,
        match="Division involving 'none' may lead to unintended results.",
    ):
        assert _match_fft_norm("none", "none", "divide") == "none"


# arithmetic operations without division
@pytest.mark.parametrize(
    "fft_norm1, fft_norm2, expected",
    [
        ("none", "none", "none"),
        ("unitary", "unitary", "unitary"),
        ("amplitude", "amplitude", "amplitude"),
        ("rms", "rms", "rms"),
        ("power", "power", "power"),
        ("psd", "psd", "psd"),
        ("none", "unitary", "none"),
        ("unitary", "none", "none"),
        ("amplitude", "none", "none"),
        ("none", "amplitude", "none"),
    ],
)
def test_result_no_division(fft_norm1, fft_norm2, expected):
    assert _match_fft_norm(fft_norm1, fft_norm2, "multiply") == expected


# division with mismatched FFT norms
@pytest.mark.parametrize(
    "fft_norm1, fft_norm2, operation, expected",
    [
        ["unitary", "amplitude", "divide", "none"],
        ["unitary", "rms", "divide", "none"],
        ["unitary", "power", "divide", "none"],
        ["unitary", "psd", "divide", "none"],
        ["amplitude", "unitary", "divide", "none"],
        ["amplitude", "rms", "divide", "none"],
        ["amplitude", "power", "divide", "none"],
        ["amplitude", "psd", "divide", "none"],
        ["rms", "unitary", "divide", "none"],
        ["rms", "amplitude", "divide", "none"],
        ["rms", "power", "divide", "none"],
        ["rms", "psd", "divide", "none"],
        ["power", "unitary", "divide", "none"],
        ["power", "amplitude", "divide", "none"],
        ["power", "rms", "divide", "none"],
        ["power", "psd", "divide", "none"],
        ["psd", "unitary", "divide", "none"],
        ["psd", "amplitude", "divide", "none"],
        ["psd", "rms", "divide", "none"],
        ["psd", "power", "divide", "none"],
    ],
)
def test_assertion_no_division_with_warnings(
    fft_norm1, fft_norm2, operation, expected
):
    with pytest.warns(UserWarning, match="Mismatched norms without override"):
        assert _match_fft_norm(fft_norm1, fft_norm2, operation) == expected


# division with 'none'
@pytest.mark.parametrize(
    "fft_norm1, fft_norm2, operation, expected",
    [
        ("unitary", "none", "divide", "none"),
        ("amplitude", "none", "divide", "none"),
        ("rms", "none", "divide", "none"),
        ("power", "none", "divide", "none"),
        ("psd", "none", "divide", "none"),
    ],
)
def test_result_division(fft_norm1, fft_norm2, operation, expected):
    if fft_norm1 != fft_norm2:
        with pytest.warns(
            UserWarning,
            match="Division involving 'none' may lead to unintended results.",
        ):
            assert _match_fft_norm(fft_norm1, fft_norm2, operation) == expected
    else:
        assert _match_fft_norm(fft_norm1, fft_norm2, operation) == expected
