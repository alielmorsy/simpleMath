import math


def pow_int_int(a: int, b: int) -> int:
    if b < 0:
        if a == 0:
            raise ZeroDivisionError("0 cannot be raised to a negative power")
        # Return float because result may not be integer
        return 1 / pow_int_int(a, -b)

    result = 1
    base = a
    exp = b
    while exp > 0:
        if exp & 1:  # odd
            result *= base
        base *= base
        exp >>= 1
    return result


def _power_integer(base: float, exp: int) -> float:
    """
    Helper function to calculate power for non-negative integer exponents.
    Uses the efficient "exponentiation by squaring" algorithm.
    """
    result = 1.0
    current_base = base

    while exp > 0:
        # If exponent is odd, multiply the base with the result
        if exp % 2 == 1:
            result *= current_base

        # Square the base and halve the exponent
        current_base *= current_base
        exp //= 2

    return result


def pow_int_float(base: int, exponent: float) -> float:
    if base == 1 or exponent == 0:
        return 1.0

    if base == 0:
        if exponent > 0:
            return 0.0
        else:
            # 0 to a negative or zero power is undefined
            raise ValueError("0 cannot be raised to a negative or zero power")

    if base < 0 and abs(exponent - int(exponent)) > 1e-7:
        # This implementation does not support complex numbers, which would be
        # the result of a negative base with a non-integer exponent.
        raise ValueError("Real result for negative base and non-integer exponent is undefined")

    # --- 2. Separate Integer and Fractional Parts of the Exponent ---
    is_negative_exponent = exponent < 0
    if is_negative_exponent:
        exponent = -exponent

    integer_part = int(exponent)
    fractional_part = exponent - integer_part

    # --- 3. Calculate Power for Each Part ---

    # For the integer part, use the efficient integer exponentiation
    integer_result = _power_integer(float(base), integer_part)

    # For the fractional part, use the identity: base^x = exp(x * log(base))
    fractional_result = 1.0
    if fractional_part > 1e-7:
        log_of_base = math.log(float(base))
        fractional_result = math.exp(fractional_part * log_of_base)

    # --- 4. Combine and Finalize ---
    total_result = integer_result * fractional_result

    if is_negative_exponent:
        return 1.0 / total_result
    else:
        return total_result


def pow_float_float(x: float, y: float) -> float:
    ln2 = math.log(2, math.e)
    # e^x= e^(q*ln(2) + r)
    # x =q*ln(2) + r
    q = int(x // ln2)  # q= w / ln(2)

    r = x - q * ln2

    n = 5
    er = sum((x ** k) / math.factorial(k) for k in range(n + 1))


if __name__ == "__main__":
    print(pow_int_int(2, 3))
    print(pow_int_float(2, 3))
    print(pow_float_float(2, 2.1 + 0.9))
