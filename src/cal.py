"""四则运算.

该程序提供了一些基本的数学函数，包括加减乘除四则运算。

Example:
    >>> add(1, 2)
    3.0
    >>> subtract(1, 2)
    -1.0
    >>> multiply(1, 2)
    2.0
    >>> divide(1, 2)
    0.5
"""

def add(a, b) -> float:
    """两数相加.

    Args:
        a (T): 第一个数
        b (T): 第二个数

    Returns:
        float: 两个数的和
    """
    return float(a + b)


def subtract(a, b) -> float:
    """两数相减.

    Args:
        a (T): 第一个数
        b (T): 第二个数

    Returns:
        float: 两个数的差
    """
    return float(a - b)


def multiply(a, b) -> float:
    """两数相乘.

    Args:
        a (int): 第一个数
        b (int): 第二个数

    Returns:
        float: 两个数的积
    """
    return float(a * b)


def divide(a, b) -> float:
    """两数相除.

    Args:
        a (T): 第一个数
        b (T): 第二个数

    Raises:
        ZeroDivisionError: 除数为0

    Returns:
        float: 两个数的商
    """
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return float(a / b)


