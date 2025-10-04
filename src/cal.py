"""谷歌风格注释.

本模块展示了如何按照《Google Python 风格指南》_进行文档编写。文档字符串可以跨越多行。节标题后跟一个冒号，然后是一段缩进的文本，即可创建一个节。

Example:
    可以使用“Example”或“Examples”部分来提供示例。这些部分支持任何reStructuredText格式，包括字面块::

        $ python example_google.py

节断点是通过恢复未缩进的文本创建的。每当新节开始时，也会隐式地创建节断点。

Attributes:
    模块级变量1（整型）：模块级变量可以记录在模块文档字符串的“属性”部分，也可以记录在变量后的内联文档字符串中。
    这两种形式都是可接受的，但不应混用。选择一种约定来记录模块级变量，并保持一致性。

Todo:
    - 对于模块中的待办事项
    - 你还必须使用“sphinx.ext.todo”扩展
"""

module_level_variable1 = 12345

module_level_variable2 = 98765
"""int: 模块级变量已内联注释。

文档字符串可以跨越多行。类型可以可选地在第一行指定，用冒号分隔。
"""


def function_with_types_in_docstring(param1, param2):
    """示例函数，其类型已在文档字符串中进行了说明.

    :pep:`484` type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.
    """


def function_with_pep484_type_annotations(param1: int, param2: str) -> bool:
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """


def module_level_function(param1, param2=None, *args, **kwargs):
    """This is an example of a module level function.

    Function parameters should be documented in the ``Args`` section. The name
    of each parameter is required. The type and description of each parameter
    is optional, but should be included if not obvious.

    If ``*args`` or ``**kwargs`` are accepted,
    they should be listed as ``*args`` and ``**kwargs``.

    The format for a parameter is::

        name (type): description
            The description may span multiple lines. Following
            lines should be indented. The "(type)" is optional.

            Multiple paragraphs are supported in parameter
            descriptions.

    Args:
        param1 (int): The first parameter.
        param2 (:obj:`str`, optional): The second parameter. Defaults to None.
            Second line of description should be indented.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        bool: True if successful, False otherwise.

        The return type is optional and may be specified at the beginning of
        the ``Returns`` section followed by a colon.

        The ``Returns`` section may span multiple lines and paragraphs.
        Following lines should be indented to match the first line.

        The ``Returns`` section supports any reStructuredText formatting,
        including literal blocks::

            {
                'param1': param1,
                'param2': param2,
            }

    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions
            that are relevant to the interface.
        ValueError: If `param2` is equal to `param1`.

    """
    if param1 == param2:
        msg = 'param1 may not be equal to param2'
        raise ValueError(msg)
    return True


def example_generator(n):
    """Generators have a ``Yields`` section instead of a ``Returns`` section.

    Args:
        n (int): The upper limit of the range to generate, from 0 to `n` - 1.

    Yields:
        int: The next number in the range of 0 to `n` - 1.

    Examples:
        示例应采用文档测试（doctest）格式编写，并应说明如何使用该函数。

        >>> print([i for i in example_generator(4)])
        [0, 1, 2, 3]

    """
    yield from range(n)


class ExampleError(Exception):
    """异常的文档记录方式与类相同。

    __init__方法可以在类级别的文档字符串中记录，也可以作为__init__方法本身的文档字符串。

    两种形式都是可以接受的，但不应混用。选择一种约定来记录__init__方法，并保持一致。

    Note:
        不要在``Args``部分包含`self`参数。

    Args:
        msg (str): 描述异常的可读字符串。
        code (:obj:`int`, 可选): 错误代码。

    Attributes:
        msg (str)：描述异常的可读字符串。
        code (int)：异常错误代码。
    """

    def __init__(self, msg, code):
        self.msg = msg
        self.code = code


class ExampleClass:
    """类文档字符串的摘要行应仅占一行。

    如果类有公共属性，可以在“属性”部分进行文档说明，并遵循与函数“参数”部分相同的格式。或者，可以在属性声明处直接进行文档说明（见下文的__init__方法）。

    使用“@property”装饰器创建的属性应在该属性的getter方法中加以说明。

    Attributes:
        attr1 (str): `attr1`的描述。
        attr2 (:obj:`int`, 可选): `attr2`的描述。

    """

    def __init__(self, param1, param2, param3):
        """__init__方法的文档字符串示例。

        __init__方法的文档可以放在类级别的文档字符串中，也可以作为__init__方法本身的文档字符串。

        两种形式都是可以接受的，但不应混用。选择一种约定来记录__init__方法，并保持一致性。

        Note:
            在“Args”部分中不要包含“self”参数。

        Args:
            param1 (str): `param1`的描述。
            param2（:obj:`int`，可选）：`param2`的描述。支持多行描述。
            param3 (list(str)): `param3`的描述。
        """
        self.attr1 = param1
        self.attr2 = param2
        self.attr3 = param3  #: Doc comment *inline* with attribute

        #: list(str): Doc comment *before* attribute, with type specified
        self.attr4 = ['attr4']

        self.attr5 = None
        """str: 带有指定类型的*属性后*文档字符串。"""

    @property
    def readonly_property(self):
        """str: 属性应在其getter方法中记录。"""
        return 'readonly_property'

    @property
    def readwrite_property(self):
        """list(str): 同时具有getter和setter方法的属性
        应该只记录在它们的getter方法中。

        如果setter方法包含值得注意的行为，则应在此处提及。
        """
        return ['readwrite_property']

    @readwrite_property.setter
    def readwrite_property(self, value):
        _ = value

    def example_method(self, param1, param2):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        return True

    def __special__(self):
        """默认情况下，带有文档字符串的特殊成员不会被包含在内。

        特殊成员是指任何以双下划线开头和结尾的方法或属性。如果将``napoleon_include_special_with_doc``设置为True，则任何带有文档字符串的特殊成员都将包含在输出中。

        可以通过更改Sphinx的conf.py中的以下设置来启用此行为：

            napoleon_include_special_with_doc = True

        """
        pass

    def __special_without_docstring__(self):
        pass

    def _private(self):
        """By default private members are not included.

        Private members are any methods or attributes that start with an
        underscore and are *not* special. By default they are not included
        in the output.

        This behavior can be changed such that private members *are* included
        by changing the following setting in Sphinx's conf.py::

            napoleon_include_private_with_doc = True

        """
        pass

    def _private_without_docstring(self):
        pass


class ExamplePEP526Class:
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. If ``napoleon_attr_annotations``
    is True, types can be specified in the class body using ``PEP 526``
    annotations.

    Attributes:
        attr1: Description of `attr1`.
        attr2: Description of `attr2`.

    """

    attr1: str
    attr2: int