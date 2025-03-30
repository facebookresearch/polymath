from libcst import BaseExpression, Name

# "Name" of boolean literal `False` used in libCST.
FALSE_NAME: str = "False"

# Python expression representing `False`.
FALSE: BaseExpression = Name(FALSE_NAME)

# "Name" of boolean literal `True` used in libCST.
TRUE_NAME: str = "True"

# Python expression representing `True`. This is the default for branch guards.
TRUE: BaseExpression = Name(TRUE_NAME)
