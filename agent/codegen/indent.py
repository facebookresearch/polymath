# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Indentation sequence used in generated SMT code.
_INDENT: str = "  "

# Length of _INDENT. This is used when dedenting.
_INDENT_SIZE: int = len(_INDENT)


class Indent:
    """
    Indentation helper when generating code.
    """

    def __init__(self) -> None:
        self.__indent_prefix: str = ""

    def dedent(self) -> None:
        """
        Lowers the current indentation level.
        """
        self.__indent_prefix = self.__indent_prefix[:-_INDENT_SIZE]

    def indent(self) -> None:
        """
        Increases the current indentation level.
        """
        self.__indent_prefix += _INDENT

    def prepend(self, code: str) -> str:
        """
        Prepends current indentaiton level to code snippet.
        """
        return self.__indent_prefix + code
