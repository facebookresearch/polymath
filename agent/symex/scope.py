# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any


# Separator between Python blocks when translating to SMT. This is to
# disambiguate same variable names in different scopes.
_SCOPE_SEP: str = "__"


class _Scope:
    """
    Represents a single scope or block in a stack of scopes managed by the
    ScopeManager.
    """

    def __init__(self, id: int, prefix: str) -> None:
        self.id: int = id
        self.prefix: str = prefix
        self.declared_variables: set[str] = set()
        self.context: dict[str, Any] = {}


class ScopeManager:
    """
    Helper to disambiguate same variable names in different scopes during
    symex.
    """

    def __init__(self) -> None:
        self.__scopes: list[_Scope] = [_Scope(0, "")]
        self.__next_scope_id: int = 0
        self.__current_scope_prefix: str = _SCOPE_SEP

    def begin_scope(self) -> None:
        """
        Starts a new scope, usually invoked when a new block is seen in the
        code.
        """
        self.__current_scope_prefix += f"{self.__next_scope_id}{_SCOPE_SEP}"
        self.__scopes.append(_Scope(self.__next_scope_id, self.__current_scope_prefix))
        self.__next_scope_id = 0

    def end_scope(self) -> None:
        """
        Ends scope, usually invoked when a new block ends in the code.
        """
        ended_scope: _Scope = self.__scopes.pop()
        self.__next_scope_id = ended_scope.id + 1
        self.__current_scope_prefix = self.__scopes[-1].prefix if self.__scopes else ""

    def declare_variable(self, name: str) -> None:
        """
        Invoked when a variable declaration is seen. Current scope will be
        taken into consideration.
        """
        self.__scopes[-1].declared_variables.add(name)

    def get_qualified_name(self, name: str) -> str:
        """
        Query fully qualified, scoped name for a name seen in the code. Current
        scope will be taken into consideration.
        """
        for scope in reversed(self.__scopes):
            if name in scope.declared_variables:
                return scope.prefix + name

        return name

    def get_context_value(self, name: str) -> Any:
        """
        Retrieves a custom user value associated with the current scope.
        """
        return self.__scopes[-1].context.get(name)

    def set_context_value(self, name: str, value: Any) -> None:
        """
        Sets a custom user value associated with the current scope.
        """
        self.__scopes[-1].context[name] = value

    @staticmethod
    def to_unqualified_name(qualified_name: str) -> str:
        """
        Converts qualified names to local names that can be used in CST nodes.

        Args:
            qualified_name (str): Scoped qualified name.
        Returns:
            Original CST node name.
        """
        offset: int = qualified_name.rfind(_SCOPE_SEP)
        return qualified_name if offset == -1 else qualified_name[offset:]
