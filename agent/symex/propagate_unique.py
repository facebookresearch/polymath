# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence, Union

from agent.symex.collect_unique_ids import SOME
from agent.symex.scope import ScopeManager

from agent.symex.unique_aliasing import UniqueAliasing

from libcst import (
    Assign,
    AssignTarget,
    BaseAssignTargetExpression,
    BaseExpression,
    BaseSuite,
    Call,
    CSTTransformer,
    IndentedBlock,
    Name,
    RemovalSentinel,
    RemoveFromParent,
    Subscript,
)


# TODO: Replace aliases
# TODO: Replace unique property comparison by identity comparison
# TODO: Test Subscript translation in Logic.py SMT constraint generator
class PropagateUnique(CSTTransformer):

    def __init__(
        self, aliases: dict[str, str], replacements: dict[str, Subscript]
    ) -> None:
        super().__init__()
        self.__aliases: dict[str, str] = aliases
        self.__replacements: dict[str, Subscript] = replacements
        self.__scope_manager = ScopeManager()

    def leave_Assign(
        self, original_node: Assign, updated_node: Assign
    ) -> Union[Assign, RemovalSentinel]:
        rhs: BaseExpression = original_node.value
        if not isinstance(rhs, Call):
            return updated_node
        func: BaseExpression = rhs.func
        if not isinstance(func, Name) or func.value != SOME:
            return updated_node

        targets: Sequence[AssignTarget] = original_node.targets
        if len(targets) != 1:
            return updated_node

        target: BaseAssignTargetExpression = targets[0].target
        if not isinstance(target, Name):
            return updated_node
        original_name: str = self.__scope_manager.get_qualified_name(target.value)
        name: str = self.__aliases.get(original_name, original_name)
        return RemoveFromParent() if name in self.__replacements else updated_node

    def leave_IndentedBlock(
        self, original_node: IndentedBlock, updated_node: IndentedBlock
    ) -> BaseSuite:
        self.__scope_manager.end_scope()
        return updated_node

    def leave_Name(self, original_node: Name, updated_node: Name) -> BaseExpression:
        name: str = self.__scope_manager.get_qualified_name(original_node.value)
        aliased_name: str = self.__aliases.get(name, name)
        expr: Optional[Subscript] = self.__replacements.get(aliased_name)
        return expr or updated_node.with_changes(
            value=ScopeManager.to_unqualified_name(aliased_name)
        )

    def visit_Assign(self, node: Assign) -> Optional[bool]:
        UniqueAliasing.declare_variable(self.__scope_manager, node)

    def visit_IndentedBlock(self, node: IndentedBlock) -> Optional[bool]:
        self.__scope_manager.begin_scope()
