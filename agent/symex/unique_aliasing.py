# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from re import compile, match, Pattern
from typing import Optional, Sequence

from agent.symex.scope import ScopeManager
from libcst import (
    Assert,
    Assign,
    Attribute,
    BaseAssignTargetExpression,
    BaseExpression,
    Comparison,
    ComparisonTarget,
    CSTVisitor,
    Equal,
    IndentedBlock,
    Module,
    Name,
)
from libcst.metadata import TypeInferenceProvider
from networkx import ancestors, Graph, has_path

from networkx.classes.reportviews import DiDegreeView


# Itentifies `Unique[...]` Pyre types that signal that an attribute is an id.
_UNIQUE_TYPES: Pattern = compile(r"module\.Unique\[.*\]")


class UniqueAliasing(CSTVisitor):
    """
    Helper to identify aliases of variables by unique properties. if there is
    an assertion in the constraint that asserts that two variables' unique
    properties are the same, we treat the two variables as aliases.
    """

    METADATA_DEPENDENCIES = (TypeInferenceProvider,)

    def __init__(self) -> None:
        super().__init__()
        self.__graph: Graph[str] = Graph()
        self.__program_order_vars: list[str] = []
        self.__program_order_vars_seen: set[str] = set()
        self.__scope_manager = ScopeManager()
        self.aliases: dict[str, str] = {}

    def leave_IndentedBlock(self, original_node: IndentedBlock) -> None:
        self.__scope_manager.end_scope()

    def leave_Module(self, original_node: Module) -> None:
        while self.__graph:
            new_program_order: list[str] = []
            for node in self.__program_order_vars:
                degree: DiDegreeView[str] = self.__graph.degree(node)
                if degree == 0 or degree == 1:
                    for ancestor in ancestors(self.__graph, node):
                        if ancestor not in self.aliases:
                            self.aliases[ancestor] = node
                    self.__graph.remove_node(node)
                else:
                    new_program_order.append(node)
            self.__program_order_vars = new_program_order

    def visit_Assert(self, node: Assert) -> Optional[bool]:
        test: BaseExpression = node.test
        if not isinstance(test, Comparison):
            return
        lhs: BaseExpression = test.left
        if not isinstance(lhs, Attribute):
            return

        comparisons: Sequence[ComparisonTarget] = test.comparisons
        if len(comparisons) > 1:
            return

        comparison: ComparisonTarget = comparisons[0]
        if not isinstance(comparison.operator, Equal):
            return

        rhs: BaseExpression = comparison.comparator
        if not isinstance(rhs, Attribute):
            return
        lhs_name: BaseExpression = lhs.value
        if not isinstance(lhs_name, Name):
            return
        rhs_name: BaseExpression = rhs.value
        if not isinstance(rhs_name, Name):
            return

        lhs_class_type: str = self.get_metadata(TypeInferenceProvider, lhs_name, "")
        rhs_class_type: str = self.get_metadata(TypeInferenceProvider, rhs_name, "")
        if lhs_class_type != rhs_class_type or lhs_class_type == "":
            return
        if lhs.attr.value != rhs.attr.value:
            return

        lhs_type: str = self.get_metadata(TypeInferenceProvider, lhs, "")
        if not UniqueAliasing.is_unique_type(lhs_type):
            return

        lhs_qualified_name: str = self.__scope_manager.get_qualified_name(
            lhs_name.value
        )
        rhs_qualified_name: str = self.__scope_manager.get_qualified_name(
            rhs_name.value
        )
        if (
            not self.__graph.has_node(lhs_qualified_name)
            or not self.__graph.has_node(rhs_qualified_name)
            or not has_path(self.__graph, lhs_qualified_name, rhs_qualified_name)
        ):
            self.__graph.add_edge(lhs_qualified_name, rhs_qualified_name)

    def visit_Assign(self, node: Assign) -> Optional[bool]:
        names_and_values: list[tuple[str, BaseExpression]] = (
            UniqueAliasing.declare_variable(self.__scope_manager, node)
        )
        for var, _ in names_and_values:
            qualified_name: str = self.__scope_manager.get_qualified_name(var)
            if qualified_name not in self.__program_order_vars_seen:
                self.__program_order_vars.append(qualified_name)

    def visit_IndentedBlock(self, node: IndentedBlock) -> Optional[bool]:
        self.__scope_manager.begin_scope()

    @staticmethod
    def declare_variable(
        scope_manager: ScopeManager, node: Assign
    ) -> list[tuple[str, BaseExpression]]:
        """
        Declares a variable in a scope manager. We assume that an assignment to
        a name is a declaration in our constraints.

        Args:
            scope_manager (ScopeManager): Scope manager in which to declare the
            variable.
            node (Assign): Python assignment interpreted as declaration.
        Returns:
            A list of tuples with a local variable name and its assigned value.
        """
        names_and_values: list[tuple[str, BaseExpression]] = []
        for target in node.targets:
            name: BaseAssignTargetExpression = target.target
            if isinstance(name, Name):
                var: str = name.value
                scope_manager.declare_variable(var)
                names_and_values.append((var, node.value))
        return names_and_values

    @staticmethod
    def is_unique_type(pyre_type: str) -> bool:
        """
        Checks whether a Type string provided by Pyre is a Logic.py
        `Unique[...]` type wrapper.

        Args:
            pyre_type (str): Type information by Pyre to clasify.
        Returns:
            `True` iff `pyre_type` is a `Unique[...]` type wrapper.
        """
        return bool(match(_UNIQUE_TYPES, pyre_type))
