# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

from libcst import matchers as m
from libcst._nodes.expression import (
    Annotation,
    Attribute,
    BaseAssignTargetExpression,
    BaseExpression,
    BinaryOperation,
    BooleanOperation,
    Call,
    Comparison,
    Float,
    Integer,
    Name,
    SimpleString,
    Subscript,
)
from libcst._nodes.module import Module
from libcst._nodes.statement import (
    Assert,
    Assign,
    Else,
    FunctionDef,
    If,
    SimpleStatementLine,
)


class LogicPyCConstraintGenerator(m.MatcherDecoratableVisitor):
    """
    Converts LogicPy constraint methods to equivalent C harness constraints.
    """

    def __init__(self, module: Module) -> None:
        super().__init__()
        self.module = module
        self.local_variable_deduplication: dict[str, int] = {}
        self.solution_type: str = "void"
        self.is_inside_validate: bool = False
        self.c_constraints = ""
        self.forbidden_nouns = ["long", "short", "double"]

    def visit_Else(self, node: Else) -> Optional[bool]:
        self.c_constraints += "{\n"
        node.body.visit(self)
        self.c_constraints += "    }\n"
        return False

    def visit_FunctionDef(self, node: FunctionDef) -> Optional[bool]:
        self.is_inside_validate = True
        self.c_constraints += f"static void {node.name.value}("
        for arg in node.params.params:
            annotation: Optional[Annotation] = arg.annotation
            if annotation is not None:
                type_expression: BaseExpression = annotation.annotation
                if isinstance(type_expression, Name):
                    self.solution_type = type_expression.value
                    self.c_constraints += f"struct {self.solution_type} "
            self.c_constraints += arg.name.value
        self.c_constraints += ") {\n"

        node.body.visit(self)
        self.c_constraints += "}\n"
        return False

    def visit_If(self, node: If) -> Optional[bool]:
        self.c_constraints += "    if ("
        node.test.visit(self)
        self.c_constraints += ") {\n"
        node.body.visit(self)
        self.c_constraints += "    }\n"
        orelse: If | Else | None = node.orelse
        if orelse is not None:
            self.c_constraints += "    else "
            orelse.visit(self)
        return False

    def visit_Call(self, node: Call) -> Optional[bool]:
        if self.is_inside_validate:
            func: BaseExpression = node.func
            value, name = LogicPyCConstraintGenerator.get_member_call(func)
            is_first_arg: bool
            if value is not None and name is not None and name.value == "index":
                self.c_constraints += "__CPROVER_index("
                value.visit(self)
                is_first_arg = False
            elif (
                isinstance(func, Name)
                and func.value == "__polymath_constraint_separator"
            ):
                return False
            else:
                func.visit(self)
                self.c_constraints += "("
                is_first_arg = True

            for arg in node.args:
                if is_first_arg:
                    is_first_arg = False
                else:
                    self.c_constraints += ", "
                arg.visit(self)

            self.c_constraints += ")"
            return False

    def visit_Attribute(self, node: Attribute) -> Optional[bool]:
        node.value.visit(self)
        self.c_constraints += "."
        node.attr.visit(self)
        return False

    def visit_Integer(self, node: Integer) -> Optional[bool]:
        if self.is_inside_validate:
            self.c_constraints += self.module.code_for_node(node)

    def visit_Float(self, node: Float) -> Optional[bool]:
        if self.is_inside_validate:
            self.c_constraints += self.module.code_for_node(node)

    def visit_BooleanOperation(self, node: BooleanOperation) -> Optional[bool]:
        node.left.visit(self)

        operator: str = self.module.code_for_node(node.operator)
        if operator == " and ":
            self.c_constraints += " && "
        elif operator == " or ":
            self.c_constraints += " || "
        else:
            self.c_constraints += operator

        node.right.visit(self)
        return False

    def visit_SimpleString(self, node: SimpleString) -> Optional[bool]:
        if self.is_inside_validate:
            self.c_constraints += self.module.code_for_node(node)

    def visit_Assign(self, node: Assign) -> Optional[bool]:
        if self.is_inside_validate:
            self.c_constraints += "typeof("
            node.value.visit(self)
            self.c_constraints += ") "

            for target in node.targets:
                expression: BaseAssignTargetExpression = target.target
                if isinstance(expression, Name):
                    name: str = expression.value
                    duplicate_count: Optional[int] = (
                        self.local_variable_deduplication.get(name)
                    )
                    if duplicate_count is None:
                        duplicate_count = 0
                    else:
                        duplicate_count += 1
                    self.local_variable_deduplication[name] = duplicate_count

                target.visit(self)
                self.c_constraints += " = "
            node.value.visit(self)
            return False

    def visit_BinaryOperation(self, node: BinaryOperation) -> Optional[bool]:
        if self.is_inside_validate:
            node.left.visit(self)
            self.c_constraints += self.module.code_for_node(node.operator)
            node.right.visit(self)
            return False

    def visit_Comparison(self, node: Comparison) -> Optional[bool]:
        if self.is_inside_validate:
            node.left.visit(self)
            for comparison in node.comparisons:
                self.c_constraints += self.module.code_for_node(comparison.operator)
                comparison.comparator.visit(self)
            return False

    def visit_Name(self, node: Name) -> Optional[bool]:
        if self.is_inside_validate:
            value: str = node.value
            duplicate_count: Optional[int] = self.local_variable_deduplication.get(
                value
            )
            if duplicate_count is not None and duplicate_count > 0:
                value += f"_{duplicate_count}"
            elif value in self.forbidden_nouns:
                value += "_"
            elif value == "abs":
                value = "__CPROVER_abs"
            elif value == "assume":
                value = "__CPROVER_assume"
            elif value == "nondet":
                value = "__CPROVER_nondet_element"
            elif value == "False":
                value = "false"
            elif value == "True":
                value = "true"

            self.c_constraints += value

    def visit_Assert(self, node: Assert) -> Optional[bool]:
        self.c_constraints += "__CPROVER_assume("

    def leave_Assert(self, original_node: Assert) -> None:
        self.c_constraints += ")"

    def visit_SimpleStatementLine(self, node: SimpleStatementLine) -> Optional[bool]:
        if self.is_inside_validate:
            self.c_constraints += "    "

    def leave_SimpleStatementLine(self, original_node: SimpleStatementLine) -> None:
        if self.is_inside_validate:
            self.c_constraints += ";\n"

    def visit_Subscript(self, node: Subscript) -> Optional[bool]:
        if self.is_inside_validate:
            node.value.visit(self)
            self.c_constraints += "["
            is_first: bool = True
            for element in node.slice:
                if is_first:
                    is_first = False
                else:
                    self.c_constraints += ", "

                element.visit(self)
            self.c_constraints += "]"
            return False

    def leave_Module(self, original_node: Module) -> None:
        self.c_constraints += f"""
#ifndef __CPROVER
void __CPROVER_output(const char *name, struct {self.solution_type} solution);
#endif

int main(void) {{
    struct {self.solution_type} solution;
    init_{self.solution_type}(&solution);
    validate(solution);

    __CPROVER_output("solution", solution);
    __CPROVER_assert(false, "");
}}
"""

    @staticmethod
    def get_member_call(
        expression: BaseExpression,
    ) -> Tuple[Optional[BaseExpression], Optional[Name]]:
        """
        Helper to identify and split calls to member functions. E.g. the `func`
        expression `object.mylist.index` is split into `object.mylist` and
        `index`. This allows us to replace member functions by CBMC internals
        if necessary, e.g. in the case of `list.index(element)` by
        `__CPROVER_Index(list, element)`.
        """
        if isinstance(expression, Attribute):
            return expression.value, expression.attr
        return None, None
