# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque
from re import compile, Match, match, Pattern
from typing import Optional, Union

from agent.codegen.indent import _INDENT, Indent
from agent.logic.logic_py_smt_data_structure_generator import _UNIVERSE_CLASS_NAME

from agent.symex.boolean import FALSE_NAME, TRUE, TRUE_NAME
from agent.symex.collect_unique_ids import SOME
from agent.symex.scope import ScopeManager

from agent.symex.unique import UNIQUE

from libcst import (
    And,
    AnnAssign,
    Annotation,
    Arg,
    Assert,
    Assign,
    AssignTarget,
    Attribute,
    BaseAssignTargetExpression,
    BaseBooleanOp,
    BaseCompOp,
    BaseExpression,
    BaseSlice,
    BaseUnaryOp,
    BooleanOperation,
    Call,
    ClassDef,
    Comparison,
    ComparisonTarget,
    CSTVisitor,
    Else,
    Equal,
    Expr,
    Float,
    For,
    FunctionDef,
    GreaterThan,
    GreaterThanEqual,
    If,
    IfExp,
    In,
    IndentedBlock,
    Index,
    Integer,
    LessThan,
    LessThanEqual,
    Module,
    Name,
    Not,
    NotEqual,
    NotIn,
    Or,
    Pass,
    SimpleStatementLine,
    SimpleString,
    Subscript,
    SubscriptElement,
    UnaryOperation,
)

from libcst.metadata.type_inference_provider import TypeInferenceProvider


# Helper to extract element types due to incomplete Pyre type info.
_ELEMENT_TYPE: Pattern = compile(r"^typing\.List\[(.*)\]$")

# Name of the conclusion function in conclusion check.
_CONCLUSION: str = "conclusion"

# Name of scope-level user variable keeping track of SMT exist expressions to
# close.
_EXISTS_STACK_COUNT: str = "exists-stack-count"

# Suffix to use for quantified list indexes.
_INDEX_SUFFIX: str = "_index"

# Name of the premise function in conclusion check.
_PREMISE: str = "premise"


class LogicPySMTConstraintGenerator(CSTVisitor):
    """
    Converts LogicPy data structures to a equivalent SMT functions.
    """

    METADATA_DEPENDENCIES = (TypeInferenceProvider,)

    def __init__(
        self, ancestors_per_class: dict[str, set[str]], smt_attributes: set[str]
    ) -> None:
        self.smt_harness: str = ""

        self.__premise: str = ""
        self.__conclusion: str = ""
        self.__negated_conclusion: str = ""

        self.__ancestors_per_class: dict[str, set[str]] = ancestors_per_class
        self.__body: str = ""
        self.__negated_body_template: str = ""
        self.__branch_guard: BaseExpression = TRUE
        self.__indent = Indent()
        self.__in_tmp_counter: int = 0
        self.__is_in_function: bool = False
        self.__smt_attributes: set[str] = smt_attributes
        self.__scope_manager = ScopeManager()
        self.__universe_variable_name: str = ""

    def visit_AnnAssign(self, node: AnnAssign) -> Optional[bool]:
        value: Optional[BaseExpression] = node.value
        if value:
            self.visit_Assign(Assign([AssignTarget(node.target)], value))
        return False

    def visit_Assert(self, node: Assert) -> Optional[bool]:
        if self.__branch_guard != TRUE:
            self.__append_line("(=>", False)
            self.__indent.indent()
            self.__append("")
            self.__branch_guard.visit(self)
            self.__append_newline()
            self.__append("")
            node.test.visit(self)
            self.__append_newline()
            self.__indent.dedent()
            self.__append(")")
        else:
            node.test.visit(self)
        return False

    def visit_Assign(self, node: Assign) -> Optional[bool]:
        value: BaseExpression = node.value
        if isinstance(value, Call):
            func: BaseExpression = value.func
            if isinstance(func, Name):
                name: str = func.value
                self.__scope_manager.declare_variable(name)
                if name == SOME:
                    self.__scope_manager.declare_variable(f"{name}{_INDEX_SUFFIX}")
                    sequence: BaseExpression = value.args[0].value

                    target_names: list[str] = []
                    for target in node.targets:
                        target_expr: BaseAssignTargetExpression = target.target
                        if isinstance(target_expr, Name):
                            name: str = target_expr.value
                            target_names.append(name)

                    self.__increment_exists_stack_counter()
                    self.__append_line("(exists", False)
                    self.__indent.indent()
                    self.__append("(")
                    is_first: bool = True
                    for target_name in target_names:
                        self.__scope_manager.declare_variable(name)
                        self.__scope_manager.declare_variable(f"{name}{_INDEX_SUFFIX}")
                        if is_first:
                            self.__append("(", False)
                        else:
                            self.__append(" (", False)

                        qualified_name: str = self.__scope_manager.get_qualified_name(
                            target_name
                        )
                        self.__append(f"{qualified_name}{_INDEX_SUFFIX} Int)", False)

                    self.__append_line(")", False)
                    self.__append("")

                    self.__increment_exists_stack_counter(2)
                    self.__append_line("(let", False)
                    self.__indent.indent()
                    self.__append("(")
                    is_first: bool = True
                    for target in node.targets:
                        if is_first:
                            self.__append("(", False)
                        else:
                            self.__append(" (", False)

                        target.visit(self)
                        self.__append(" ", False)
                        index_name: str = (
                            f"{LogicPySMTConstraintGenerator.__get_quantified_variable_name(target.target)}{_INDEX_SUFFIX}"
                        )
                        index = SubscriptElement(Index(Name(index_name)))
                        element = Subscript(sequence, [index])
                        element.visit(self)
                        self.__append(")", False)
                    self.__append_line(")", False)

                    self.__append("(and")
                    self.__indent.indent()

        return False

    def visit_Attribute(self, node: Attribute) -> Optional[bool]:
        self.__append_function_application(node.attr.value, [node.value])
        return False

    def visit_BooleanOperation(self, node: BooleanOperation) -> Optional[bool]:
        operator: BaseBooleanOp = node.operator
        if isinstance(operator, Or):
            self.__append_line("(not", False)
            self.__indent.indent()
            self.__append_line("(=")
            self.__indent.indent()
            self.__append("")
            node.left.visit(self)
            self.__append_newline()
            self.__append("")
            node.right.visit(self)
            self.__append_newline()
            self.__indent.dedent()
            self.__append_line(")")
            self.__indent.dedent()
            self.__append(")")
        else:
            self.__append("(", False)
            self.visit_BaseBooleanOp(operator)
            self.__append_newline()
            self.__indent.indent()
            self.__append("")
            node.left.visit(self)
            self.__append_newline()
            self.__append("")
            node.right.visit(self)
            self.__append_newline()
            self.__indent.dedent()
            self.__append(")")

        return False

    def visit_BaseBooleanOp(self, node: BaseBooleanOp) -> Optional[bool]:
        if isinstance(node, And):
            self.__append("and", False)
        elif isinstance(node, Or):
            self.__append("or", False)
        else:
            self.__append(f"missing_boolean_op: {type(node).__name__}", False)

    def visit_Call(self, node: Call) -> Optional[bool]:
        func: BaseExpression = node.func
        if isinstance(func, Name):
            name: str = func.value
            match name:
                case "len":
                    iterable: BaseExpression = node.args[0].value
                    if isinstance(iterable, Attribute):
                        list_name: str = iterable.attr.value
                        func_name: str = f"{list_name}_size"
                        self.__append_function_application(func_name, [iterable.value])
                        return False

    def visit_ClassDef(self, node: ClassDef) -> Optional[bool]:
        return node.name.value != UNIQUE

    def visit_Comparison(self, node: Comparison) -> Optional[bool]:
        has_conjunction: bool = len(node.comparisons) > 1
        if has_conjunction:
            self.__append_line("(and", False)
            self.__indent.indent()
            self.__append("")

        left: BaseExpression = node.left
        is_first: bool = True
        for target in node.comparisons:
            if is_first:
                is_first = False
            else:
                self.__append_newline()
                self.__append("")

            operator: BaseCompOp = target.operator
            right: BaseExpression = target.comparator
            if isinstance(operator, NotEqual):
                not_equal = UnaryOperation(
                    Not(), Comparison(left, [ComparisonTarget(Equal(), right)])
                )
                not_equal.visit(self)
            elif isinstance(operator, In) or isinstance(operator, NotIn):
                should_negate: bool = isinstance(operator, NotIn)
                if should_negate:
                    self.__not_prefix()

                tmp_var = Name(f"__logicpy_in_tmp_{self.__in_tmp_counter}")
                self.__in_tmp_counter += 1
                target = AssignTarget(tmp_var)
                some_func = Name(SOME)
                some = Call(some_func, [Arg(right)])
                assign = Assign([target], some)
                equal_to = Comparison(tmp_var, [ComparisonTarget(Equal(), left)])
                block = IndentedBlock(
                    [
                        SimpleStatementLine([assign]),
                        SimpleStatementLine([Expr(equal_to)]),
                    ]
                )
                block.visit(self)

                if should_negate:
                    self.__not_suffix()
            else:
                smt_operator: str = f"missing_operator: {type(operator).__name__}"
                if isinstance(operator, LessThan):
                    smt_operator = "<"
                elif isinstance(operator, LessThanEqual):
                    smt_operator = "<="
                elif isinstance(operator, GreaterThan):
                    smt_operator = ">"
                elif isinstance(operator, GreaterThanEqual):
                    smt_operator = ">="
                elif isinstance(operator, Equal):
                    smt_operator = "="

                self.__append_line(f"({smt_operator}", False)
                self.__indent.indent()
                self.__append("")
                left.visit(self)
                self.__append_newline()
                self.__append("")
                right.visit(self)
                self.__append_newline()
                self.__indent.dedent()
                self.__append(")")

            left = right

        if has_conjunction:
            self.__indent.dedent()
            self.__append_newline()
            self.__append(")")

        return False

    def visit_If(self, node: If) -> Optional[bool]:
        guard: BaseExpression = node.test
        previous_branch_guard: BaseExpression = self.__branch_guard
        self.__branch_guard = LogicPySMTConstraintGenerator.__conjunction(
            previous_branch_guard, guard
        )
        self.__scope_manager.begin_scope()
        node.body.visit(self)
        self.__scope_manager.end_scope()

        orelse: Optional[Union[If, Else]] = node.orelse
        if orelse:
            self.__branch_guard = LogicPySMTConstraintGenerator.__conjunction(
                previous_branch_guard, UnaryOperation(Not(), guard)
            )
            orelse.visit(self)

        self.__branch_guard = previous_branch_guard
        return False

    def visit_IfExp(self, node: IfExp) -> Optional[bool]:
        self.__append_line("(ite", False)
        self.__indent.indent()
        self.__append("")
        node.test.visit(self)
        self.__append_newline()
        self.__append("")
        node.body.visit(self)
        self.__append_newline()
        self.__append("")
        node.orelse.visit(self)
        self.__indent.dedent()
        self.__append_newline()
        self.__append(")")
        return False

    def visit_Integer(self, node: Integer) -> Optional[bool]:
        self.__append(f"{node.evaluated_value}", False)

    def visit_Float(self, node: Float) -> Optional[bool]:
        self.__append(f"{node.evaluated_value}", False)

    def visit_FunctionDef(self, node: FunctionDef) -> Optional[bool]:
        super().visit_FunctionDef(node)
        for param in node.params.params:
            annotation: Optional[Annotation] = param.annotation
            if not annotation:
                continue
            param_name: BaseExpression = annotation.annotation
            if not isinstance(param_name, Name):
                continue

            declared_type: str = param_name.value
            is_universe: bool = declared_type == _UNIVERSE_CLASS_NAME
            if is_universe:
                self.__universe_variable_name = param.name.value

        self.__append_line("(assert")
        self.__indent.indent()
        self.__is_in_function = True
        self.__append("")
        node.body.visit(self)
        self.__indent.dedent()
        self.__append_newline()
        self.__append_line(")")

        name: str = node.name.value
        if name == _PREMISE:
            self.__premise = self.__body
        if name == _CONCLUSION:
            self.__conclusion = self.__body
            self.__negated_conclusion = self.__negated_body_template.replace(
                "(assert", "(not"
            )

        self.__negated_body_template = ""
        self.__body = ""
        self.__is_in_function = False
        self.__indent = Indent()
        self.__scope_manager = ScopeManager()
        return False

    def visit_For(self, node: For) -> Optional[bool]:
        self.__append_line("(forall", False)
        self.__scope_manager.begin_scope()
        self.__indent.indent()
        self.__append("((")
        target: BaseAssignTargetExpression = node.target
        if isinstance(target, Name):
            name: str = target.value
            self.__scope_manager.declare_variable(name)
            self.__scope_manager.declare_variable(f"{name}{_INDEX_SUFFIX}")

        node.target.visit(self)
        self.__append_line(f"{_INDEX_SUFFIX} Int))", False)
        self.__append_line(f"(let")
        self.__indent.indent()
        self.__append("((")
        node.target.visit(self)
        self.__append(" ", False)
        index_name: str = (
            f"{LogicPySMTConstraintGenerator.__get_quantified_variable_name(node.target)}{_INDEX_SUFFIX}"
        )
        index = SubscriptElement(Index(Name(index_name)))
        element = Subscript(node.iter, [index])
        element.visit(self)
        self.__append_line("))", False)

        self.__append("")
        node.body.visit(self)
        self.__append_newline()
        self.__indent.dedent()
        self.__append_line(")")
        self.__indent.dedent()
        self.__append(")")
        self.__scope_manager.end_scope()
        return False

    def visit_IndentedBlock(self, node: IndentedBlock) -> Optional[bool]:
        if not self.__is_in_function:
            return False

        self.__append_line("(and", False)
        self.__indent.indent()
        self.__scope_manager.begin_scope()
        for element in node.body:
            self.__append("")
            element.visit(self)
            self.__append_newline()

        exists_count: Optional[int] = self.__scope_manager.get_context_value(
            _EXISTS_STACK_COUNT
        )
        if exists_count:
            for _ in range(exists_count):
                self.__indent.dedent()
                self.__append_line(")")
        self.__scope_manager.end_scope()

        self.__indent.dedent()
        self.__append(")")
        return False

    def leave_Module(self, original_node: Module) -> None:
        self.smt_harness = f"""{self.__premise}
(check-sat)

(push)
{self.__conclusion}
(check-sat)
(pop)

(assert
{self.__negated_conclusion})
(check-sat)
"""

    def visit_Name(self, node: Name) -> Optional[bool]:
        if self.__is_in_function:
            value: str = node.value
            smt_value: str
            if value == FALSE_NAME:
                smt_value = "false"
            elif value == TRUE_NAME:
                smt_value = "true"
            else:
                smt_value = self.__scope_manager.get_qualified_name(value)
            self.__append(smt_value, False)

    def visit_Pass(self, node: Pass) -> Optional[bool]:
        Assert(TRUE).visit(self)

    def visit_SimpleString(self, node: SimpleString) -> Optional[bool]:
        self.__append(f'"{node.evaluated_value}"', False)

    def visit_Subscript(self, node: Subscript) -> Optional[bool]:
        if self.__is_in_function:
            subscript_element: BaseSlice = node.slice[0].slice
            if isinstance(subscript_element, Index):
                index: BaseExpression = subscript_element.value
                if isinstance(index, Name) or isinstance(index, Integer):
                    value: BaseExpression = node.value
                    if isinstance(value, Attribute):
                        self.__append_function_application(
                            value.attr.value, [value.value, index]
                        )
                        return False

    def visit_UnaryOperation(self, node: UnaryOperation) -> Optional[bool]:
        operator: BaseUnaryOp = node.operator
        if isinstance(operator, Not):
            self.__not_prefix()
            node.expression.visit(self)
            self.__not_suffix()
        else:
            raise ValueError(f"Unsupported unary operator: {operator}")
        return False

    def __not_prefix(self) -> None:
        """
        Prefix to negate the statement contained. Usually, if we want to negate
        an expression, we just wrap it in a unary expression with the not
        operator and visit that. This helper is extracted for situations where
        the SMT grammar does not align with the Python grammar, e.g. when we
        want to negate an SMT expression that would be a block or a statement
        in Python.
        """
        self.__append_line("(not", False)
        self.__indent.indent()
        self.__append("", True)

    def __not_suffix(self) -> None:
        """
        Terminates an SMT negation expression started with `__not_prefix()`.
        """
        self.__append_newline()
        self.__indent.dedent()
        self.__append(")")

    def __increment_exists_stack_counter(self, delta: int = 1) -> None:
        """
        Local variable declarations using `some(...)` lead to an existential
        quantifier as well as a `let` expression in SMT, which need to be
        closed afterwards. We use a custom user context variable in the current
        stack scope to keep track of this.
        Args:
            delta (int): Number of additional SMT expressions to close at the
            end of the current scope block.
        """
        exists_stack_count: int = (
            self.__scope_manager.get_context_value(_EXISTS_STACK_COUNT) or 0
        )
        self.__scope_manager.set_context_value(
            _EXISTS_STACK_COUNT,
            exists_stack_count + delta,
        )

    def __append_function_application(
        self, func_name: str, args: list[BaseExpression]
    ) -> None:
        """
        Helper to create an SMT function application, e.g. for constants,
        attributes or list element access.
        Args:
            func_name (str): Name of the constant, attribute or list to access.
            args (list[BaseExpression]): Attribute value or list index.
        """
        this_arg: BaseExpression = args[0]
        this_type: Optional[str] = self.get_metadata(
            TypeInferenceProvider, this_arg, None
        )
        if this_type is None:
            if isinstance(this_arg, Subscript):
                container_type: Optional[str] = self.get_metadata(
                    TypeInferenceProvider, this_arg.value, None
                )
                if container_type is not None:
                    element_type: Optional[Match] = match(_ELEMENT_TYPE, container_type)
                    if element_type is not None:
                        this_type = element_type.group(1)

        if this_type is not None:
            this_type = this_type.rsplit(".", 1)[-1]

        original_attribute_name: str = f"__attribute_{this_type}_{func_name}"
        attribute_name: str = original_attribute_name
        potential_bases: deque[str] = deque()
        self.__add_ancestors(potential_bases, this_type)
        while not attribute_name in self.__smt_attributes and potential_bases:
            ancestor: str = potential_bases.popleft()
            self.__add_ancestors(potential_bases, ancestor)
            attribute_name = f"__attribute_{ancestor}_{func_name}"

        if not attribute_name in self.__smt_attributes:
            attribute_name = original_attribute_name

        args_without_universe: list[BaseExpression] = [
            arg
            for arg in args
            if not isinstance(arg, Name) or arg.value != self.__universe_variable_name
        ]
        if not args_without_universe:
            self.__append(attribute_name, False)
        else:
            self.__append(f"({attribute_name}", False)
            for arg in args_without_universe:
                self.__append(f" ", False)
                arg.visit(self)
            self.__append(f")", False)

    def __add_ancestors(
        self, potential_bases: deque[str], python_class_name: Optional[str]
    ) -> None:
        if not python_class_name:
            return

        ancestors: set[str] = self.__ancestors_per_class[python_class_name]
        for ancestor in ancestors:
            potential_bases.append(ancestor)

    def __append(self, code: str, indented: bool = True) -> None:
        """
        Appends the given code to the currently processed SMT output body,
        prepending the current indentation level.
        Args:
            code (str): SMT code to append to generated SMT output.
            indented (bool): Whether the given code should be indented at the
            curent indentation level.
        """
        if indented:
            indented_code: str = self.__indent.prepend(code)
            self.__body += indented_code
            self.__negated_body_template += _INDENT + indented_code
        else:
            self.__body += code
            self.__negated_body_template += code

    def __append_line(self, line: str, indented: bool = True) -> None:
        """
        Equivalent to `__append(...)`, but terminates the line with a newline.
        Args:
            line (str): SMT code to append to generated SMT output.
            indented (bool): Whether the given code should be indented at the
            curent indentation level.
        """
        self.__append(line + "\n", indented)

    def __append_newline(self) -> None:
        """
        Adds a newline to SMT output, without indentation.
        """
        self.__append_line("", False)

    @staticmethod
    def __conjunction(lhs: BaseExpression, rhs: BaseExpression) -> BaseExpression:
        """
        Creates a logical conjunction between lhs and rhs as a libCST
        expression.
        Args:
            lhs (BaseExpression): Left-hand side of conjunction.
            rhs (BaseExpression): Right-hand side of conjunction.
        """
        if lhs is TRUE:
            return rhs

        return BooleanOperation(lhs, And(), rhs)

    @staticmethod
    def __get_quantified_variable_name(expr: BaseExpression) -> str:
        """
        Helper to extract quantified SMT variable name out of assignment or for
        loop target.
        Args:
            expr (BaseExpression): Expression to convert to quantifiable
            symbol, if possible
        """
        if isinstance(expr, Name):
            return expr.value

        raise ValueError(f"Quantifier variable needs to be a name: {expr}")
