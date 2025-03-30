from re import compile, Pattern
from typing import Optional, Sequence, Tuple, Union

from agent.symex.boolean import FALSE_NAME, TRUE_NAME
from agent.symex.scope import ScopeManager

from libcst import (
    Assert,
    Assign,
    Attribute,
    BaseAssignTargetExpression,
    BaseExpression,
    BaseNumber,
    BaseStatement,
    Comparison,
    ComparisonTarget,
    CSTVisitor,
    Equal,
    FlattenSentinel,
    FunctionDef,
    IndentedBlock,
    Name,
    RemovalSentinel,
    SimpleString,
)
from libcst.codemod import CodemodContext, VisitorBasedCodemodCommand

from libcst.metadata import TypeInferenceProvider


_UNIQUE_TYPE_PATTERN: Pattern = compile(r"Unique\[.*\]")


class CollectUniquelyIdentifiedVars(CSTVisitor):

    METADATA_DEPENDENCIES = (TypeInferenceProvider,)

    def __init__(self) -> None:
        super().__init__()
        self.__scope_manager = ScopeManager()
        self.__is_in_function: bool = False

    def leave_FunctionDef(self, original_node: FunctionDef) -> None:
        self.__is_in_function = False

    def leave_IndentedBlock(self, original_node: IndentedBlock) -> None:
        self.__scope_manager.end_scope()

    def visit_Assert(self, node: Assert) -> Optional[bool]:
        test: BaseExpression = node.test
        if not isinstance(test, Comparison):
            return

        comparisons: Sequence[ComparisonTarget] = test.comparisons
        if len(comparisons) > 1:
            return

        comparison: ComparisonTarget = comparisons[0]
        if not isinstance(comparison.operator, Equal):
            return

        attribute, value = CollectUniquelyIdentifiedVars.__get_atrribute_and_value(
            test.left, comparison.comparator
        )

        if not attribute or not value:
            return

        attr: str = attribute.attr.value
        name: BaseExpression = attribute.value
        if not isinstance(name, Name):
            return

        attr_type: Optional[str] = self.get_metadata(
            TypeInferenceProvider, attribute, None
        )

        return super().visit_Assert(node)

    def visit_Assign(self, node: Assign) -> Optional[bool]:
        if not self.__is_in_function:
            return

        for target in node.targets:
            name: BaseAssignTargetExpression = target.target
            if not isinstance(name, Name):
                continue

            self.__scope_manager.declare_variable(name.value)

    def visit_FunctionDef(self, node: FunctionDef) -> Optional[bool]:
        self.__is_in_function = True

    def visit_IndentedBlock(self, node: IndentedBlock) -> Optional[bool]:
        self.__scope_manager.begin_scope()

    @staticmethod
    def __get_atrribute_and_value(
        left: BaseExpression, comparator: BaseExpression
    ) -> Tuple[Optional[Attribute], Optional[BaseExpression]]:
        value: Optional[BaseExpression] = CollectUniquelyIdentifiedVars.__get_value(
            left
        )
        if value:
            if isinstance(comparator, Attribute):
                return comparator, value
        else:
            value = CollectUniquelyIdentifiedVars.__get_value(comparator)
            if value and isinstance(left, Attribute):
                return left, value

        return None, None

    @staticmethod
    def __get_value(
        value: BaseExpression,
    ) -> Optional[SimpleString | BaseNumber | Name]:
        if isinstance(value, SimpleString) or isinstance(value, BaseNumber):
            return value

        if not isinstance(value, Name):
            return None

        name: str = value.value
        return value if name == TRUE_NAME or name == FALSE_NAME else None


class PropagateUnique(VisitorBasedCodemodCommand):

    def __init__(self, context: CodemodContext) -> None:
        super().__init__(context)
        self.__is_in_function: bool = False

    def leave_FunctionDef(
        self, original_node: FunctionDef, updated_node: FunctionDef
    ) -> Union[BaseStatement, FlattenSentinel[BaseStatement], RemovalSentinel]:
        self.__is_in_function = False
        return updated_node

    def visit_FunctionDef(self, node: FunctionDef) -> Optional[bool]:
        self.__is_in_function = True
