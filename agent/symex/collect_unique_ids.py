# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence, TypeAlias, Union

from agent.symex.boolean import FALSE_NAME, TRUE_NAME
from agent.symex.scope import ScopeManager
from agent.symex.unique_aliasing import UniqueAliasing
from libcst import (
    Assert,
    Assign,
    Attribute,
    BaseExpression,
    Call,
    Comparison,
    ComparisonTarget,
    CSTVisitor,
    Equal,
    Float,
    IndentedBlock,
    Index,
    Integer,
    Name,
    SimpleString,
    Subscript,
    SubscriptElement,
)
from libcst.metadata import TypeInferenceProvider


# Function used for existential quantification in Logic.py
SOME: str = "some"

# Types that we support for `Unique[...]` attributes.
SupportedUniqueIdTypes: TypeAlias = Union[str, int, float, bool]


class UniquePropertyToObjectId:

    def __init__(self) -> None:
        self.__unique_property_to_next_id: dict[str, dict[str, int]] = {}
        self.__unique_property_to_id: dict[
            str, dict[str, dict[SupportedUniqueIdTypes, int]]
        ] = {}
        self.id_expression: dict[int, Subscript] = {}

    def get_object_id(
        self, class_type: str, attribute_name: str, value: SupportedUniqueIdTypes
    ) -> Optional[int]:
        object_id: int = self.__get_or_create_object_id(
            class_type, attribute_name, value, False
        )
        return object_id if object_id != -1 else None

    def get_or_create_object_id(
        self, class_type: str, attribute_name: str, value: SupportedUniqueIdTypes
    ) -> int:
        return self.__get_or_create_object_id(class_type, attribute_name, value, True)

    def __get_or_create_object_id(
        self,
        class_type: str,
        attribute_name: str,
        value: SupportedUniqueIdTypes,
        generate_if_missing: bool,
    ) -> int:
        property_to_id: dict[str, dict[SupportedUniqueIdTypes, int]] = (
            self.__unique_property_to_id.setdefault(class_type, {})
        )
        value_to_id: dict[SupportedUniqueIdTypes, int] = property_to_id.setdefault(
            attribute_name, {}
        )

        object_id: Optional[int] = value_to_id.get(value)
        if object_id is not None:
            return object_id
        if not generate_if_missing:
            return -1

        property_to_next_id: dict[str, int] = (
            self.__unique_property_to_next_id.setdefault(class_type, {})
        )
        object_id = property_to_next_id.setdefault(attribute_name, 0)
        value_to_id[value] = object_id
        property_to_next_id[attribute_name] = object_id + 1
        return object_id


class CollectUniquelyIdentifiedVars(CSTVisitor):
    """
    Collects all variables which are uniquely identified by an explicit
    assertion on one of their `Unique[...]` attributes.

    Example:
    ```python
    john = some(universe.persons)
    assert john.name == "John"
    ```

    This will lead to a replacement entry `"1__john" -> unverse.persons[0]`,
    assuming that "John" is the first explicit person name. The following
    assumptions apply:
    - At most one `Unique[...]` attribute exists per class.
    - All variables are read-only and not assigned a new value.

    TODO: Ignore classes with multiple `Unique[...]` attributes.
    """

    METADATA_DEPENDENCIES = (TypeInferenceProvider,)

    def __init__(self) -> None:
        super().__init__()
        self.__name_to_universe_collection: dict[str, BaseExpression] = {}
        self.__scope_manager = ScopeManager()
        self.replacements: dict[str, Subscript] = {}
        self.unique_property_to_object_id = UniquePropertyToObjectId()

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

        if attribute is None or value is None:
            return

        attr_type: Optional[str] = self.get_metadata(
            TypeInferenceProvider, attribute, None
        )
        if attr_type is None or not UniqueAliasing.is_unique_type(attr_type):
            return

        attr: str = attribute.attr.value
        name: BaseExpression = attribute.value
        if not isinstance(name, Name):
            return

        class_type: Optional[str] = self.get_metadata(TypeInferenceProvider, name, None)
        if class_type is None:
            return

        qualified_name: str = self.__scope_manager.get_qualified_name(name.value)
        universe_collection: Optional[BaseExpression] = (
            self.__name_to_universe_collection.get(qualified_name)
        )
        if universe_collection is None:
            return

        object_id: int = self.unique_property_to_object_id.get_or_create_object_id(
            class_type, attr, value
        )
        index = Index(Integer(str(object_id)))
        replacement = Subscript(universe_collection, [SubscriptElement(index)])
        self.unique_property_to_object_id.id_expression[object_id] = replacement
        self.replacements[qualified_name] = replacement

    def visit_Assign(self, node: Assign) -> Optional[bool]:
        names_and_values: list[tuple[str, BaseExpression]] = (
            UniqueAliasing.declare_variable(self.__scope_manager, node)
        )
        for var, value in names_and_values:
            qualified_name: str = self.__scope_manager.get_qualified_name(var)
            if isinstance(value, Call):
                func: BaseExpression = value.func
                if isinstance(func, Name) and func.value == SOME:
                    arg: BaseExpression = value.args[0].value
                    self.__name_to_universe_collection[qualified_name] = arg

    def visit_IndentedBlock(self, node: IndentedBlock) -> Optional[bool]:
        self.__scope_manager.begin_scope()

    @staticmethod
    def __get_atrribute_and_value(
        left: BaseExpression, comparator: BaseExpression
    ) -> tuple[Optional[Attribute], Optional[SupportedUniqueIdTypes]]:
        """
        Helper to convert the left-hand and right-hand side of an assertion to
        an attribute and a supported unique ID type, irrespective of which is
        on the left or the right side.

        Args:
            left (BaseExpression): Attribute or unique ID value.
            comparator (BaseExpression): Attribute or unique ID value.
        Returns:
            Tuple of attribute and its asserted value. If either value in the
            tuple is None, no match was found.
        """
        value: Optional[SupportedUniqueIdTypes] = (
            CollectUniquelyIdentifiedVars.get_value(left)
        )
        if value:
            if isinstance(comparator, Attribute):
                return comparator, value
        else:
            value = CollectUniquelyIdentifiedVars.get_value(comparator)
            if value and isinstance(left, Attribute):
                return left, value

        return None, None

    @staticmethod
    def get_value(
        value: BaseExpression,
    ) -> Optional[SupportedUniqueIdTypes]:
        """
        Converts a libCST value to a Python value that we can compare.

        Args:
            value (BaseExpression): libCST value to convert.
        Returns:
            Comparable Python value, used to associated values to integer IDs.
        """
        if isinstance(value, SimpleString):
            return value.raw_value
        if isinstance(value, Integer):
            return value.evaluated_value
        if isinstance(value, Float):
            return value.evaluated_value

        if not isinstance(value, Name):
            return None

        name: str = value.value
        if name == TRUE_NAME:
            return True
        if name == FALSE_NAME:
            return False
        return None
