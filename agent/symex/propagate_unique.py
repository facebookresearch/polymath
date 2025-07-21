# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence, Type, Union

from agent.symex.collect_unique_ids import (
    CollectUniquelyIdentifiedVars,
    SOME,
    SupportedUniqueIdTypes,
    UniquePropertyToObjectId,
)
from agent.symex.module_with_type_info_factory import ModuleWithTypeInfoFactory
from agent.symex.scope import ScopeManager

from agent.symex.unique_aliasing import UniqueAliasing

from libcst import (
    Assign,
    AssignTarget,
    Attribute,
    BaseAssignTargetExpression,
    BaseCompOp,
    BaseExpression,
    BaseSuite,
    Call,
    Comparison,
    ComparisonTarget,
    CSTTransformer,
    In,
    IndentedBlock,
    Is,
    IsNot,
    MetadataWrapper,
    Module,
    Name,
    NotIn,
    RemovalSentinel,
    RemoveFromParent,
    Subscript,
)

from libcst.metadata import TypeInferenceProvider


class PropagateUnique(CSTTransformer):
    """
    For unique attribute types that our preprocessing can handle, this
    transformer converts objects with a unique ID attribute to a unique object
    ID. As an example, if `person.name` is a unique property, we e.g. map
    "John" to object ID `0` if we encounter it in an assertion. As a result, we
    can avoid a quantifier for this object in SMT, and we can replace string
    comparisons in SMT by integer comparisons.

    All transformations are intended to yield equi-satisfiable Logic.py code in
    SMT and are pure performance optimisations.
    """

    METADATA_DEPENDENCIES = (TypeInferenceProvider,)

    @staticmethod
    async def preprocess(module: MetadataWrapper) -> Module:
        aliasing = UniqueAliasing()
        module.visit(aliasing)
        collect_ids = CollectUniquelyIdentifiedVars()
        module.visit(collect_ids)

        propagate = PropagateUnique(
            aliasing.aliases,
            collect_ids.replacements,
            collect_ids.unique_property_to_object_id,
        )
        return module.visit(propagate)

    def __init__(
        self,
        aliases: dict[str, str],
        replacements: dict[str, Subscript],
        unique_property_to_object_id: UniquePropertyToObjectId,
    ) -> None:
        """
        Args:
            aliases (dict[str, str]): Aliases from `UniqueAliasing`. We use this
            to replace alises by unique attributes, before applying further
            optimisations.
            replacements (dict[str, Subscript]): Direct `universe` expressions
            replacing local variables from `CollectUniquelyIdentifiedVars`. We
            use this to e.g. replace the local variable `john` by
            `universe.persons[1]` if John's object ID could be set to `1`
            without loss of generality.
            unique_property_to_object_id (UniquePropertyToObjectId): Mapping
            from unique attribute values (literals) to object ids from
            `CollectUniquelyIdentifiedVars`. We use this to replace constant
            literals of unique attributes by their object id. That means we
            might e.g. replace `person.name == "Peter"` by `person == 3` of
            Peter's ID could be set to `3` without loss of generality.
        """
        super().__init__()
        self.__aliases: dict[str, str] = aliases
        self.__replacements: dict[str, Subscript] = replacements
        self.__unique_property_to_object_id: UniquePropertyToObjectId = (
            unique_property_to_object_id
        )
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

    def leave_Comparison(
        self, original_node: Comparison, updated_node: Comparison
    ) -> Comparison:
        original_left: BaseExpression = original_node.left
        updated_left: BaseExpression = updated_node.left
        new_updated_left: Optional[BaseExpression] = None
        new_updated_comparisons: list[ComparisonTarget] = []
        was_modified: bool = False

        for i in range(len(updated_node.comparisons)):
            original_comparison: ComparisonTarget = original_node.comparisons[i]
            comp_op: BaseCompOp = original_comparison.operator
            if (
                isinstance(comp_op, In)
                or isinstance(comp_op, NotIn)
                or isinstance(comp_op, Is)
                or isinstance(comp_op, IsNot)
            ):
                return updated_node

            original_right: BaseExpression = original_comparison.comparator
            updated_comparison: ComparisonTarget = updated_node.comparisons[i]
            updated_right: BaseExpression = updated_comparison.comparator

            left_type: str = self.get_metadata(TypeInferenceProvider, original_left, "")
            right_type: str = self.get_metadata(
                TypeInferenceProvider, original_right, ""
            )
            is_unique: bool = UniqueAliasing.is_unique_type(
                left_type
            ) or UniqueAliasing.is_unique_type(right_type)
            if is_unique:
                was_modified = True
                updated_left = self.__convert_unique_property(
                    updated_left, original_right
                )
                updated_right = self.__convert_unique_property(
                    updated_right, original_left
                )

            if new_updated_left is None:
                new_updated_left = updated_left
            new_updated_comparisons.append(
                ComparisonTarget(updated_comparison.operator, updated_right)
            )
            original_left = original_right
            updated_left = updated_right

        return (
            updated_node.with_changes(
                left=new_updated_left, comparisons=new_updated_comparisons
            )
            if was_modified
            else updated_node
        )

    def leave_IndentedBlock(
        self, original_node: IndentedBlock, updated_node: IndentedBlock
    ) -> BaseSuite:
        self.__scope_manager.end_scope()
        return updated_node

    def leave_Name(self, original_node: Name, updated_node: Name) -> BaseExpression:
        name: str = self.__scope_manager.get_qualified_name(original_node.value)
        aliased_name: str = self.__aliases.get(name, name)
        expr: Optional[Subscript] = self.__replacements.get(aliased_name)
        if expr is not None:
            return expr

        if name == aliased_name:
            return updated_node
        return updated_node.with_changes(
            value=ScopeManager.to_unqualified_name(aliased_name)
        )

    def visit_Assign(self, node: Assign) -> Optional[bool]:
        UniqueAliasing.declare_variable(self.__scope_manager, node)

    def visit_IndentedBlock(self, node: IndentedBlock) -> Optional[bool]:
        self.__scope_manager.begin_scope()

    def __convert_unique_property(
        self, updated_node: BaseExpression, original_other: BaseExpression
    ) -> BaseExpression:
        """
        Helper to convert a unique property in a comparison to an object ID, if
        possible. Will e.g. replace `object.unique_property` by just `object`,
        and "Jane" by `universe.persons[7]`. The intend is to convert the
        comparison to an integer comparison.
        """
        if isinstance(updated_node, Attribute):
            return updated_node.value

        if not isinstance(original_other, Attribute):
            return updated_node

        value: Optional[SupportedUniqueIdTypes] = (
            CollectUniquelyIdentifiedVars.get_value(updated_node)
        )
        if value is None:
            return updated_node

        class_type: str = self.get_metadata(
            TypeInferenceProvider, original_other.value, ""
        )
        attribute_name: str = original_other.attr.value
        object_id: Optional[int] = self.__unique_property_to_object_id.get_object_id(
            class_type, attribute_name, value
        )
        return (
            updated_node
            if object_id is None
            else self.__unique_property_to_object_id.id_expression[object_id]
        )
