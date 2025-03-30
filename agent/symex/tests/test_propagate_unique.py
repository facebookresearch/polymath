# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from unittest import IsolatedAsyncioTestCase

from agent.logic.z3_conclusion_check_engine_strategy import _PYTHON_CODE_PREFIX
from agent.symex.collect_unique_ids import CollectUniquelyIdentifiedVars
from agent.symex.module_with_type_info_factory import ModuleWithTypeInfoFactory
from agent.symex.propagate_unique import PropagateUnique

from agent.symex.unique_aliasing import UniqueAliasing

from libcst import MetadataWrapper


class TestPropagateUnique(IsolatedAsyncioTestCase):

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.maxDiff = None

    async def test_inline(self) -> None:
        await self.__run_harness(
            """
class Person:
    name: Unique[str]
    age: int

class Universe:
    persons: list[Person]

def premise(universe: Universe) -> None:
    peter = some(universe.persons)
    assert peter.name == "Peter"
    assert peter.age == 35
    peter_2 = some(universe.persons)
    assert peter.name == peter_2.name

    for person in universe.persons:
        if person.age > 30:
            assert person.name == "Peter"
        if person.name == "Peter":
            assert person.age > 30
        if person.name == peter.name:
            assert person.age > 31
""",
            """
class Person:
    name: Unique[str]
    age: int

class Universe:
    persons: list[Person]

def premise(universe: Universe) -> None:
    assert universe.persons[0].name == "Peter"
    assert universe.persons[0].age == 35
    assert universe.persons[0].name == universe.persons[0].name

    for person in universe.persons:
        if person.age > 30:
            assert person == universe.persons[0]
        if person == universe.persons[0]:
            assert person.age > 30
        if person == universe.persons[0]:
            assert person.age > 31
""",
        )

    async def __run_harness(self, before: str, after: str) -> None:
        code: str = _PYTHON_CODE_PREFIX + before
        wrapper: MetadataWrapper = await ModuleWithTypeInfoFactory.create_module(code)
        aliasing = UniqueAliasing()
        wrapper.visit(aliasing)
        collect_ids = CollectUniquelyIdentifiedVars()
        wrapper.visit(collect_ids)

        propagate = PropagateUnique(aliasing.aliases, collect_ids.replacements)
        transformed: str = wrapper.visit(propagate).code
        self.assertEqual(_PYTHON_CODE_PREFIX + after, transformed)
