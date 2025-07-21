# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from unittest import IsolatedAsyncioTestCase

from agent.logic.z3_conclusion_check_engine_strategy import _PYTHON_CODE_PREFIX
from agent.symex.collect_unique_ids import CollectUniquelyIdentifiedVars
from agent.symex.module_with_type_info_factory import ModuleWithTypeInfoFactory

from libcst import MetadataWrapper


class TestCollectUniquelyIdentifiedVars(IsolatedAsyncioTestCase):

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.maxDiff = None

    async def test_single(self) -> None:
        await self.__run_harness(
            {"4__peter": "universe.persons[0]"},
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
        )

    async def test_multiple(self) -> None:
        await self.__run_harness(
            {
                "4__peter": "universe.persons[0]",
                "4__peter_again": "universe.persons[0]",
                "4__john": "universe.persons[1]",
                "4__0__danielle": "universe.persons[2]",
                "4__1__jane": "universe.persons[3]",
            },
            """
class Person:
    name: Unique[str]
    age: int

class Universe:
    persons: list[Person]

def premise(universe: Universe) -> None:
    peter = some(universe.persons)
    assert peter.name == "Peter"
    peter_alias = some(universe.persons)
    assert peter.name == peter_alias.name
    peter_again = some(universe.persons)
    assert peter_again.name == "Peter"
    john = some(universe.persons)
    assert "John" == john.name

    if john.age > 30:
        danielle = some(universe.persons)
        assert danielle.name == "Danielle"

    for person in universe.persons:
        jane = some(universe.persons)
        assert jane.name == "Jane"
        if person.age > 30:
            assert person.name == "Peter"
""",
        )

    async def __run_harness(
        self, expected_replacements: dict[str, str], code: str
    ) -> None:
        prefixed_code: str = _PYTHON_CODE_PREFIX + code
        wrapper: MetadataWrapper = await ModuleWithTypeInfoFactory.create_module(
            prefixed_code
        )
        collect_ids = CollectUniquelyIdentifiedVars()
        wrapper.visit(collect_ids)
        actual_replacements: dict[str, str] = {
            name: wrapper.module.code_for_node(replacement)
            for name, replacement in collect_ids.replacements.items()
        }
        self.assertEqual(expected_replacements, actual_replacements)
