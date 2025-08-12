# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from unittest import IsolatedAsyncioTestCase

from agent.logic.z3_conclusion_check_engine_strategy import _PYTHON_CODE_PREFIX
from agent.symex.module_with_type_info_factory import ModuleWithTypeInfoFactory

from agent.symex.unique_aliasing import UniqueAliasing

from libcst import MetadataWrapper


class TestUniqueAliasing(IsolatedAsyncioTestCase):

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.maxDiff = None

    async def test_single(self) -> None:
        await self.__run_harness(
            {"4__peter_2": "4__peter"},
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
            {"4__peter_2": "4__peter", "4__0__peter_3": "4__peter"},
            """
class Person:
    name: Unique[str]
    age: int

class Universe:
    persons: list[Person]

def premise(universe: Universe) -> None:
    peter = some(universe.persons)
    peter_2 = some(universe.persons)
    assert peter.name == peter_2.name
    if peter_2.name == "Peter":
        peter_3 = some(universe.persons)
        assert peter_3.name == peter_2.name
    john = some(universe.persons)
""",
        )

    async def test_order(self) -> None:
        await self.__run_harness(
            {"4__peter_2": "4__peter", "4__peter_3": "4__peter"},
            """
class Person:
    name: Unique[str]
    age: int

class Universe:
    persons: list[Person]

def premise(universe: Universe) -> None:
    peter = some(universe.persons)
    peter_2 = some(universe.persons)
    peter_3 = some(universe.persons)
    assert peter_2.name == peter_3.name
    assert peter.name == peter_3.name
""",
        )

    async def test_circular(self) -> None:
        await self.__run_harness(
            {"4__peter_2": "4__peter", "4__peter_3": "4__peter"},
            """
class Person:
    name: Unique[str]
    age: int

class Universe:
    persons: list[Person]

def premise(universe: Universe) -> None:
    peter = some(universe.persons)
    peter_2 = some(universe.persons)
    peter_3 = some(universe.persons)
    assert peter.name == peter_2.name
    assert peter_2.name == peter_3.name
    assert peter_3.name == peter.name
""",
        )

    async def __run_harness(self, expected_aliases: dict[str, str], code: str) -> None:
        prefixed_code: str = _PYTHON_CODE_PREFIX + code
        wrapper: MetadataWrapper = await ModuleWithTypeInfoFactory.create_module(
            prefixed_code
        )
        aliasing = UniqueAliasing()
        wrapper.visit(aliasing)
        self.assertEqual(expected_aliases, aliasing.aliases)
