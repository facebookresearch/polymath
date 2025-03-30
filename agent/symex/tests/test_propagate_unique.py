# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from agent.symex.propagate_unique import PropagateUnique
from libcst.codemod import CodemodTest


class TestPropagateUnique(CodemodTest):
    TRANSFORM = PropagateUnique

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.maxDiff = None

    def test_inline(self) -> None:
        before = """
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
        """
        after = """
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
        """

        self.assertCodemod(before, after)
