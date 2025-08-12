# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from unittest import IsolatedAsyncioTestCase

from agent.logic.logic_py_smt_data_structure_generator import (
    LogicPySMTDataStructureGenerator,
)

from agent.symex.module_with_type_info_factory import ModuleWithTypeInfoFactory
from libcst import MetadataWrapper


class TestLogicPySMTDataStructureGenerator(IsolatedAsyncioTestCase):

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.maxDiff = None

    async def test_lipstick(self) -> None:
        await self.__test_harness(
            """
class Lipstick:
    has_velvet_finish: bool
    has_satin_finish: bool
    has_rosewood_in_description: bool
    creation_year: int
    length_in_cm: float
    is_refillable: bool

class LipstickSet:
    lipsticks: list[Lipstick]

class Universe:
    lipstick_sets: list[LipstickSet]

def premise(universe: Universe) -> None:
    rouge_dior_set = some(universe.lipstick_sets)

    # All velvet-finish lipsticks in the Rouge Dior set, Lunar New Year Limited Edition are refillable.
    for lipstick in rouge_dior_set.lipsticks:
        if lipstick.has_velvet_finish:
            assert lipstick.is_refillable

    # Lipsticks in the Rouge Dior set, Lunar New Year Limited Edition have either a velvet-finish or a satin-finish.
    for lipstick in rouge_dior_set.lipsticks:
        assert lipstick.has_velvet_finish or lipstick.has_satin_finish

    # No satin-finish lipsticks in the set do not have "rosewood" in its official description.
    for lipstick in rouge_dior_set.lipsticks:
        if lipstick.has_satin_finish:
            assert lipstick.has_rosewood_in_description

    # Lipstcks in the Rouge Dior set, Lunar New Year Limited Edition either does not have "rosewood" in its offical description or it has "rosewood" in its official description.
    for lipstick in rouge_dior_set.lipsticks:
        assert lipstick.has_rosewood_in_description or not lipstick.has_rosewood_in_description

    # ROUGE Dior Colored Lip Balm 999 is a lipstick in the set, and it either has "rosewood" in its official description or has a velvet finish.
    rouge_dior_colored_lip_balm_999 = some(rouge_dior_set.lipsticks)
    assert rouge_dior_colored_lip_balm_999.has_rosewood_in_description or rouge_dior_colored_lip_balm_999.has_velvet_finish

def conclusion(universe: Universe) -> None:
    rouge_dior_set = some(universe.lipstick_sets)

    # ROUGE Dior Colored Lip Balm 999 is a lipstick in the set...
    rouge_dior_colored_lip_balm_999 = some(rouge_dior_set.lipsticks)

    # ...and it has a satin finish and has "rosewood" in its official description.
    assert rouge_dior_colored_lip_balm_999.has_satin_finish
    assert rouge_dior_colored_lip_balm_999.has_rosewood_in_description
""",
            """; Lipstick
(declare-fun __attribute_Lipstick_has_velvet_finish (Int) Bool)
(declare-fun __attribute_Lipstick_has_satin_finish (Int) Bool)
(declare-fun __attribute_Lipstick_has_rosewood_in_description (Int) Bool)
(declare-fun __attribute_Lipstick_creation_year (Int) Int)
(declare-fun __attribute_Lipstick_length_in_cm (Int) Real)
(declare-fun __attribute_Lipstick_is_refillable (Int) Bool)

; LipstickSet
(declare-fun __attribute_LipstickSet_lipsticks_backing (Int Int) Int)
(declare-fun __attribute_LipstickSet_lipsticks_size (Int) Int)
(define-fun __attribute_LipstickSet_lipsticks ((this Int) (index Int)) Int (__attribute_LipstickSet_lipsticks_backing this (ite (and (>= index 0) (< index (__attribute_LipstickSet_lipsticks_size this))) index 0)))

; Universe
(declare-fun __attribute_Universe_lipstick_sets_backing (Int) Int)
(declare-fun __attribute_Universe_lipstick_sets_size () Int)
(define-fun __attribute_Universe_lipstick_sets ((index Int)) Int (__attribute_Universe_lipstick_sets_backing (ite (and (>= index 0) (< index __attribute_Universe_lipstick_sets_size)) index 0)))

""",
        )

    async def test_list(self) -> None:
        await self.__test_harness(
            """
class Club:
    members: list[Person]

class Person:
    pass

class Universe:
    clubs: list[Club]
    people: list[Person]
""",
            """; Club
(declare-fun __attribute_Club_members_backing (Int Int) Int)
(declare-fun __attribute_Club_members_size (Int) Int)
(define-fun __attribute_Club_members ((this Int) (index Int)) Int (__attribute_Club_members_backing this (ite (and (>= index 0) (< index (__attribute_Club_members_size this))) index 0)))

; Person

; Universe
(declare-fun __attribute_Universe_clubs_backing (Int) Int)
(declare-fun __attribute_Universe_clubs_size () Int)
(define-fun __attribute_Universe_clubs ((index Int)) Int (__attribute_Universe_clubs_backing (ite (and (>= index 0) (< index __attribute_Universe_clubs_size)) index 0)))
(declare-fun __attribute_Universe_people_backing (Int) Int)
(declare-fun __attribute_Universe_people_size () Int)
(define-fun __attribute_Universe_people ((index Int)) Int (__attribute_Universe_people_backing (ite (and (>= index 0) (< index __attribute_Universe_people_size)) index 0)))

""",
        )

    async def test_scalar(self) -> None:
        await self.__test_harness(
            """
class Person:
    name: str
    spouse: Person
    age: int
    score: float
    is_tall: bool

class Universe:
    people: list[Person]
""",
            """; Person
(declare-fun __attribute_Person_name (Int) String)
(declare-fun __attribute_Person_spouse (Int) Int)
(declare-fun __attribute_Person_age (Int) Int)
(declare-fun __attribute_Person_score (Int) Real)
(declare-fun __attribute_Person_is_tall (Int) Bool)

; Universe
(declare-fun __attribute_Universe_people_backing (Int) Int)
(declare-fun __attribute_Universe_people_size () Int)
(define-fun __attribute_Universe_people ((index Int)) Int (__attribute_Universe_people_backing (ite (and (>= index 0) (< index __attribute_Universe_people_size)) index 0)))

""",
        )

    async def test_inheritance(self) -> None:
        await self.__test_harness(
            """
class Shape:
    colour: str

class Circle:
    radius: float

class Square:
    length: float

class Universe:
    shapes: list[Shape]
""",
            """; Shape
(declare-fun __attribute_Shape_colour (Int) String)

; Circle
(declare-fun __attribute_Circle_radius (Int) Real)

; Square
(declare-fun __attribute_Square_length (Int) Real)

; Universe
(declare-fun __attribute_Universe_shapes_backing (Int) Int)
(declare-fun __attribute_Universe_shapes_size () Int)
(define-fun __attribute_Universe_shapes ((index Int)) Int (__attribute_Universe_shapes_backing (ite (and (>= index 0) (< index __attribute_Universe_shapes_size)) index 0)))

""",
        )

    async def test_unique(self) -> None:
        await self.__test_harness(
            """
class Person:
    name: Unique[str]
    hobbies: list[str]

class Universe:
    people: list[Person]
""",
            """; Person
(declare-fun __attribute_Person_name (Int) String)
(declare-fun __attribute_Person_hobbies_backing (Int Int) String)
(declare-fun __attribute_Person_hobbies_size (Int) Int)
(define-fun __attribute_Person_hobbies ((this Int) (index Int)) String (__attribute_Person_hobbies_backing this (ite (and (>= index 0) (< index (__attribute_Person_hobbies_size this))) index 0)))

; Universe
(declare-fun __attribute_Universe_people_backing (Int) Int)
(declare-fun __attribute_Universe_people_size () Int)
(define-fun __attribute_Universe_people ((index Int)) Int (__attribute_Universe_people_backing (ite (and (>= index 0) (< index __attribute_Universe_people_size)) index 0)))
(declare-fun __attribute_Universe_people_by_name (String) Int)
(assert
  (forall
    ((person_index Int))
    (let
      ((person (__attribute_Universe_people person_index)))
      (=
        person
        (__attribute_Universe_people_by_name (__attribute_Person_name person))
      )
    )
  )
)

""",
        )

    async def __test_harness(self, code: str, expected_harness: str) -> None:
        module: MetadataWrapper = await ModuleWithTypeInfoFactory.create_module(code)
        visitor = LogicPySMTDataStructureGenerator()
        module.visit(visitor)
        self.assertEqual(expected_harness, visitor.smt_harness)
