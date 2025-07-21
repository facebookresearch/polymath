# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from unittest import IsolatedAsyncioTestCase, skip

from agent.logic.logic_py_smt_constraint_generator import LogicPySMTConstraintGenerator
from agent.logic.logic_py_smt_data_structure_generator import (
    LogicPySMTDataStructureGenerator,
)
from agent.logic.z3_conclusion_check_engine_strategy import _PYTHON_CODE_PREFIX

from agent.symex.module_with_type_info_factory import ModuleWithTypeInfoFactory
from libcst import MetadataWrapper


class TestLogicPySMTConstraintGenerator(IsolatedAsyncioTestCase):

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
        other_lipstick = some(rouge_dior_set.lipsticks)
        assert lipstick.creation_year >= other_lipstick.creation_year

    # Lipsticks in the Rouge Dior set, Lunar New Year Limited Edition have either a velvet-finish or a satin-finish.
    for lipstick in rouge_dior_set.lipsticks:
        assert lipstick.has_velvet_finish or lipstick.has_satin_finish

    # No satin-finish lipsticks in the set do not have "rosewood" in its official description.
    for lipstick in rouge_dior_set.lipsticks:
        if lipstick.has_satin_finish:
            assert lipstick.has_rosewood_in_description and lipstick.has_satin_finish

    # Lipstcks in the Rouge Dior set, Lunar New Year Limited Edition either does not have "rosewood" in its offical description or it has "rosewood" in its official description.
    for lipstick in rouge_dior_set.lipsticks:
        assert lipstick.has_rosewood_in_description or not lipstick.has_rosewood_in_description

    # ROUGE Dior Colored Lip Balm 999 is a lipstick in the set, and it either has "rosewood" in its official description or has a velvet finish.
    rouge_dior_colored_lip_balm_999 = some(rouge_dior_set.lipsticks)
    assert rouge_dior_colored_lip_balm_999.has_rosewood_in_description or rouge_dior_colored_lip_balm_999.has_velvet_finish

    assert rouge_dior_colored_lip_balm_999.has_rosewood_in_description if rouge_dior_colored_lip_balm_999.has_velvet_finish else True

def conclusion(universe: Universe) -> None:
    rouge_dior_set = some(universe.lipstick_sets)

    # ROUGE Dior Colored Lip Balm 999 is a lipstick in the set...
    rouge_dior_colored_lip_balm_999 = some(rouge_dior_set.lipsticks)

    # ...and it has a satin finish and has "rosewood" in its official description.
    assert rouge_dior_colored_lip_balm_999.has_satin_finish
    assert rouge_dior_colored_lip_balm_999.has_rosewood_in_description

    # All lipsticks have a satin finish.
    for lipstick in rouge_dior_set.lipsticks:
        assert lipstick.has_satin_finish
""",
            """(assert
  (and
    (exists
      ((__0__rouge_dior_set_index Int))
      (let
        ((__0__rouge_dior_set (__attribute_Universe_lipstick_sets __0__rouge_dior_set_index)))
        (and
          (forall
            ((__0__0__lipstick_index Int))
            (let
              ((__0__0__lipstick (__attribute_LipstickSet_lipsticks __0__rouge_dior_set __0__0__lipstick_index)))
              (and
                (and
                  (=>
                    (__attribute_Lipstick_has_velvet_finish __0__0__lipstick)
                    (__attribute_Lipstick_is_refillable __0__0__lipstick)
                  )
                )
                (exists
                  ((__0__0__0__other_lipstick_index Int))
                  (let
                    ((__0__0__0__other_lipstick (__attribute_LipstickSet_lipsticks __0__rouge_dior_set __0__0__0__other_lipstick_index)))
                    (and
                      (>=
                        (__attribute_Lipstick_creation_year __0__0__lipstick)
                        (__attribute_Lipstick_creation_year __0__0__0__other_lipstick)
                      )
                    )
                  )
                )
              )
            )
          )
          (forall
            ((__0__1__lipstick_index Int))
            (let
              ((__0__1__lipstick (__attribute_LipstickSet_lipsticks __0__rouge_dior_set __0__1__lipstick_index)))
              (and
                (not
                  (=
                    (__attribute_Lipstick_has_velvet_finish __0__1__lipstick)
                    (__attribute_Lipstick_has_satin_finish __0__1__lipstick)
                  )
                )
              )
            )
          )
          (forall
            ((__0__2__lipstick_index Int))
            (let
              ((__0__2__lipstick (__attribute_LipstickSet_lipsticks __0__rouge_dior_set __0__2__lipstick_index)))
              (and
                (and
                  (=>
                    (__attribute_Lipstick_has_satin_finish __0__2__lipstick)
                    (and
                      (__attribute_Lipstick_has_rosewood_in_description __0__2__lipstick)
                      (__attribute_Lipstick_has_satin_finish __0__2__lipstick)
                    )
                  )
                )
              )
            )
          )
          (forall
            ((__0__3__lipstick_index Int))
            (let
              ((__0__3__lipstick (__attribute_LipstickSet_lipsticks __0__rouge_dior_set __0__3__lipstick_index)))
              (and
                (not
                  (=
                    (__attribute_Lipstick_has_rosewood_in_description __0__3__lipstick)
                    (not
                      (__attribute_Lipstick_has_rosewood_in_description __0__3__lipstick)
                    )
                  )
                )
              )
            )
          )
          (exists
            ((__0__rouge_dior_colored_lip_balm_999_index Int))
            (let
              ((__0__rouge_dior_colored_lip_balm_999 (__attribute_LipstickSet_lipsticks __0__rouge_dior_set __0__rouge_dior_colored_lip_balm_999_index)))
              (and
                (not
                  (=
                    (__attribute_Lipstick_has_rosewood_in_description __0__rouge_dior_colored_lip_balm_999)
                    (__attribute_Lipstick_has_velvet_finish __0__rouge_dior_colored_lip_balm_999)
                  )
                )
                (ite
                  (__attribute_Lipstick_has_velvet_finish __0__rouge_dior_colored_lip_balm_999)
                  (__attribute_Lipstick_has_rosewood_in_description __0__rouge_dior_colored_lip_balm_999)
                  true
                )
              )
            )
          )
        )
      )
    )
  )
)

(check-sat)

(push)
(assert
  (and
    (exists
      ((__0__rouge_dior_set_index Int))
      (let
        ((__0__rouge_dior_set (__attribute_Universe_lipstick_sets __0__rouge_dior_set_index)))
        (and
          (exists
            ((__0__rouge_dior_colored_lip_balm_999_index Int))
            (let
              ((__0__rouge_dior_colored_lip_balm_999 (__attribute_LipstickSet_lipsticks __0__rouge_dior_set __0__rouge_dior_colored_lip_balm_999_index)))
              (and
                (__attribute_Lipstick_has_satin_finish __0__rouge_dior_colored_lip_balm_999)
                (__attribute_Lipstick_has_rosewood_in_description __0__rouge_dior_colored_lip_balm_999)
                (forall
                  ((__0__0__lipstick_index Int))
                  (let
                    ((__0__0__lipstick (__attribute_LipstickSet_lipsticks __0__rouge_dior_set __0__0__lipstick_index)))
                    (and
                      (__attribute_Lipstick_has_satin_finish __0__0__lipstick)
                    )
                  )
                )
              )
            )
          )
        )
      )
    )
  )
)

(check-sat)
(pop)

(assert
  (not
    (and
      (exists
        ((__0__rouge_dior_set_index Int))
        (let
          ((__0__rouge_dior_set (__attribute_Universe_lipstick_sets __0__rouge_dior_set_index)))
          (and
            (exists
              ((__0__rouge_dior_colored_lip_balm_999_index Int))
              (let
                ((__0__rouge_dior_colored_lip_balm_999 (__attribute_LipstickSet_lipsticks __0__rouge_dior_set __0__rouge_dior_colored_lip_balm_999_index)))
                (and
                  (__attribute_Lipstick_has_satin_finish __0__rouge_dior_colored_lip_balm_999)
                  (__attribute_Lipstick_has_rosewood_in_description __0__rouge_dior_colored_lip_balm_999)
                  (forall
                    ((__0__0__lipstick_index Int))
                    (let
                      ((__0__0__lipstick (__attribute_LipstickSet_lipsticks __0__rouge_dior_set __0__0__lipstick_index)))
                      (and
                        (__attribute_Lipstick_has_satin_finish __0__0__lipstick)
                      )
                    )
                  )
                )
              )
            )
          )
        )
      )
    )
  )
)
(check-sat)
""",
        )

    async def test_not_equal(self) -> None:
        await self.__test_harness(
            """
class Wrestler:
    pass

class Universe:
    wrestlers: list[Wrestler]

def premise(universe: Universe) -> None:
    hulk = some(universe.wrestlers)
    ninja = some(universe.wrestlers)
    assert hulk != ninja
""",
            """(assert
  (and
    (exists
      ((__0__hulk_index Int))
      (let
        ((__0__hulk (__attribute_Universe_wrestlers __0__hulk_index)))
        (and
          (exists
            ((__0__ninja_index Int))
            (let
              ((__0__ninja (__attribute_Universe_wrestlers __0__ninja_index)))
              (and
                (not
                  (=
                    __0__hulk
                    __0__ninja
                  )
                )
              )
            )
          )
        )
      )
    )
  )
)

(check-sat)

(push)

(check-sat)
(pop)

(assert
)
(check-sat)
""",
        )

    async def test_chained_boolean_operation(self) -> None:
        await self.__test_harness(
            """
class Wrestler:
    is_tall: bool

class Universe:
    wrestlers: list[Wrestler]

def premise(universe: Universe) -> None:
    hulk = some(universe.wrestlers)
    ninja = some(universe.wrestlers)
    undertaker = some(universe.wrestlers)
    assert hulk.is_tall or ninja.is_tall or undertaker.is_tall
""",
            """(assert
  (and
    (exists
      ((__0__hulk_index Int))
      (let
        ((__0__hulk (__attribute_Universe_wrestlers __0__hulk_index)))
        (and
          (exists
            ((__0__ninja_index Int))
            (let
              ((__0__ninja (__attribute_Universe_wrestlers __0__ninja_index)))
              (and
                (exists
                  ((__0__undertaker_index Int))
                  (let
                    ((__0__undertaker (__attribute_Universe_wrestlers __0__undertaker_index)))
                    (and
                      (not
                        (=
                          (not
                            (=
                              (__attribute_Wrestler_is_tall __0__hulk)
                              (__attribute_Wrestler_is_tall __0__ninja)
                            )
                          )
                          (__attribute_Wrestler_is_tall __0__undertaker)
                        )
                      )
                    )
                  )
                )
              )
            )
          )
        )
      )
    )
  )
)

(check-sat)

(push)

(check-sat)
(pop)

(assert
)
(check-sat)
""",
        )

    async def test_chained_comparison(self) -> None:
        await self.__test_harness(
            """
class Wrestler:
    height: int

class Universe:
    wrestlers: list[Wrestler]

def premise(universe: Universe) -> None:
    hulk = some(universe.wrestlers)
    ninja = some(universe.wrestlers)
    undertaker = some(universe.wrestlers)
    john = some(universe.wrestlers)
    assert hulk.height != ninja.height < undertaker.height > john.height
""",
            """(assert
  (and
    (exists
      ((__0__hulk_index Int))
      (let
        ((__0__hulk (__attribute_Universe_wrestlers __0__hulk_index)))
        (and
          (exists
            ((__0__ninja_index Int))
            (let
              ((__0__ninja (__attribute_Universe_wrestlers __0__ninja_index)))
              (and
                (exists
                  ((__0__undertaker_index Int))
                  (let
                    ((__0__undertaker (__attribute_Universe_wrestlers __0__undertaker_index)))
                    (and
                      (exists
                        ((__0__john_index Int))
                        (let
                          ((__0__john (__attribute_Universe_wrestlers __0__john_index)))
                          (and
                            (and
                              (not
                                (=
                                  (__attribute_Wrestler_height __0__hulk)
                                  (__attribute_Wrestler_height __0__ninja)
                                )
                              )
                              (<
                                (__attribute_Wrestler_height __0__ninja)
                                (__attribute_Wrestler_height __0__undertaker)
                              )
                              (>
                                (__attribute_Wrestler_height __0__undertaker)
                                (__attribute_Wrestler_height __0__john)
                              )
                            )
                          )
                        )
                      )
                    )
                  )
                )
              )
            )
          )
        )
      )
    )
  )
)

(check-sat)

(push)

(check-sat)
(pop)

(assert
)
(check-sat)
""",
        )

    async def test_len(self) -> None:
        await self.__test_harness(
            """
class Person:
    name: str
    team: Team

class Team:
    members: list[Person]

class Universe:
    people: list[Person]
    teams: list[Team]

def premise(universe: Universe) -> None:
    assert len(universe.teams) == 7

    team = some(universe.teams)
    assert len(team.members) == 10

    person = some(universe.people)
    assert len(person.team.members) == 99
""",
            """(assert
  (and
    (=
      __attribute_Universe_teams_size
      7
    )
    (exists
      ((__0__team_index Int))
      (let
        ((__0__team (__attribute_Universe_teams __0__team_index)))
        (and
          (=
            (__attribute_Team_members_size __0__team)
            10
          )
          (exists
            ((__0__person_index Int))
            (let
              ((__0__person (__attribute_Universe_people __0__person_index)))
              (and
                (=
                  (__attribute_Team_members_size (__attribute_Person_team __0__person))
                  99
                )
              )
            )
          )
        )
      )
    )
  )
)

(check-sat)

(push)

(check-sat)
(pop)

(assert
)
(check-sat)
""",
        )

    async def test_literals(self) -> None:
        await self.__test_harness(
            """
class Person:
    name: str
    age: int
    score: float

class Universe:
    people: list[Person]

def premise(universe: Universe) -> None:
    peter: Person = some(universe.people)
    assert peter.name == "Peter"
    assert peter.age == 33
    assert peter.score == 12.75
""",
            """(assert
  (and
    (exists
      ((__0__peter_index Int))
      (let
        ((__0__peter (__attribute_Universe_people __0__peter_index)))
        (and
          (=
            (__attribute_Person_name __0__peter)
            "Peter"
          )
          (=
            (__attribute_Person_age __0__peter)
            33
          )
          (=
            (__attribute_Person_score __0__peter)
            12.75
          )
        )
      )
    )
  )
)

(check-sat)

(push)

(check-sat)
(pop)

(assert
)
(check-sat)
""",
        )

    async def test_in_operator(self) -> None:
        await self.__test_harness(
            """
class Person:
    name: str
    age: int
    score: float

class Team:
    members: list[Person]

class Universe:
    people: list[Person]
    teams: list[Team]

def premise(universe: Universe) -> None:
    peter: Person = some(universe.people)
    fair: Team = some(universe.teams)
    assert peter in fair.members
""",
            """(assert
  (and
    (exists
      ((__0__peter_index Int))
      (let
        ((__0__peter (__attribute_Universe_people __0__peter_index)))
        (and
          (exists
            ((__0__fair_index Int))
            (let
              ((__0__fair (__attribute_Universe_teams __0__fair_index)))
              (and
                (and
                  (exists
                    ((__0__0____logicpy_in_tmp_0_index Int))
                    (let
                      ((__0__0____logicpy_in_tmp_0 (__attribute_Team_members __0__fair __0__0____logicpy_in_tmp_0_index)))
                      (and
                        (=
                          __0__0____logicpy_in_tmp_0
                          __0__peter
                        )
                      )
                    )
                  )
                )
              )
            )
          )
        )
      )
    )
  )
)

(check-sat)

(push)

(check-sat)
(pop)

(assert
)
(check-sat)
""",
        )

    async def test_not_in_operator(self) -> None:
        await self.__test_harness(
            """
class Person:
    name: str
    age: int
    score: float

class Team:
    members: list[Person]

class Universe:
    people: list[Person]
    teams: list[Team]

def premise(universe: Universe) -> None:
    peter: Person = some(universe.people)
    fair: Team = some(universe.teams)
    assert peter not in fair.members
""",
            """(assert
  (and
    (exists
      ((__0__peter_index Int))
      (let
        ((__0__peter (__attribute_Universe_people __0__peter_index)))
        (and
          (exists
            ((__0__fair_index Int))
            (let
              ((__0__fair (__attribute_Universe_teams __0__fair_index)))
              (and
                (not
                  (and
                    (exists
                      ((__0__0____logicpy_in_tmp_0_index Int))
                      (let
                        ((__0__0____logicpy_in_tmp_0 (__attribute_Team_members __0__fair __0__0____logicpy_in_tmp_0_index)))
                        (and
                          (=
                            __0__0____logicpy_in_tmp_0
                            __0__peter
                          )
                        )
                      )
                    )
                  )
                )
              )
            )
          )
        )
      )
    )
  )
)

(check-sat)

(push)

(check-sat)
(pop)

(assert
)
(check-sat)
""",
        )

    async def test_global(self) -> None:
        await self.__test_harness(
            """
class Person:
    age: int

class Universe:
    people: list[Person]

def premise(universe: Universe) -> None:
    peter: Person = some(universe.people)
    assert peter.age > 20

def conclusion(universe: Universe) -> None:
    peter: Person = some(universe.people)
    assert peter.age < 10
""",
            """(assert
  (and
    (exists
      ((__0__peter_index Int))
      (let
        ((__0__peter (__attribute_Universe_people __0__peter_index)))
        (and
          (>
            (__attribute_Person_age __0__peter)
            20
          )
        )
      )
    )
  )
)

(check-sat)

(push)
(assert
  (and
    (exists
      ((__0__peter_index Int))
      (let
        ((__0__peter (__attribute_Universe_people __0__peter_index)))
        (and
          (<
            (__attribute_Person_age __0__peter)
            10
          )
        )
      )
    )
  )
)

(check-sat)
(pop)

(assert
  (not
    (and
      (exists
        ((__0__peter_index Int))
        (let
          ((__0__peter (__attribute_Universe_people __0__peter_index)))
          (and
            (<
              (__attribute_Person_age __0__peter)
              10
            )
          )
        )
      )
    )
  )
)
(check-sat)
""",
        )

    async def test_recursion_bug(self) -> None:
        await self.__test_harness(
            """
class Being:
    name: Unique[str]

class Virus:
    name: Unique[str]
    occurs_in: list[Being]

class Animal:
    name: Unique[str]
    is_mammal: bool

class Human:
    name: Unique[str]

class Symptom:
    name: Unique[str]

class Disease:
    name: Unique[str]
    symptoms: list[Symptom]
    causes_tiredness: bool

class Universe:
    beings: list[Being]
    viruses: list[Virus]
    animals: list[Animal]
    humans: list[Human]
    symptoms: list[Symptom]
    diseases: list[Disease]

def premise(universe: Universe) -> None:
    # When the Monkeypox virus occurs in a being, it may get Monkeypox.
    for being in universe.beings:
        for virus in universe.viruses:
            if virus.name == "Monkeypox" and being in virus.occurs_in:
                disease = some(universe.diseases)
                assert disease.name == "Monkeypox"


def conclusion(universe: Universe) -> None:
    # There is an animal.
    assert len(universe.animals) > 0
""",
            """(assert
  (and
    (forall
      ((__0__0__being_index Int))
      (let
        ((__0__0__being (__attribute_Universe_beings __0__0__being_index)))
        (and
          (forall
            ((__0__0__0__0__virus_index Int))
            (let
              ((__0__0__0__0__virus (__attribute_Universe_viruses __0__0__0__0__virus_index)))
              (and
                (and
                  (exists
                    ((__0__0__0__0__0__0__0__disease_index Int))
                    (let
                      ((__0__0__0__0__0__0__0__disease (__attribute_Universe_diseases __0__0__0__0__0__0__0__disease_index)))
                      (and
                        (=>
                          (and
                            (=
                              (__attribute_Virus_name __0__0__0__0__virus)
                              "Monkeypox"
                            )
                            (and
                              (exists
                                ((__0__0__0__0__0__0__0__0____logicpy_in_tmp_0_index Int))
                                (let
                                  ((__0__0__0__0__0__0__0__0____logicpy_in_tmp_0 (__attribute_Virus_occurs_in __0__0__0__0__virus __0__0__0__0__0__0__0__0____logicpy_in_tmp_0_index)))
                                  (and
                                    (=
                                      __0__0__0__0__0__0__0__0____logicpy_in_tmp_0
                                      __0__0__being
                                    )
                                  )
                                )
                              )
                            )
                          )
                          (=
                            (__attribute_Disease_name __0__0__0__0__0__0__0__disease)
                            "Monkeypox"
                          )
                        )
                      )
                    )
                  )
                )
              )
            )
          )
        )
      )
    )
  )
)

(check-sat)

(push)
(assert
  (and
    (>
      __attribute_Universe_animals_size
      0
    )
  )
)

(check-sat)
(pop)

(assert
  (not
    (and
      (>
        __attribute_Universe_animals_size
        0
      )
    )
  )
)
(check-sat)
""",
        )

    async def test_pass(self) -> None:
        await self.__test_harness(
            """
class Virus:
    is_dangerous: bool

class Universe:
    viruses: list[virus]

def premise(universe: Universe) -> None:
    for virus in viruses:
        if virus in universe.viruses:
            pass
""",
            """(assert
  (and
    (forall
      ((__0__0__virus_index Int))
      (let
        ((__0__0__virus viruses__0__0__virus_index))
        (and
          (and
            (=>
              (and
                (exists
                  ((__0__0__0__0__0__0____logicpy_in_tmp_0_index Int))
                  (let
                    ((__0__0__0__0__0__0____logicpy_in_tmp_0 (__attribute_Universe_viruses __0__0__0__0__0__0____logicpy_in_tmp_0_index)))
                    (and
                      (=
                        __0__0__0__0__0__0____logicpy_in_tmp_0
                        __0__0__virus
                      )
                    )
                  )
                )
              )
              true
            )
          )
        )
      )
    )
  )
)

(check-sat)

(push)

(check-sat)
(pop)

(assert
)
(check-sat)
""",
        )

    async def test_lightweight_id(self) -> None:
        await self.__test_harness(
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
    for person in universe.persons:
        if person.age > 30:
            assert person.name == "Peter"
        if person.name == "Peter":
            assert person.age > 30
        if person.name == peter.name:
            assert person.age > 31
""",
            """(assert
  (and
    (=
      (__attribute_Person_name 0)
      "Peter"
    )
    (=
      (__attribute_Person_age 0)
      35
    )
    (forall
      ((__0__0__person_index Int))
      (let
        ((__0__0__person (__attribute_Universe_persons __0__0__person_index)))
        (and
          (and
            (=>
              (>
                (__attribute_Person_age __0__0__person)
                30
              )
              (=
                __0__0__person
                0
              )
            )
          )
          (and
            (=>
              (=
                __0__0__person
                0
              )
              (>
                (__attribute_Person_age __0__0__person)
                30
              )
            )
          )
          (and
            (=>
              (=
                __0__0__person
                0
              )
              (>
                (__attribute_Person_age __0__0__person)
                31
              )
            )
          )
        )
      )
    )
  )
)

(check-sat)

(push)

(check-sat)
(pop)

(assert
)
(check-sat)
""",
        )

    async def __test_harness(self, code: str, expected_harness: str) -> None:
        harness: str = _PYTHON_CODE_PREFIX + code
        module: MetadataWrapper = await ModuleWithTypeInfoFactory.create_module(harness)
        data_structure = LogicPySMTDataStructureGenerator()
        module.visit(data_structure)
        constraints = LogicPySMTConstraintGenerator(
            data_structure.ancestors_per_class, data_structure.smt_attributes
        )
        module.visit(constraints)
        self.assertEqual(expected_harness, constraints.smt_harness)
