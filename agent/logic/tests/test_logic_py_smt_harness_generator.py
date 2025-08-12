# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from unittest import IsolatedAsyncioTestCase

from agent.logic.logic_py_smt_harness_generator import LogicPySMTHarnessGenerator
from agent.logic.z3_conclusion_check_engine_strategy import _PYTHON_CODE_PREFIX

from agent.symex.module_with_type_info_factory import ModuleWithTypeInfoFactory

from libcst import MetadataWrapper


class TestLogicPySMTHarnessGenerator(IsolatedAsyncioTestCase):

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.maxDiff = None

    async def test_generate(self) -> None:
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


(assert
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
                    (__attribute_Lipstick_has_rosewood_in_description __0__2__lipstick)
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

    async def test_circular(self) -> None:
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

def conclusion(universe: Universe) -> None:
    assert len(universe.teams) < 10
""",
            """; Person
(declare-fun __attribute_Person_name (Int) String)
(declare-fun __attribute_Person_team (Int) Int)

; Team
(declare-fun __attribute_Team_members_backing (Int Int) Int)
(declare-fun __attribute_Team_members_size (Int) Int)
(define-fun __attribute_Team_members ((this Int) (index Int)) Int (__attribute_Team_members_backing this (ite (and (>= index 0) (< index (__attribute_Team_members_size this))) index 0)))

; Universe
(declare-fun __attribute_Universe_people_backing (Int) Int)
(declare-fun __attribute_Universe_people_size () Int)
(define-fun __attribute_Universe_people ((index Int)) Int (__attribute_Universe_people_backing (ite (and (>= index 0) (< index __attribute_Universe_people_size)) index 0)))
(declare-fun __attribute_Universe_teams_backing (Int) Int)
(declare-fun __attribute_Universe_teams_size () Int)
(define-fun __attribute_Universe_teams ((index Int)) Int (__attribute_Universe_teams_backing (ite (and (>= index 0) (< index __attribute_Universe_teams_size)) index 0)))


(assert
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
(assert
  (and
    (<
      __attribute_Universe_teams_size
      10
    )
  )
)

(check-sat)
(pop)

(assert
  (not
    (and
      (<
        __attribute_Universe_teams_size
        10
      )
    )
  )
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
    assert peter.age > 33
    assert peter.score == 12.75

def conclusion(universe: Universe) -> None:
    peter: Person = some(universe.people)
    assert peter.age > 30
""",
            """; Person
(declare-fun __attribute_Person_name (Int) String)
(declare-fun __attribute_Person_age (Int) Int)
(declare-fun __attribute_Person_score (Int) Real)

; Universe
(declare-fun __attribute_Universe_people_backing (Int) Int)
(declare-fun __attribute_Universe_people_size () Int)
(define-fun __attribute_Universe_people ((index Int)) Int (__attribute_Universe_people_backing (ite (and (>= index 0) (< index __attribute_Universe_people_size)) index 0)))


(assert
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
          (>
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
(assert
  (and
    (exists
      ((__0__peter_index Int))
      (let
        ((__0__peter (__attribute_Universe_people __0__peter_index)))
        (and
          (>
            (__attribute_Person_age __0__peter)
            30
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
            (>
              (__attribute_Person_age __0__peter)
              30
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

    async def test_inheritance(self) -> None:
        await self.__test_harness(
            """
class Shape:
    colour: str

class Circle(Shape):
    radius: float

class Square(Shape):
    length: float

class Universe:
    squares: list[Square]

def premise(universe: Universe) -> None:
    red_square: Square = some(universe.squares)
    assert red_square.colour == "red"

def conclusion(universe: Universe) -> None:
    red_square: Square = some(universe.squares)
    assert red_square.length > 10
""",
            """; Shape
(declare-fun __attribute_Shape_colour (Int) String)

; Circle
(declare-fun __attribute_Circle_radius (Int) Real)

; Square
(declare-fun __attribute_Square_length (Int) Real)

; Universe
(declare-fun __attribute_Universe_squares_backing (Int) Int)
(declare-fun __attribute_Universe_squares_size () Int)
(define-fun __attribute_Universe_squares ((index Int)) Int (__attribute_Universe_squares_backing (ite (and (>= index 0) (< index __attribute_Universe_squares_size)) index 0)))


(assert
  (and
    (exists
      ((__0__red_square_index Int))
      (let
        ((__0__red_square (__attribute_Universe_squares __0__red_square_index)))
        (and
          (=
            (__attribute_Shape_colour __0__red_square)
            "red"
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
      ((__0__red_square_index Int))
      (let
        ((__0__red_square (__attribute_Universe_squares __0__red_square_index)))
        (and
          (>
            (__attribute_Square_length __0__red_square)
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
        ((__0__red_square_index Int))
        (let
          ((__0__red_square (__attribute_Universe_squares __0__red_square_index)))
          (and
            (>
              (__attribute_Square_length __0__red_square)
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

    async def test_unique(self) -> None:
        await self.__test_harness(
            """
class Person:
    name: Unique[str]
    age: int

class Team:
    name: Unique[str]
    description: str

class Universe:
    people: list[Person]
    teams: list[Team]

def premise(universe: Universe) -> None:
    peter = some(universe.people)
    assert peter.name == "Peter"
    apex = some(universe.teams)
    assert apex.name == "Apex"
    assert peter in apex

def conclusion(universe: Universe) -> None:
    peter = some(universe.people)
    assert peter.name == "Peter"
    apex = some(universe.teams)
    assert apex.name == "Apex"
    assert not peter in apex
""",
            """; Person
(declare-fun __attribute_Person_name (Int) String)
(declare-fun __attribute_Person_age (Int) Int)

; Team
(declare-fun __attribute_Team_name (Int) String)
(declare-fun __attribute_Team_description (Int) String)

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
(declare-fun __attribute_Universe_teams_backing (Int) Int)
(declare-fun __attribute_Universe_teams_size () Int)
(define-fun __attribute_Universe_teams ((index Int)) Int (__attribute_Universe_teams_backing (ite (and (>= index 0) (< index __attribute_Universe_teams_size)) index 0)))
(declare-fun __attribute_Universe_teams_by_name (String) Int)
(assert
  (forall
    ((team_index Int))
    (let
      ((team (__attribute_Universe_teams team_index)))
      (=
        team
        (__attribute_Universe_teams_by_name (__attribute_Team_name team))
      )
    )
  )
)


(assert
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
          (exists
            ((__0__apex_index Int))
            (let
              ((__0__apex (__attribute_Universe_teams __0__apex_index)))
              (and
                (=
                  (__attribute_Team_name __0__apex)
                  "Apex"
                )
                (and
                  (exists
                    ((__0__0____logicpy_in_tmp_0_index Int))
                    (let
                      ((__0__0____logicpy_in_tmp_0 __0__apex__0__0____logicpy_in_tmp_0_index))
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
(assert
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
          (exists
            ((__0__apex_index Int))
            (let
              ((__0__apex (__attribute_Universe_teams __0__apex_index)))
              (and
                (=
                  (__attribute_Team_name __0__apex)
                  "Apex"
                )
                (not
                  (and
                    (exists
                      ((__0__0____logicpy_in_tmp_1_index Int))
                      (let
                        ((__0__0____logicpy_in_tmp_1 __0__apex__0__0____logicpy_in_tmp_1_index))
                        (and
                          (=
                            __0__0____logicpy_in_tmp_1
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
(pop)

(assert
  (not
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
            (exists
              ((__0__apex_index Int))
              (let
                ((__0__apex (__attribute_Universe_teams __0__apex_index)))
                (and
                  (=
                    (__attribute_Team_name __0__apex)
                    "Apex"
                  )
                  (not
                    (and
                      (exists
                        ((__0__0____logicpy_in_tmp_1_index Int))
                        (let
                          ((__0__0____logicpy_in_tmp_1 __0__apex__0__0____logicpy_in_tmp_1_index))
                          (and
                            (=
                              __0__0____logicpy_in_tmp_1
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
)
(check-sat)
""",
        )

    async def test_duplicate_attribute(self) -> None:
        await self.__test_harness(
            """
class Person:
    name: Unique[str]
    description: str

class Team:
    name: str

class Universe:
    people: list[Person]
    teams: list[Team]

def premise(universe: Universe) -> None:
    for person in universe.people:
        assert person.name != ""

    peter = some(universe.people)
    assert peter.name == "Peter"
    apex = some(universe.teams)
    assert apex.name == "Apex"

def conclusion(universe: Universe) -> None:
    peter = some(universe.people)
    assert peter.name == "Peter"
    apex = some(universe.teams)
    assert apex.name == "Apex"
""",
            """; Person
(declare-fun __attribute_Person_name (Int) String)
(declare-fun __attribute_Person_description (Int) String)

; Team
(declare-fun __attribute_Team_name (Int) String)

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
(declare-fun __attribute_Universe_teams_backing (Int) Int)
(declare-fun __attribute_Universe_teams_size () Int)
(define-fun __attribute_Universe_teams ((index Int)) Int (__attribute_Universe_teams_backing (ite (and (>= index 0) (< index __attribute_Universe_teams_size)) index 0)))


(assert
  (and
    (forall
      ((__0__0__person_index Int))
      (let
        ((__0__0__person (__attribute_Universe_people __0__0__person_index)))
        (and
          (not
            (=
              (__attribute_Person_name __0__0__person)
              ""
            )
          )
        )
      )
    )
    (exists
      ((__0__peter_index Int))
      (let
        ((__0__peter (__attribute_Universe_people __0__peter_index)))
        (and
          (=
            (__attribute_Person_name __0__peter)
            "Peter"
          )
          (exists
            ((__0__apex_index Int))
            (let
              ((__0__apex (__attribute_Universe_teams __0__apex_index)))
              (and
                (=
                  (__attribute_Team_name __0__apex)
                  "Apex"
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
      ((__0__peter_index Int))
      (let
        ((__0__peter (__attribute_Universe_people __0__peter_index)))
        (and
          (=
            (__attribute_Person_name __0__peter)
            "Peter"
          )
          (exists
            ((__0__apex_index Int))
            (let
              ((__0__apex (__attribute_Universe_teams __0__apex_index)))
              (and
                (=
                  (__attribute_Team_name __0__apex)
                  "Apex"
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
        ((__0__peter_index Int))
        (let
          ((__0__peter (__attribute_Universe_people __0__peter_index)))
          (and
            (=
              (__attribute_Person_name __0__peter)
              "Peter"
            )
            (exists
              ((__0__apex_index Int))
              (let
                ((__0__apex (__attribute_Universe_teams __0__apex_index)))
                (and
                  (=
                    (__attribute_Team_name __0__apex)
                    "Apex"
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

    async def test_wrestling(self) -> None:
        await self.__test_harness(
            """
class Person:
    name: Unique[str]
    description: str

class Stable:
    name: Unique[str]
    description: str
    leader: Person
    members: list[Person]

class Feud:
    stable1: Stable
    stable2: Stable

class Universe:
    people: list[Person]
    stables: list[Stable]
    feuds: list[Feud]

def premise(universe: Universe) -> None:
    diamond_mine = some(universe.stables)
    assert diamond_mine.name == "Diamond Mine"
    assert diamond_mine.description == "a professional wrestling stable formed in WWE"

    roderick_strong = some(universe.people)
    assert roderick_strong.name == "Roderick Strong"
    assert roderick_strong in diamond_mine.members
    assert roderick_strong == diamond_mine.leader

    creed_brother1 = some(universe.people)
    creed_brother2 = some(universe.people)
    assert creed_brother1.name == "Creed Brother 1"
    assert creed_brother2.name == "Creed Brother 2"
    assert creed_brother1 in diamond_mine.members
    assert creed_brother2 in diamond_mine.members

    ivy_nile = some(universe.people)
    assert ivy_nile.name == "Ivy Nile"
    assert ivy_nile in diamond_mine.members

    imperium = some(universe.stables)
    assert imperium.name == "Imperium"

    feud = some(universe.feuds)
    assert feud.stable1 == diamond_mine or feud.stable1 == imperium
    assert feud.stable2 == diamond_mine or feud.stable2 == imperium
    assert feud.stable1 != feud.stable2

def conclusion(universe: Universe) -> None:
    roderick_strong = some(universe.people)
    assert roderick_strong.name == "Roderick Strong"
    stable = some(universe.stables)
    assert roderick_strong == stable.leader
    assert stable.description == "a professional wrestling stable"
""",
            """; Person
(declare-fun __attribute_Person_name (Int) String)
(declare-fun __attribute_Person_description (Int) String)

; Stable
(declare-fun __attribute_Stable_name (Int) String)
(declare-fun __attribute_Stable_description (Int) String)
(declare-fun __attribute_Stable_leader (Int) Int)
(declare-fun __attribute_Stable_members_backing (Int Int) Int)
(declare-fun __attribute_Stable_members_size (Int) Int)
(define-fun __attribute_Stable_members ((this Int) (index Int)) Int (__attribute_Stable_members_backing this (ite (and (>= index 0) (< index (__attribute_Stable_members_size this))) index 0)))

; Feud
(declare-fun __attribute_Feud_stable1 (Int) Int)
(declare-fun __attribute_Feud_stable2 (Int) Int)

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
(declare-fun __attribute_Universe_stables_backing (Int) Int)
(declare-fun __attribute_Universe_stables_size () Int)
(define-fun __attribute_Universe_stables ((index Int)) Int (__attribute_Universe_stables_backing (ite (and (>= index 0) (< index __attribute_Universe_stables_size)) index 0)))
(declare-fun __attribute_Universe_stables_by_name (String) Int)
(assert
  (forall
    ((stable_index Int))
    (let
      ((stable (__attribute_Universe_stables stable_index)))
      (=
        stable
        (__attribute_Universe_stables_by_name (__attribute_Stable_name stable))
      )
    )
  )
)
(declare-fun __attribute_Universe_feuds_backing (Int) Int)
(declare-fun __attribute_Universe_feuds_size () Int)
(define-fun __attribute_Universe_feuds ((index Int)) Int (__attribute_Universe_feuds_backing (ite (and (>= index 0) (< index __attribute_Universe_feuds_size)) index 0)))


(assert
  (and
    (exists
      ((__0__diamond_mine_index Int))
      (let
        ((__0__diamond_mine (__attribute_Universe_stables __0__diamond_mine_index)))
        (and
          (=
            (__attribute_Stable_name __0__diamond_mine)
            "Diamond Mine"
          )
          (=
            (__attribute_Stable_description __0__diamond_mine)
            "a professional wrestling stable formed in WWE"
          )
          (exists
            ((__0__roderick_strong_index Int))
            (let
              ((__0__roderick_strong (__attribute_Universe_people __0__roderick_strong_index)))
              (and
                (=
                  (__attribute_Person_name __0__roderick_strong)
                  "Roderick Strong"
                )
                (and
                  (exists
                    ((__0__0____logicpy_in_tmp_0_index Int))
                    (let
                      ((__0__0____logicpy_in_tmp_0 (__attribute_Stable_members __0__diamond_mine __0__0____logicpy_in_tmp_0_index)))
                      (and
                        (=
                          __0__0____logicpy_in_tmp_0
                          __0__roderick_strong
                        )
                      )
                    )
                  )
                )
                (=
                  __0__roderick_strong
                  (__attribute_Stable_leader __0__diamond_mine)
                )
                (exists
                  ((__0__creed_brother1_index Int))
                  (let
                    ((__0__creed_brother1 (__attribute_Universe_people __0__creed_brother1_index)))
                    (and
                      (exists
                        ((__0__creed_brother2_index Int))
                        (let
                          ((__0__creed_brother2 (__attribute_Universe_people __0__creed_brother2_index)))
                          (and
                            (=
                              (__attribute_Person_name __0__creed_brother1)
                              "Creed Brother 1"
                            )
                            (=
                              (__attribute_Person_name __0__creed_brother2)
                              "Creed Brother 2"
                            )
                            (and
                              (exists
                                ((__0__1____logicpy_in_tmp_1_index Int))
                                (let
                                  ((__0__1____logicpy_in_tmp_1 (__attribute_Stable_members __0__diamond_mine __0__1____logicpy_in_tmp_1_index)))
                                  (and
                                    (=
                                      __0__1____logicpy_in_tmp_1
                                      __0__creed_brother1
                                    )
                                  )
                                )
                              )
                            )
                            (and
                              (exists
                                ((__0__2____logicpy_in_tmp_2_index Int))
                                (let
                                  ((__0__2____logicpy_in_tmp_2 (__attribute_Stable_members __0__diamond_mine __0__2____logicpy_in_tmp_2_index)))
                                  (and
                                    (=
                                      __0__2____logicpy_in_tmp_2
                                      __0__creed_brother2
                                    )
                                  )
                                )
                              )
                            )
                            (exists
                              ((__0__ivy_nile_index Int))
                              (let
                                ((__0__ivy_nile (__attribute_Universe_people __0__ivy_nile_index)))
                                (and
                                  (=
                                    (__attribute_Person_name __0__ivy_nile)
                                    "Ivy Nile"
                                  )
                                  (and
                                    (exists
                                      ((__0__3____logicpy_in_tmp_3_index Int))
                                      (let
                                        ((__0__3____logicpy_in_tmp_3 (__attribute_Stable_members __0__diamond_mine __0__3____logicpy_in_tmp_3_index)))
                                        (and
                                          (=
                                            __0__3____logicpy_in_tmp_3
                                            __0__ivy_nile
                                          )
                                        )
                                      )
                                    )
                                  )
                                  (exists
                                    ((__0__imperium_index Int))
                                    (let
                                      ((__0__imperium (__attribute_Universe_stables __0__imperium_index)))
                                      (and
                                        (=
                                          (__attribute_Stable_name __0__imperium)
                                          "Imperium"
                                        )
                                        (exists
                                          ((__0__feud_index Int))
                                          (let
                                            ((__0__feud (__attribute_Universe_feuds __0__feud_index)))
                                            (and
                                              (not
                                                (=
                                                  (=
                                                    (__attribute_Feud_stable1 __0__feud)
                                                    __0__diamond_mine
                                                  )
                                                  (=
                                                    (__attribute_Feud_stable1 __0__feud)
                                                    __0__imperium
                                                  )
                                                )
                                              )
                                              (not
                                                (=
                                                  (=
                                                    (__attribute_Feud_stable2 __0__feud)
                                                    __0__diamond_mine
                                                  )
                                                  (=
                                                    (__attribute_Feud_stable2 __0__feud)
                                                    __0__imperium
                                                  )
                                                )
                                              )
                                              (not
                                                (=
                                                  (__attribute_Feud_stable1 __0__feud)
                                                  (__attribute_Feud_stable2 __0__feud)
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
      ((__0__roderick_strong_index Int))
      (let
        ((__0__roderick_strong (__attribute_Universe_people __0__roderick_strong_index)))
        (and
          (=
            (__attribute_Person_name __0__roderick_strong)
            "Roderick Strong"
          )
          (exists
            ((__0__stable_index Int))
            (let
              ((__0__stable (__attribute_Universe_stables __0__stable_index)))
              (and
                (=
                  __0__roderick_strong
                  (__attribute_Stable_leader __0__stable)
                )
                (=
                  (__attribute_Stable_description __0__stable)
                  "a professional wrestling stable"
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
        ((__0__roderick_strong_index Int))
        (let
          ((__0__roderick_strong (__attribute_Universe_people __0__roderick_strong_index)))
          (and
            (=
              (__attribute_Person_name __0__roderick_strong)
              "Roderick Strong"
            )
            (exists
              ((__0__stable_index Int))
              (let
                ((__0__stable (__attribute_Universe_stables __0__stable_index)))
                (and
                  (=
                    __0__roderick_strong
                    (__attribute_Stable_leader __0__stable)
                  )
                  (=
                    (__attribute_Stable_description __0__stable)
                    "a professional wrestling stable"
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

    async def test_subscript(self) -> None:
        await self.__test_harness(
            """
class Person:
    name: Unique[str]
    age: int

class Universe:
    persons: list[Person]

def premise(universe: Universe) -> None:
    assert universe.persons[0] == universe.persons[0]
    assert universe.persons[0].age == 35
    assert universe.persons[0] == universe.persons[0]

    for person in universe.persons:
        if person.age > 30:
            assert person == universe.persons[0]
        if person == universe.persons[0]:
            assert person.age > 30
        if universe.persons[0] == person:
            assert person.age > 31
        if person == universe.persons[0]:
            assert person.age > 32
""",
            """; Person
(declare-fun __attribute_Person_name (Int) String)
(declare-fun __attribute_Person_age (Int) Int)

; Universe
(declare-fun __attribute_Universe_persons_backing (Int) Int)
(declare-fun __attribute_Universe_persons_size () Int)
(define-fun __attribute_Universe_persons ((index Int)) Int (__attribute_Universe_persons_backing (ite (and (>= index 0) (< index __attribute_Universe_persons_size)) index 0)))
(declare-fun __attribute_Universe_persons_by_name (String) Int)
(assert
  (forall
    ((person_index Int))
    (let
      ((person (__attribute_Universe_persons person_index)))
      (=
        person
        (__attribute_Universe_persons_by_name (__attribute_Person_name person))
      )
    )
  )
)


(assert
  (and
    (=
      (__attribute_Universe_persons 0)
      (__attribute_Universe_persons 0)
    )
    (=
      (__attribute_Person_age (__attribute_Universe_persons 0))
      35
    )
    (=
      (__attribute_Universe_persons 0)
      (__attribute_Universe_persons 0)
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
                (__attribute_Universe_persons 0)
              )
            )
          )
          (and
            (=>
              (=
                __0__0__person
                (__attribute_Universe_persons 0)
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
                (__attribute_Universe_persons 0)
                __0__0__person
              )
              (>
                (__attribute_Person_age __0__0__person)
                31
              )
            )
          )
          (and
            (=>
              (=
                __0__0__person
                (__attribute_Universe_persons 0)
              )
              (>
                (__attribute_Person_age __0__0__person)
                32
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
        harness: str = LogicPySMTHarnessGenerator.generate(module)
        self.assertEqual(expected_harness, harness)
