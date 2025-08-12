# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

from agent.logic.agent import LogicAgent

from dotenv import load_dotenv


load_dotenv()


class TestLogicPyCHarnessGenerator(TestCase):

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.maxDiff = None

    def test_extract_code(self) -> None:
        self.assertEqual(
            """class House:
    id: Unique[Domain[int, range(1, 7)]]
    name: Unique[Domain[str, "Alice", "Eric", "Peter", "Carol", "Bob", "Arnold"]]
    smoothie: Unique[Domain[str, "watermelon", "blueberry", "desert", "cherry", "dragonfruit", "lime"]]
    lunch: Unique[Domain[str, "stew", "pizza", "grilled cheese", "stir fry", "soup", "spaghetti"]]
    phone: Unique[Domain[str, "google pixel 6", "iphone 13", "xiaomi mi 11", "huawei p50", "samsung galaxy s21", "oneplus 9"]]
    car: Unique[Domain[str, "tesla model 3", "honda civic", "toyota camry", "ford f150", "chevrolet silverado", "bmw 3 series"]]
    house_style: Unique[Domain[str, "craftsman", "ranch", "modern", "victorian", "mediterranean", "colonial"]]

class Solution:
    houses: list[House, 6]
""",
            LogicAgent._LogicAgent__extract_code(  # pyright: ignore
                """Based on the puzzle, I define the data type for a valid puzzle solution as follows:

```
class House:
    id: Unique[Domain[int, range(1, 7)]]
    name: Unique[Domain[str, "Alice", "Eric", "Peter", "Carol", "Bob", "Arnold"]]
    smoothie: Unique[Domain[str, "watermelon", "blueberry", "desert", "cherry", "dragonfruit", "lime"]]
    lunch: Unique[Domain[str, "stew", "pizza", "grilled cheese", "stir fry", "soup", "spaghetti"]]
    phone: Unique[Domain[str, "google pixel 6", "iphone 13", "xiaomi mi 11", "huawei p50", "samsung galaxy s21", "oneplus 9"]]
    car: Unique[Domain[str, "tesla model 3", "honda civic", "toyota camry", "ford f150", "chevrolet silverado", "bmw 3 series"]]
    house_style: Unique[Domain[str, "craftsman", "ranch", "modern", "victorian", "mediterranean", "colonial"]]

class Solution:
    houses: list[House, 6]
```

This data structure captures the unique attributes of each house, with each field marked as `Unique` to ensure that no two houses have the same value for that attribute. The `Domain` constraints specify the allowed values for each field. The `Solution` class contains a list of 6 `House` objects, representing the 6 houses in the puzzle."""
            ),
        )

    def test_extract_code_with_language(self) -> None:
        self.assertEqual(
            """class House:
    id: Unique[Domain[int, range(1, 7)]]
    name: Unique[Domain[str, "Alice", "Eric", "Peter", "Carol", "Bob", "Arnold"]]
    smoothie: Unique[Domain[str, "watermelon", "blueberry", "desert", "cherry", "dragonfruit", "lime"]]
    lunch: Unique[Domain[str, "stew", "pizza", "grilled cheese", "stir fry", "soup", "spaghetti"]]
    phone: Unique[Domain[str, "google pixel 6", "iphone 13", "xiaomi mi 11", "huawei p50", "samsung galaxy s21", "oneplus 9"]]
    car: Unique[Domain[str, "tesla model 3", "honda civic", "toyota camry", "ford f150", "chevrolet silverado", "bmw 3 series"]]
    house_style: Unique[Domain[str, "craftsman", "ranch", "modern", "victorian", "mediterranean", "colonial"]]

class Solution:
    houses: list[House, 6]
""",
            LogicAgent._LogicAgent__extract_code(  # pyright: ignore
                """Based on the puzzle, I define the data type for a valid puzzle solution as follows:

```python
class House:
    id: Unique[Domain[int, range(1, 7)]]
    name: Unique[Domain[str, "Alice", "Eric", "Peter", "Carol", "Bob", "Arnold"]]
    smoothie: Unique[Domain[str, "watermelon", "blueberry", "desert", "cherry", "dragonfruit", "lime"]]
    lunch: Unique[Domain[str, "stew", "pizza", "grilled cheese", "stir fry", "soup", "spaghetti"]]
    phone: Unique[Domain[str, "google pixel 6", "iphone 13", "xiaomi mi 11", "huawei p50", "samsung galaxy s21", "oneplus 9"]]
    car: Unique[Domain[str, "tesla model 3", "honda civic", "toyota camry", "ford f150", "chevrolet silverado", "bmw 3 series"]]
    house_style: Unique[Domain[str, "craftsman", "ranch", "modern", "victorian", "mediterranean", "colonial"]]

class Solution:
    houses: list[House, 6]
```

This data structure captures the unique attributes of each house, with each field marked as `Unique` to ensure that no two houses have the same value for that attribute. The `Domain` constraints specify the allowed values for each field. The `Solution` class contains a list of 6 `House` objects, representing the 6 houses in the puzzle."""
            ),
        )

    def test_extract_concatenated_code(self) -> None:
        self.assertEqual(
            """def validate(solution: PuzzleSolution) -> None:
    # Clue 1: The person who loves pop music is the cat lover.
    pop_lover = nondet(solution.houses)
    assume(pop_lover.music_genre == "pop")
    assert pop_lover.animal == "cat"

    # Clue 2: The rabbit owner is directly left of the person whose mother's name is Aniya.
    rabbit_owner = nondet(solution.houses)
    aniya_child = nondet(solution.houses)
    assume(rabbit_owner.animal == "rabbit")
    assume(aniya_child.mother == "Aniya")
    assert rabbit_owner.house_number == aniya_child.house_number - 1

    # Clue 3: The person whose mother's name is Holly is directly left of Carol.
    holly_child = nondet(solution.houses)
    carol = nondet(solution.houses)
    assume(holly_child.mother == "Holly")
    assume(carol.name == "Carol")
    assert holly_child.house_number == carol.house_number - 1

    # Clue 4: The person whose mother's name is Holly is the person's child is named Alice.
    holly_child = nondet(solution.houses)
    assume(holly_child.mother == "Holly")
    assert holly_child.child == "Alice"

    # Clue 5: The person whose mother's name is Holly is the person who loves classical music.
    holly_child = nondet(solution.houses)
    assume(holly_child.mother == "Holly")
    assert holly_child.music_genre == "classical"

    # Clue 6: The person who loves jazz music is the person whose mother's name is Sarah.
    jazz_lover = nondet(solution.houses)
    assume(jazz_lover.music_genre == "jazz")
    assert jazz_lover.mother == "Sarah"

    # Clue 7: The person's child is named Meredith is somewhere to the right of the person whose mother's name is Aniya.
    meredith_parent = nondet(solution.houses)
    aniya_child = nondet(solution.houses)
    assume(meredith_parent.child == "Meredith")
    assume(aniya_child.mother == "Aniya")
    assert meredith_parent.house_number > aniya_child.house_number

    # Clue 8: The person who is super tall is the person whose mother's name is Holly.
    super_tall_person = nondet(solution.houses)
    assume(super_tall_person.height == "super tall")
    assert super_tall_person.mother == "Holly"

    # Clue 9: The person who is the mother of Timothy is Bob.
    timothy_parent = nondet(solution.houses)
    assume(timothy_parent.child == "Timothy")
    assert timothy_parent.name == "Bob"

    # Clue 10: The person who is very short is somewhere to the left of the person whose mother's name is Aniya.
    very_short_person = nondet(solution.houses)
    aniya_child = nondet(solution.houses)
    assume(very_short_person.height == "very short")
    assume(aniya_child.mother == "Aniya")
    assert very_short_person.house_number < aniya_child.house_number

    # Clue 11: Eric is the fish enthusiast.
    eric = nondet(solution.houses)
    assume(eric.name == "Eric")
    assert eric.animal == "fish"

    # Clue 12: The person's child is named Samantha is somewhere to the right of the person who is very tall.
    samantha_parent = nondet(solution.houses)
    very_tall_person = nondet(solution.houses)
    assume(samantha_parent.child == "Samantha")
    assume(very_tall_person.height == "very tall")
    assert samantha_parent.house_number > very_tall_person.house_number

    # Clue 13: The person who loves rock music is the person whose mother's name is Janelle.
    rock_lover = nondet(solution.houses)
    assume(rock_lover.music_genre == "rock")
    assert rock_lover.mother == "Janelle"

    # Clue 14: There is one house between the person who keeps horses and the person's child is named Meredith.
    horse_owner = nondet(solution.houses)
    meredith_parent = nondet(solution.houses)
    assume(horse_owner.animal == "horse")
    assume(meredith_parent.child == "Meredith")
    assert __CPROVER_abs(horse_owner.house_number - meredith_parent.house_number) == 2

    # Clue 15: The person's child is named Bella is somewhere to the right of Peter.
    bella_parent = nondet(solution.houses)
    peter = nondet(solution.houses)
    assume(bella_parent.child == "Bella")
    assume(peter.name == "Peter")
    assert bella_parent.house_number > peter.house_number

    # Clue 16: The fish enthusiast is somewhere to the left of the bird keeper.
    fish_enthusiast = nondet(solution.houses)
    bird_keeper = nondet(solution.houses)
    assume(fish_enthusiast.animal == "fish")
    assume(bird_keeper.animal == "bird")
    assert fish_enthusiast.house_number < bird_keeper.house_number

    # Clue 17: The fish enthusiast is somewhere to the right of the person's child is named Alice.
    fish_enthusiast = nondet(solution.houses)
    alice_parent = nondet(solution.houses)
    assume(fish_enthusiast.animal == "fish")
    assume(alice_parent.child == "Alice")
    assert fish_enthusiast.house_number > alice_parent.house_number

    # Clue 18: There is one house between the person's child is named Bella and the person who loves rock music.
    bella_parent = nondet(solution.houses)
    rock_lover = nondet(solution.houses)
    assume(bella_parent.child == "Bella")
    assume(rock_lover.music_genre == "rock")
    assert __CPROVER_abs(bella_parent.house_number - rock_lover.house_number) == 2

    # Clue 19: The person who is short is the cat lover.
    short_person = nondet(solution.houses)
    assume(short_person.height == "short")
    assert short_person.animal == "cat"

    # Clue 20: Alice is directly left of the person who loves classical music.
    alice = nondet(solution.houses)
    classical_lover = nondet(solution.houses)
    assume(alice.name == "Alice")
    assume(classical_lover.music_genre == "classical")
    assert alice.house_number == classical_lover.house_number - 1

    # Clue 21: The person's child is named Bella is the person whose mother's name is Aniya.
    bella_parent = nondet(solution.houses)
    assume(bella_parent.child == "Bella")
    assert bella_parent.mother == "Aniya"

    # Clue 22: There are two houses between the person whose mother's name is Penny and the person who is short.
    penny_child = nondet(solution.houses)
    short_person = nondet(solution.houses)
    assume(penny_child.mother == "Penny")
    assume(short_person.height == "short")
    assert __CPROVER_abs(penny_child.house_number - short_person.house_number) == 3

    # Clue 23: The person who loves hip-hop music is in the first house.
    hip_hop_lover = nondet(solution.houses)
    assume(hip_hop_lover.music_genre == "hip hop")
    assert hip_hop_lover.house_number == 1

    # Clue 24: Carol is the person who is tall.
    carol = nondet(solution.houses)
    assume(carol.name == "Carol")
    assert carol.height == "tall"
""",
            LogicAgent._LogicAgent__extract_code(  # pyright: ignore
                """```python
def validate(solution: PuzzleSolution) -> None:
    # Clue 1: The person who loves pop music is the cat lover.
    pop_lover = nondet(solution.houses)
    assume(pop_lover.music_genre == "pop")
    assert pop_lover.animal == "cat"

    # Clue 2: The rabbit owner is directly left of the person whose mother's name is Aniya.
    rabbit_owner = nondet(solution.houses)
    aniya_child = nondet(solution.houses)
    assume(rabbit_owner.animal == "rabbit")
    assume(aniya_child.mother == "Aniya")
    assert rabbit_owner.house_number == aniya_child.house_number - 1

    # Clue 3: The person whose mother's name is Holly is directly left of Carol.
    holly_child = nondet(solution.houses)
    carol = nondet(solution.houses)
    assume(holly_child.mother == "Holly")
    assume(carol.name == "Carol")
    assert holly_child.house_number == carol.house_number - 1

    # Clue 4: The person whose mother's name is Holly is the person's child is named Alice.
    holly_child = nondet(solution.houses)
    assume(holly_child.mother == "Holly")
    assert holly_child.child == "Alice"

    # Clue 5: The person whose mother's name is Holly is the person who loves classical music.
    holly_child = nondet(solution.houses)
    assume(holly_child.mother == "Holly")
    assert holly_child.music_genre == "classical"

    # Clue 6: The person who loves jazz music is the person whose mother's name is Sarah.
    jazz_lover = nondet(solution.houses)
    assume(jazz_lover.music_genre == "jazz")
    assert jazz_lover.mother == "Sarah"

    # Clue 7: The person's child is named Meredith is somewhere to the right of the person whose mother's name is Aniya.
    meredith_parent = nondet(solution.houses)
    aniya_child = nondet(solution.houses)
    assume(meredith_parent.child == "Meredith")
    assume(aniya_child.mother == "Aniya")
    assert meredith_parent.house_number > aniya_child.house_number

    # Clue 8: The person who is super tall is the person whose mother's name is Holly.
    super_tall_person = nondet(solution.houses)
    assume(super_tall_person.height == "super tall")
    assert super_tall_person.mother == "Holly"

    # Clue 9: The person who is the mother of Timothy is Bob.
    timothy_parent = nondet(solution.houses)
    assume(timothy_parent.child == "Timothy")
    assert timothy_parent.name == "Bob"

    # Clue 10: The person who is very short is somewhere to the left of the person whose mother's name is Aniya.
    very_short_person = nondet(solution.houses)
    aniya_child = nondet(solution.houses)
    assume(very_short_person.height == "very short")
    assume(aniya_child.mother == "Aniya")
    assert very_short_person.house_number < aniya_child.house_number

    # Clue 11: Eric is the fish enthusiast.
    eric = nondet(solution.houses)
    assume(eric.name == "Eric")
    assert eric.animal == "fish"

    # Clue 12: The person's child is named Samantha is somewhere to the right of the person who is very tall.
    samantha_parent = nondet(solution.houses)
    very_tall_person = nondet(solution.houses)
    assume(samantha_parent.child == "Samantha")
    assume(very_tall_person.height == "very tall")
    assert samantha_parent.house_number > very_tall_person.house_number

    # Clue 13: The person who loves rock music is the person whose mother's name is Janelle.
    rock_lover = nondet(solution.houses)
    assume(rock_lover.music_genre == "rock")
    assert rock_lover.mother == "Janelle"

    # Clue 14: There is one house between the person who keeps horses and the person's child is named Meredith.
    horse_owner = nondet(solution.houses)
    meredith_parent = nondet(solution.houses)
    assume(horse_owner.animal == "horse")
    assume(meredith_parent.child == "Meredith")
    assert __CPROVER_abs(horse_owner.house_number - meredith_parent.house_number) == 2

    # Clue 15: The person's child is named Bella is somewhere to the right of Peter.
    bella_parent = nondet(solution.houses)
    peter = nondet(solution.houses)```python
    assume(bella_parent.child == "Bella")
    assume(peter.name == "Peter")
    assert bella_parent.house_number > peter.house_number

    # Clue 16: The fish enthusiast is somewhere to the left of the bird keeper.
    fish_enthusiast = nondet(solution.houses)
    bird_keeper = nondet(solution.houses)
    assume(fish_enthusiast.animal == "fish")
    assume(bird_keeper.animal == "bird")
    assert fish_enthusiast.house_number < bird_keeper.house_number

    # Clue 17: The fish enthusiast is somewhere to the right of the person's child is named Alice.
    fish_enthusiast = nondet(solution.houses)
    alice_parent = nondet(solution.houses)
    assume(fish_enthusiast.animal == "fish")
    assume(alice_parent.child == "Alice")
    assert fish_enthusiast.house_number > alice_parent.house_number

    # Clue 18: There is one house between the person's child is named Bella and the person who loves rock music.
    bella_parent = nondet(solution.houses)
    rock_lover = nondet(solution.houses)
    assume(bella_parent.child == "Bella")
    assume(rock_lover.music_genre == "rock")
    assert __CPROVER_abs(bella_parent.house_number - rock_lover.house_number) == 2

    # Clue 19: The person who is short is the cat lover.
    short_person = nondet(solution.houses)
    assume(short_person.height == "short")
    assert short_person.animal == "cat"

    # Clue 20: Alice is directly left of the person who loves classical music.
    alice = nondet(solution.houses)
    classical_lover = nondet(solution.houses)
    assume(alice.name == "Alice")
    assume(classical_lover.music_genre == "classical")
    assert alice.house_number == classical_lover.house_number - 1

    # Clue 21: The person's child is named Bella is the person whose mother's name is Aniya.
    bella_parent = nondet(solution.houses)
    assume(bella_parent.child == "Bella")
    assert bella_parent.mother == "Aniya"

    # Clue 22: There are two houses between the person whose mother's name is Penny and the person who is short.
    penny_child = nondet(solution.houses)
    short_person = nondet(solution.houses)
    assume(penny_child.mother == "Penny")
    assume(short_person.height == "short")
    assert __CPROVER_abs(penny_child.house_number - short_person.house_number) == 3

    # Clue 23: The person who loves hip-hop music is in the first house.
    hip_hop_lover = nondet(solution.houses)
    assume(hip_hop_lover.music_genre == "hip hop")
    assert hip_hop_lover.house_number == 1

    # Clue 24: Carol is the person who is tall.
    carol = nondet(solution.houses)
    assume(carol.name == "Carol")
    assert carol.height == "tall"
```"""
            ),
        )
