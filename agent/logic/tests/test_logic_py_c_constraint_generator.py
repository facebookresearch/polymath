# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

from agent.logic.logic_py_c_constraint_generator import LogicPyCConstraintGenerator
from libcst import Module, parse_module


class TestLogicPyCConstraintGenerator(TestCase):

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.maxDiff = None

    def test_6x6(self) -> None:
        self.__test_harness(
            """
class House:
    house_number: Unique[Domain[int, range(1, 7)]]
    name: Unique[Domain[str, "Alice", "Eric", "Peter", "Carol", "Bob", "Arnold"]]
    smoothie: Unique[Domain[str, "watermelon", "blueberry", "desert", "cherry", "dragonfruit", "lime"]]
    lunch: Unique[Domain[str, "stew", "pizza", "grilled cheese", "stir fry", "soup", "spaghetti"]]
    phone: Unique[Domain[str, "google pixel 6", "iphone 13", "xiaomi mi 11", "huawei p50", "samsung galaxy s21", "oneplus 9"]]
    car: Unique[Domain[str, "tesla model 3", "honda civic", "toyota camry", "ford f150", "chevrolet silverado", "bmw 3 series"]]
    house_style: Unique[Domain[str, "craftsman", "ranch", "modern", "victorian", "mediterranean", "colonial"]]

class PuzzleSolution:
    houses: list[House, 6]

def validate(solution: PuzzleSolution) -> None:
    # Clue 1: Bob is the person who uses a Xiaomi Mi 11.
    bob = nondet(solution.houses)
    assume(bob.name == "Bob")
    assert bob.phone == "xiaomi mi 11"

    # Clue 2: The person who loves the soup is in the fourth house.
    soup_lover = nondet(solution.houses)
    assume(soup_lover.lunch == "soup")
    assert soup_lover.house_number == 4

    # Clue 3: The Dragonfruit smoothie lover is somewhere to the left of the person in a ranch-style home.
    dragonfruit_lover = nondet(solution.houses)
    assume(dragonfruit_lover.smoothie == "dragonfruit")
    ranch_owner = nondet(solution.houses)
    assume(ranch_owner.house_style == "ranch")
    assert dragonfruit_lover.house_number < ranch_owner.house_number

    # Clue 4: There is one house between the person who owns a Chevrolet Silverado and the person residing in a Victorian house.
    silverado_owner = nondet(solution.houses)
    assume(silverado_owner.car == "chevrolet silverado")
    victorian_owner = nondet(solution.houses)
    assume(victorian_owner.house_style == "victorian")
    assert __CPROVER_abs(silverado_owner.house_number - victorian_owner.house_number) == 2

    # Clue 5: The person in a Mediterranean-style villa is the person who drinks Lime smoothies.
    mediterranean_owner = nondet(solution.houses)
    assume(mediterranean_owner.house_style == "mediterranean")
    assert mediterranean_owner.smoothie == "lime"

    # Clue 6: Eric is in the sixth house.
    eric = nondet(solution.houses)
    assume(eric.name == "Eric")
    assert eric.house_number == 6

    # Clue 7: The Desert smoothie lover is the person who is a pizza lover.
    desert_lover = nondet(solution.houses)
    assume(desert_lover.smoothie == "desert")
    pizza_lover = nondet(solution.houses)
    assume(pizza_lover.lunch == "pizza")
    assert desert_lover == pizza_lover

    # Clue 8: The person living in a colonial-style house is the person who drinks Blueberry smoothies.
    colonial_owner = nondet(solution.houses)
    assume(colonial_owner.house_style == "colonial")
    assert colonial_owner.smoothie == "blueberry"

    # Clue 9: The Dragonfruit smoothie lover and the person who uses a Google Pixel 6 are next to each other.
    dragonfruit_lover = nondet(solution.houses)
    assume(dragonfruit_lover.smoothie == "dragonfruit")
    pixel_owner = nondet(solution.houses)
    assume(pixel_owner.phone == "google pixel 6")
    assert __CPROVER_abs(dragonfruit_lover.house_number - pixel_owner.house_number) == 1

    # Clue 10: The person who loves the soup is Peter.
    peter = nondet(solution.houses)
    assume(peter.name == "Peter")
    assert peter.lunch == "soup"

    # Clue 11: Alice is somewhere to the right of the person who owns a BMW 3 Series.
    alice = nondet(solution.houses)
    assume(alice.name == "Alice")
    bmw_owner = nondet(solution.houses)
    assume(bmw_owner.car == "bmw 3 series")
    assert alice.house_number > bmw_owner.house_number

    # Clue 12: The person who loves stir fry is the person in a ranch-style home.
    stir_fry_lover = nondet(solution.houses)
    assume(stir_fry_lover.lunch == "stir fry")
    ranch_owner = nondet(solution.houses)
    assume(ranch_owner.house_style == "ranch")
    assert stir_fry_lover == ranch_owner

    # Clue 13: The person who owns a Ford F-150 is the person living in a colonial-style house.
    ford_owner = nondet(solution.houses)
    assume(ford_owner.car == "ford f150")
    colonial_owner = nondet(solution.houses)
    assume(colonial_owner.house_style == "colonial")
    assert ford_owner == colonial_owner

    # Clue 14: The person in a Craftsman-style house is somewhere to the right of the person in a modern-style house.
    craftsman_owner = nondet(solution.houses)
    assume(craftsman_owner.house_style == "craftsman")
    modern_owner = nondet(solution.houses)
    assume(modern_owner.house_style == "modern")
    assert craftsman_owner.house_number > modern_owner.house_number

    # Clue 15: The person who loves the stew is directly left of the person in a ranch-style home.
    stew_lover = nondet(solution.houses)
    assume(stew_lover.lunch == "stew")
    ranch_owner = nondet(solution.houses)
    assume(ranch_owner.house_style == "ranch")
    assert stew_lover.house_number == ranch_owner.house_number - 1

    # Clue 16: The person who owns a Tesla Model 3 is directly left of the person who loves stir fry.
    tesla_owner = nondet(solution.houses)
    assume(tesla_owner.car == "tesla model 3")
    stir_fry_lover = nondet(solution.houses)
    assume(stir_fry_lover.lunch == "stir fry")
    assert tesla_owner.house_number == stir_fry_lover.house_number - 1

    # Clue 17: The person who loves eating grilled cheese is the person who owns a Honda Civic.
    grilled_cheese_lover = nondet(solution.houses)
    assume(grilled_cheese_lover.lunch == "grilled cheese")
    honda_owner = nondet(solution.houses)
    assume(honda_owner.car == "honda civic")
    assert grilled_cheese_lover == honda_owner

    # Clue 18: The person in a Mediterranean-style villa is the person who uses a Google Pixel 6.
    mediterranean_owner = nondet(solution.houses)
    assume(mediterranean_owner.house_style == "mediterranean")
    pixel_owner = nondet(solution.houses)
    assume(pixel_owner.phone == "google pixel 6")
    assert mediterranean_owner == pixel_owner

    # Clue 19: The person in a Craftsman-style house is the Watermelon smoothie lover.
    craftsman_owner = nondet(solution.houses)
    assume(craftsman_owner.house_style == "craftsman")
    assert craftsman_owner.smoothie == "watermelon"
""",
            """static void validate(struct PuzzleSolution solution) {
    typeof(__CPROVER_nondet_element(solution.houses)) bob = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(bob.name == "Bob");
    __CPROVER_assume(bob.phone == "xiaomi mi 11");
    typeof(__CPROVER_nondet_element(solution.houses)) soup_lover = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(soup_lover.lunch == "soup");
    __CPROVER_assume(soup_lover.house_number == 4);
    typeof(__CPROVER_nondet_element(solution.houses)) dragonfruit_lover = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(dragonfruit_lover.smoothie == "dragonfruit");
    typeof(__CPROVER_nondet_element(solution.houses)) ranch_owner = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(ranch_owner.house_style == "ranch");
    __CPROVER_assume(dragonfruit_lover.house_number < ranch_owner.house_number);
    typeof(__CPROVER_nondet_element(solution.houses)) silverado_owner = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(silverado_owner.car == "chevrolet silverado");
    typeof(__CPROVER_nondet_element(solution.houses)) victorian_owner = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(victorian_owner.house_style == "victorian");
    __CPROVER_assume(__CPROVER_abs(silverado_owner.house_number - victorian_owner.house_number) == 2);
    typeof(__CPROVER_nondet_element(solution.houses)) mediterranean_owner = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(mediterranean_owner.house_style == "mediterranean");
    __CPROVER_assume(mediterranean_owner.smoothie == "lime");
    typeof(__CPROVER_nondet_element(solution.houses)) eric = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(eric.name == "Eric");
    __CPROVER_assume(eric.house_number == 6);
    typeof(__CPROVER_nondet_element(solution.houses)) desert_lover = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(desert_lover.smoothie == "desert");
    typeof(__CPROVER_nondet_element(solution.houses)) pizza_lover = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(pizza_lover.lunch == "pizza");
    __CPROVER_assume(desert_lover == pizza_lover);
    typeof(__CPROVER_nondet_element(solution.houses)) colonial_owner = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(colonial_owner.house_style == "colonial");
    __CPROVER_assume(colonial_owner.smoothie == "blueberry");
    typeof(__CPROVER_nondet_element(solution.houses)) dragonfruit_lover_1 = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(dragonfruit_lover_1.smoothie == "dragonfruit");
    typeof(__CPROVER_nondet_element(solution.houses)) pixel_owner = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(pixel_owner.phone == "google pixel 6");
    __CPROVER_assume(__CPROVER_abs(dragonfruit_lover_1.house_number - pixel_owner.house_number) == 1);
    typeof(__CPROVER_nondet_element(solution.houses)) peter = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(peter.name == "Peter");
    __CPROVER_assume(peter.lunch == "soup");
    typeof(__CPROVER_nondet_element(solution.houses)) alice = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(alice.name == "Alice");
    typeof(__CPROVER_nondet_element(solution.houses)) bmw_owner = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(bmw_owner.car == "bmw 3 series");
    __CPROVER_assume(alice.house_number > bmw_owner.house_number);
    typeof(__CPROVER_nondet_element(solution.houses)) stir_fry_lover = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(stir_fry_lover.lunch == "stir fry");
    typeof(__CPROVER_nondet_element(solution.houses)) ranch_owner_1 = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(ranch_owner_1.house_style == "ranch");
    __CPROVER_assume(stir_fry_lover == ranch_owner_1);
    typeof(__CPROVER_nondet_element(solution.houses)) ford_owner = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(ford_owner.car == "ford f150");
    typeof(__CPROVER_nondet_element(solution.houses)) colonial_owner_1 = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(colonial_owner_1.house_style == "colonial");
    __CPROVER_assume(ford_owner == colonial_owner_1);
    typeof(__CPROVER_nondet_element(solution.houses)) craftsman_owner = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(craftsman_owner.house_style == "craftsman");
    typeof(__CPROVER_nondet_element(solution.houses)) modern_owner = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(modern_owner.house_style == "modern");
    __CPROVER_assume(craftsman_owner.house_number > modern_owner.house_number);
    typeof(__CPROVER_nondet_element(solution.houses)) stew_lover = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(stew_lover.lunch == "stew");
    typeof(__CPROVER_nondet_element(solution.houses)) ranch_owner_2 = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(ranch_owner_2.house_style == "ranch");
    __CPROVER_assume(stew_lover.house_number == ranch_owner_2.house_number - 1);
    typeof(__CPROVER_nondet_element(solution.houses)) tesla_owner = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(tesla_owner.car == "tesla model 3");
    typeof(__CPROVER_nondet_element(solution.houses)) stir_fry_lover_1 = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(stir_fry_lover_1.lunch == "stir fry");
    __CPROVER_assume(tesla_owner.house_number == stir_fry_lover_1.house_number - 1);
    typeof(__CPROVER_nondet_element(solution.houses)) grilled_cheese_lover = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(grilled_cheese_lover.lunch == "grilled cheese");
    typeof(__CPROVER_nondet_element(solution.houses)) honda_owner = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(honda_owner.car == "honda civic");
    __CPROVER_assume(grilled_cheese_lover == honda_owner);
    typeof(__CPROVER_nondet_element(solution.houses)) mediterranean_owner_1 = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(mediterranean_owner_1.house_style == "mediterranean");
    typeof(__CPROVER_nondet_element(solution.houses)) pixel_owner_1 = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(pixel_owner_1.phone == "google pixel 6");
    __CPROVER_assume(mediterranean_owner_1 == pixel_owner_1);
    typeof(__CPROVER_nondet_element(solution.houses)) craftsman_owner_1 = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(craftsman_owner_1.house_style == "craftsman");
    __CPROVER_assume(craftsman_owner_1.smoothie == "watermelon");
}

#ifndef __CPROVER
void __CPROVER_output(const char *name, struct PuzzleSolution solution);
#endif

int main(void) {
    struct PuzzleSolution solution;
    init_PuzzleSolution(&solution);
    validate(solution);

    __CPROVER_output("solution", solution);
    __CPROVER_assert(false, "");
}
""",
        )

    def test_index(self) -> None:
        self.__test_harness(
            """class House:
    id: Unique[Domain[int, range(1, 7)]]
    name: Unique[Domain[str, "Alice", "Carol", "Eric", "Peter", "Bob", "Arnold"]]
    music: Unique[Domain[str, "classical", "hip hop", "jazz", "pop", "rock", "country"]]
    mother: Unique[Domain[str, "Sarah", "Penny", "Aniya", "Janelle", "Kailyn", "Holly"]]
    child: Unique[Domain[str, "Alice", "Fred", "Timothy", "Bella", "Samantha", "Meredith"]]
    height: Unique[Domain[str, "very short", "tall", "short", "very tall", "super tall", "average"]]
    animal: Unique[Domain[str, "bird", "dog", "horse", "rabbit", "cat", "fish"]]

class Table:
    rows: list[House, 6]

def validate(solution: Table) -> None:
    # Clue 1: The person who loves pop music is the cat lover.
    pop_music_lover = nondet(solution.rows)
    assume(pop_music_lover.music == "pop")
    assert pop_music_lover.animal == "cat"

    # Clue 2: The rabbit owner is directly left of The person whose mother\'s name is Aniya.
    aniyas_mother = nondet(solution.rows)
    assume(aniyas_mother.mother == "Aniya")
    rabbit_owner = nondet(solution.rows)
    assume(rabbit_owner.animal == "rabbit")
    assert solution.rows.index(rabbit_owner) == solution.rows.index(aniyas_mother) - 1

    # Clue 3: The person whose mother\'s name is Holly is directly left of Carol.
    hollys_mother = nondet(solution.rows)
    assume(hollys_mother.mother == "Holly")
    carol = nondet(solution.rows)
    assume(carol.name == "Carol")
    assert solution.rows.index(hollys_mother) == solution.rows.index(carol) - 1

    # Clue 4: The person whose mother\'s name is Holly is the person\'s child is named Alice.
    hollys_mother = nondet(solution.rows)
    assume(hollys_mother.mother == "Holly")
    assert hollys_mother.child == "Alice"

    # Clue 5: The person whose mother\'s name is Holly is the person who loves classical music.
    hollys_mother = nondet(solution.rows)
    assume(hollys_mother.mother == "Holly")
    assert hollys_mother.music == "classical"

    # Clue 6: The person who loves jazz music is The person whose mother\'s name is Sarah.
    jazz_music_lover = nondet(solution.rows)
    assume(jazz_music_lover.music == "jazz")
    assert jazz_music_lover.mother == "Sarah"

    # Clue 7: The person\'s child is named Meredith is somewhere to the right of The person whose mother\'s name is Aniya.
    aniyas_mother = nondet(solution.rows)
    assume(aniyas_mother.mother == "Aniya")
    meredith_child = nondet(solution.rows)
    assume(meredith_child.child == "Meredith")
    assert solution.rows.index(aniyas_mother) < solution.rows.index(meredith_child)

    # Clue 8: The person who is super tall is The person whose mother\'s name is Holly.
    hollys_mother = nondet(solution.rows)
    assume(hollys_mother.mother == "Holly")
    assert hollys_mother.height == "super tall"

    # Clue 9: The person who is the mother of Timothy is Bob.
    bob = nondet(solution.rows)
    assume(bob.name == "Bob")
    assert bob.child == "Timothy"

    # Clue 10: The person who is very short is somewhere to the left of The person whose mother\'s name is Aniya.
    aniyas_mother = nondet(solution.rows)
    assume(aniyas_mother.mother == "Aniya")
    very_short = nondet(solution.rows)
    assume(very_short.height == "very short")
    assert solution.rows.index(very_short) < solution.rows.index(aniyas_mother)

    # Clue 11: Eric is the fish enthusiast.
    eric = nondet(solution.rows)
    assume(eric.name == "Eric")
    assert eric.animal == "fish"

    # Clue 12: The person\'s child is named Samantha is somewhere to the right of the person who is very tall.
    very_tall = nondet(solution.rows)
    assume(very_tall.height == "very tall")
    samantha_child = nondet(solution.rows)
    assume(samantha_child.child == "Samantha")
    assert solution.rows.index(very_tall) < solution.rows.index(samantha_child)

    # Clue 13: The person who loves rock music is The person whose mother\'s name is Janelle.
    rock_music_lover = nondet(solution.rows)
    assume(rock_music_lover.music == "rock")
    assert rock_music_lover.mother == "Janelle"

    # Clue 14: There is one house between the person who keeps horses and the person\'s child is named Meredith.
    horse_keeper = nondet(solution.rows)
    assume(horse_keeper.animal == "horse")
    meredith_child = nondet(solution.rows)
    assume(meredith_child.child == "Meredith")
    assert __CPROVER_abs(solution.rows.index(horse_keeper) - solution.rows.index(meredith_child)) == 2

    # Clue 15: The person\'s child is named Bella is somewhere to the right of Peter.
    peter = nondet(solution.rows)
    assume(peter.name == "Peter")
    bella_child = nondet(solution.rows)
    assume(bella_child.child == "Bella")
    assert solution.rows.index(peter) < solution.rows.index(bella_child)

    # Clue 16: The fish enthusiast is somewhere to the left of the bird keeper.
    fish_enthusiast = nondet(solution.rows)
    assume(fish_enthusiast.animal == "fish")
    bird_keeper = nondet(solution.rows)
    assume(bird_keeper.animal == "bird")
    assert solution.rows.index(fish_enthusiast) < solution.rows.index(bird_keeper)

    # Clue 17: The fish enthusiast is somewhere to the right of the person\'s child is named Alice.
    alice_child = nondet(solution.rows)
    assume(alice_child.child == "Alice")
    fish_enthusiast = nondet(solution.rows)
    assume(fish_enthusiast.animal == "fish")
    assert solution.rows.index(alice_child) < solution.rows.index(fish_enthusiast)

    # Clue 18: There is one house between the person\'s child is named Bella and the person who loves rock music.
    bella_child = nondet(solution.rows)
    assume(bella_child.child == "Bella")
    rock_music_lover = nondet(solution.rows)
    assume(rock_music_lover.music == "rock")
    assert __CPROVER_abs(solution.rows.index(bella_child) - solution.rows.index(rock_music_lover)) == 2

    # Clue 19: The person who is short is the cat lover.
    short_person = nondet(solution.rows)
    assume(short_person.height == "short")
    assert short_person.animal == "cat"

    # Clue 20: Alice is directly left of the person who loves classical music.
    alice = nondet(solution.rows)
    assume(alice.name == "Alice")
    classical_music_lover = nondet(solution.rows)
    assume(classical_music_lover.music == "classical")
    assert solution.rows.index(alice) == solution.rows.index(classical_music_lover) - 1

    # Clue 21: The person\'s child is named Bella is The person whose mother\'s name is Aniya.
    aniyas_mother = nondet(solution.rows)
    assume(aniyas_mother.mother == "Aniya")
    assert aniyas_mother.child == "Bella"

    # Clue 22: There are two houses between The person whose mother\'s name is Penny and the person who is short.
    penny_mother = nondet(solution.rows)
    assume(penny_mother.mother == "Penny")
    short_person = nondet(solution.rows)
    assume(short_person.height == "short")
    assert __CPROVER_abs(solution.rows.index(penny_mother) - solution.rows.index(short_person)) == 3

    # Clue 23: The person who loves hip-hop music is in the first house.
    hip_hop_music_lover = nondet(solution.rows)
    assume(hip_hop_music_lover.music == "hip hop")
    assert solution.rows.index(hip_hop_music_lover) == 0

    # Clue 24: Carol is the person who is tall.
    carol = nondet(solution.rows)
    assume(carol.name == "Carol")
    assert carol.height == "tall"
""",
            """static void validate(struct Table solution) {
    typeof(__CPROVER_nondet_element(solution.rows)) pop_music_lover = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(pop_music_lover.music == "pop");
    __CPROVER_assume(pop_music_lover.animal == "cat");
    typeof(__CPROVER_nondet_element(solution.rows)) aniyas_mother = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(aniyas_mother.mother == "Aniya");
    typeof(__CPROVER_nondet_element(solution.rows)) rabbit_owner = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(rabbit_owner.animal == "rabbit");
    __CPROVER_assume(__CPROVER_index(solution.rows, rabbit_owner) == __CPROVER_index(solution.rows, aniyas_mother) - 1);
    typeof(__CPROVER_nondet_element(solution.rows)) hollys_mother = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(hollys_mother.mother == "Holly");
    typeof(__CPROVER_nondet_element(solution.rows)) carol = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(carol.name == "Carol");
    __CPROVER_assume(__CPROVER_index(solution.rows, hollys_mother) == __CPROVER_index(solution.rows, carol) - 1);
    typeof(__CPROVER_nondet_element(solution.rows)) hollys_mother_1 = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(hollys_mother_1.mother == "Holly");
    __CPROVER_assume(hollys_mother_1.child == "Alice");
    typeof(__CPROVER_nondet_element(solution.rows)) hollys_mother_2 = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(hollys_mother_2.mother == "Holly");
    __CPROVER_assume(hollys_mother_2.music == "classical");
    typeof(__CPROVER_nondet_element(solution.rows)) jazz_music_lover = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(jazz_music_lover.music == "jazz");
    __CPROVER_assume(jazz_music_lover.mother == "Sarah");
    typeof(__CPROVER_nondet_element(solution.rows)) aniyas_mother_1 = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(aniyas_mother_1.mother == "Aniya");
    typeof(__CPROVER_nondet_element(solution.rows)) meredith_child = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(meredith_child.child == "Meredith");
    __CPROVER_assume(__CPROVER_index(solution.rows, aniyas_mother_1) < __CPROVER_index(solution.rows, meredith_child));
    typeof(__CPROVER_nondet_element(solution.rows)) hollys_mother_3 = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(hollys_mother_3.mother == "Holly");
    __CPROVER_assume(hollys_mother_3.height == "super tall");
    typeof(__CPROVER_nondet_element(solution.rows)) bob = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(bob.name == "Bob");
    __CPROVER_assume(bob.child == "Timothy");
    typeof(__CPROVER_nondet_element(solution.rows)) aniyas_mother_2 = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(aniyas_mother_2.mother == "Aniya");
    typeof(__CPROVER_nondet_element(solution.rows)) very_short = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(very_short.height == "very short");
    __CPROVER_assume(__CPROVER_index(solution.rows, very_short) < __CPROVER_index(solution.rows, aniyas_mother_2));
    typeof(__CPROVER_nondet_element(solution.rows)) eric = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(eric.name == "Eric");
    __CPROVER_assume(eric.animal == "fish");
    typeof(__CPROVER_nondet_element(solution.rows)) very_tall = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(very_tall.height == "very tall");
    typeof(__CPROVER_nondet_element(solution.rows)) samantha_child = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(samantha_child.child == "Samantha");
    __CPROVER_assume(__CPROVER_index(solution.rows, very_tall) < __CPROVER_index(solution.rows, samantha_child));
    typeof(__CPROVER_nondet_element(solution.rows)) rock_music_lover = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(rock_music_lover.music == "rock");
    __CPROVER_assume(rock_music_lover.mother == "Janelle");
    typeof(__CPROVER_nondet_element(solution.rows)) horse_keeper = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(horse_keeper.animal == "horse");
    typeof(__CPROVER_nondet_element(solution.rows)) meredith_child_1 = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(meredith_child_1.child == "Meredith");
    __CPROVER_assume(__CPROVER_abs(__CPROVER_index(solution.rows, horse_keeper) - __CPROVER_index(solution.rows, meredith_child_1)) == 2);
    typeof(__CPROVER_nondet_element(solution.rows)) peter = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(peter.name == "Peter");
    typeof(__CPROVER_nondet_element(solution.rows)) bella_child = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(bella_child.child == "Bella");
    __CPROVER_assume(__CPROVER_index(solution.rows, peter) < __CPROVER_index(solution.rows, bella_child));
    typeof(__CPROVER_nondet_element(solution.rows)) fish_enthusiast = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(fish_enthusiast.animal == "fish");
    typeof(__CPROVER_nondet_element(solution.rows)) bird_keeper = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(bird_keeper.animal == "bird");
    __CPROVER_assume(__CPROVER_index(solution.rows, fish_enthusiast) < __CPROVER_index(solution.rows, bird_keeper));
    typeof(__CPROVER_nondet_element(solution.rows)) alice_child = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(alice_child.child == "Alice");
    typeof(__CPROVER_nondet_element(solution.rows)) fish_enthusiast_1 = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(fish_enthusiast_1.animal == "fish");
    __CPROVER_assume(__CPROVER_index(solution.rows, alice_child) < __CPROVER_index(solution.rows, fish_enthusiast_1));
    typeof(__CPROVER_nondet_element(solution.rows)) bella_child_1 = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(bella_child_1.child == "Bella");
    typeof(__CPROVER_nondet_element(solution.rows)) rock_music_lover_1 = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(rock_music_lover_1.music == "rock");
    __CPROVER_assume(__CPROVER_abs(__CPROVER_index(solution.rows, bella_child_1) - __CPROVER_index(solution.rows, rock_music_lover_1)) == 2);
    typeof(__CPROVER_nondet_element(solution.rows)) short_person = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(short_person.height == "short");
    __CPROVER_assume(short_person.animal == "cat");
    typeof(__CPROVER_nondet_element(solution.rows)) alice = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(alice.name == "Alice");
    typeof(__CPROVER_nondet_element(solution.rows)) classical_music_lover = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(classical_music_lover.music == "classical");
    __CPROVER_assume(__CPROVER_index(solution.rows, alice) == __CPROVER_index(solution.rows, classical_music_lover) - 1);
    typeof(__CPROVER_nondet_element(solution.rows)) aniyas_mother_3 = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(aniyas_mother_3.mother == "Aniya");
    __CPROVER_assume(aniyas_mother_3.child == "Bella");
    typeof(__CPROVER_nondet_element(solution.rows)) penny_mother = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(penny_mother.mother == "Penny");
    typeof(__CPROVER_nondet_element(solution.rows)) short_person_1 = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(short_person_1.height == "short");
    __CPROVER_assume(__CPROVER_abs(__CPROVER_index(solution.rows, penny_mother) - __CPROVER_index(solution.rows, short_person_1)) == 3);
    typeof(__CPROVER_nondet_element(solution.rows)) hip_hop_music_lover = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(hip_hop_music_lover.music == "hip hop");
    __CPROVER_assume(__CPROVER_index(solution.rows, hip_hop_music_lover) == 0);
    typeof(__CPROVER_nondet_element(solution.rows)) carol_1 = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(carol_1.name == "Carol");
    __CPROVER_assume(carol_1.height == "tall");
}

#ifndef __CPROVER
void __CPROVER_output(const char *name, struct Table solution);
#endif

int main(void) {
    struct Table solution;
    init_Table(&solution);
    validate(solution);

    __CPROVER_output("solution", solution);
    __CPROVER_assert(false, "");
}
""",
        )

    def test_float(self) -> None:
        self.__test_harness(
            """class House:
    id: Unique[Domain[int, range(1, 7)]]
    name: Unique[Domain[str, "Alice", "Carol", "Eric", "Peter", "Bob", "Arnold"]]
    music: Unique[Domain[str, "classical", "hip hop", "jazz", "pop", "rock", "country"]]
    mother: Unique[Domain[str, "Sarah", "Penny", "Aniya", "Janelle", "Kailyn", "Holly"]]
    child: Unique[Domain[str, "Alice", "Fred", "Timothy", "Bella", "Samantha", "Meredith"]]
    height: Unique[Domain[str, "very short", "tall", "short", "very tall", "super tall", "average"]]
    animal: Unique[Domain[str, "bird", "dog", "horse", "rabbit", "cat", "fish"]]

class Table:
    rows: list[House, 6]

def validate(solution: Table) -> None:
    # Clue 14: There is one house between the person who keeps horses and the person\'s child is named Meredith.
    horse_keeper = nondet(solution.rows)
    assume(horse_keeper.animal == "horse")
    meredith_child = nondet(solution.rows)
    assume(meredith_child.child == "Meredith")
    assert __CPROVER_abs(solution.rows.index(horse_keeper) - solution.rows.index(meredith_child)) == 2.
""",
            """static void validate(struct Table solution) {
    typeof(__CPROVER_nondet_element(solution.rows)) horse_keeper = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(horse_keeper.animal == "horse");
    typeof(__CPROVER_nondet_element(solution.rows)) meredith_child = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(meredith_child.child == "Meredith");
    __CPROVER_assume(__CPROVER_abs(__CPROVER_index(solution.rows, horse_keeper) - __CPROVER_index(solution.rows, meredith_child)) == 2.);
}

#ifndef __CPROVER
void __CPROVER_output(const char *name, struct Table solution);
#endif

int main(void) {
    struct Table solution;
    init_Table(&solution);
    validate(solution);

    __CPROVER_output("solution", solution);
    __CPROVER_assert(false, "");
}
""",
        )

    def test_logical_operator(self) -> None:
        self.__test_harness(
            """class House:
    id: Unique[Domain[int, range(1, 7)]]
    name: Unique[Domain[str, "Alice", "Carol", "Eric", "Peter", "Bob", "Arnold"]]
    music: Unique[Domain[str, "classical", "hip hop", "jazz", "pop", "rock", "country"]]
    mother: Unique[Domain[str, "Sarah", "Penny", "Aniya", "Janelle", "Kailyn", "Holly"]]
    child: Unique[Domain[str, "Alice", "Fred", "Timothy", "Bella", "Samantha", "Meredith"]]
    height: Unique[Domain[str, "very short", "tall", "short", "very tall", "super tall", "average"]]
    animal: Unique[Domain[str, "bird", "dog", "horse", "rabbit", "cat", "fish"]]

class Table:
    rows: list[House, 6]

def validate(solution: Table) -> None:
    # Clue 14: There is one house between the person who keeps horses and the person\'s child is named Meredith.
    horse_keeper = nondet(solution.rows)
    assume(horse_keeper.animal == "horse")
    meredith_child = nondet(solution.rows)
    assume(meredith_child.child == "Meredith")
    assert horse_keeper.music == "classical" and __CPROVER_abs(solution.rows.index(horse_keeper) - solution.rows.index(meredith_child)) == 2.
    assert horse_keeper.music == "classical" or __CPROVER_abs(solution.rows.index(horse_keeper) - solution.rows.index(meredith_child)) == 2.
""",
            """static void validate(struct Table solution) {
    typeof(__CPROVER_nondet_element(solution.rows)) horse_keeper = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(horse_keeper.animal == "horse");
    typeof(__CPROVER_nondet_element(solution.rows)) meredith_child = __CPROVER_nondet_element(solution.rows);
    __CPROVER_assume(meredith_child.child == "Meredith");
    __CPROVER_assume(horse_keeper.music == "classical" && __CPROVER_abs(__CPROVER_index(solution.rows, horse_keeper) - __CPROVER_index(solution.rows, meredith_child)) == 2.);
    __CPROVER_assume(horse_keeper.music == "classical" || __CPROVER_abs(__CPROVER_index(solution.rows, horse_keeper) - __CPROVER_index(solution.rows, meredith_child)) == 2.);
}

#ifndef __CPROVER
void __CPROVER_output(const char *name, struct Table solution);
#endif

int main(void) {
    struct Table solution;
    init_Table(&solution);
    validate(solution);

    __CPROVER_output("solution", solution);
    __CPROVER_assert(false, "");
}
""",
        )

    def test_subscript(self) -> None:
        self.__test_harness(
            """class House:
    id: Unique[Domain[int, range(1, 7)]]
    name: Unique[Domain[str, "Alice", "Carol", "Eric", "Peter", "Bob", "Arnold"]]
    music: Unique[Domain[str, "classical", "hip hop", "jazz", "pop", "rock", "country"]]
    mother: Unique[Domain[str, "Sarah", "Penny", "Aniya", "Janelle", "Kailyn", "Holly"]]
    child: Unique[Domain[str, "Alice", "Fred", "Timothy", "Bella", "Samantha", "Meredith"]]
    height: Unique[Domain[str, "very short", "tall", "short", "very tall", "super tall", "average"]]
    animal: Unique[Domain[str, "bird", "dog", "horse", "rabbit", "cat", "fish"]]

class Table:
    rows: list[House, 6]

def validate(solution: Table) -> None:
    horse_keeper = solution.rows[1]
    assume(horse_keeper.animal == "horse")
""",
            """static void validate(struct Table solution) {
    typeof(solution.rows[1]) horse_keeper = solution.rows[1];
    __CPROVER_assume(horse_keeper.animal == "horse");
}

#ifndef __CPROVER
void __CPROVER_output(const char *name, struct Table solution);
#endif

int main(void) {
    struct Table solution;
    init_Table(&solution);
    validate(solution);

    __CPROVER_output("solution", solution);
    __CPROVER_assert(false, "");
}
""",
        )

    def test_bool_literal(self) -> None:
        self.__test_harness(
            """def validate(solution: Solution) -> None:
    eric = nondet(solution.houses)
    assume(eric.name == "Eric")
    arnold = nondet(solution.houses)
    assume(arnold.name == "Arnold")
    assert eric != arnold  # implicit in the problem statement

    dog_owner = nondet(solution.houses)
    assume(dog_owner.pet == "dog")
    assert dog_owner != solution.houses[0]  # The person who owns a dog is not in the first house.

    # Eric is somewhere to the left of Arnold
    # We can\\'t directly compare the indices of eric and arnold because the solver
    # might not assign them in the order we expect. Instead, we can use the fact
    # that the houses are numbered from 1 to 2, and Eric must be in house 1 if
    # Arnold is in house 2, or vice versa.
    if eric == solution.houses[0]:
        assert arnold == solution.houses[1]
    elif eric == solution.houses[1]:
        assert arnold == solution.houses[0]
    else:
        assert False  # This should never happen
""",
            """static void validate(struct Solution solution) {
    typeof(__CPROVER_nondet_element(solution.houses)) eric = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(eric.name == "Eric");
    typeof(__CPROVER_nondet_element(solution.houses)) arnold = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(arnold.name == "Arnold");
    __CPROVER_assume(eric != arnold);
    typeof(__CPROVER_nondet_element(solution.houses)) dog_owner = __CPROVER_nondet_element(solution.houses);
    __CPROVER_assume(dog_owner.pet == "dog");
    __CPROVER_assume(dog_owner != solution.houses[0]);
    if (eric == solution.houses[0]) {
    __CPROVER_assume(arnold == solution.houses[1]);
    }
    else     if (eric == solution.houses[1]) {
    __CPROVER_assume(arnold == solution.houses[0]);
    }
    else {
    __CPROVER_assume(false);
    }
}

#ifndef __CPROVER
void __CPROVER_output(const char *name, struct Solution solution);
#endif

int main(void) {
    struct Solution solution;
    init_Solution(&solution);
    validate(solution);

    __CPROVER_output("solution", solution);
    __CPROVER_assert(false, "");
}
""",
        )

    def __test_harness(self, code: str, expected_harness: str) -> None:
        source_tree: Module = parse_module(code)
        visitor = LogicPyCConstraintGenerator(source_tree)
        source_tree.visit(visitor)
        self.assertEqual(expected_harness, visitor.c_constraints)
