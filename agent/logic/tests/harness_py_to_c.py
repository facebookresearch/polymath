# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

from agent.logic.logic_py_c_harness_generator import LogicPyCHarnessGenerator
from libcst import Module, parse_module
from agent.logic.cbmc_search_engine_strategy import CBMCSearchEngineStrategy


class TestLogicPyCHarnessGenerator(TestCase):

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.maxDiff = None

    def test_6x6(self) -> None:
        self.__test_harness(
            """
class House:
    number: Unique[Domain[int, range(1, 3)]]          # 1 = left, 2 = right
    name:   Unique[Domain[str, "Eric", "Arnold"]]     # one name per house
    pet:    Unique[Domain[str, "dog", "cat"]]         # one pet type per house

class Solution:
    houses: list[House, 2]   
    
def validate(solution: Solution) -> None:
    # --- Clue 1: "Eric is somewhere to the left of Arnold." ------------------
    eric_house = nondet(solution.houses)
    assume(eric_house.name == "Eric")

    arnold_house = nondet(solution.houses)
    assume(arnold_house.name == "Arnold")

    # Left-to-right corresponds to lower house numbers.
    assert eric_house.number < arnold_house.number

    
# --- Clue 2: "The person who owns a dog is not in the first house." ------
    dog_house = nondet(solution.houses)
    assume(dog_house.pet == "dog")

    # House 1 is the leftmost house.
    assert dog_house.number != 1
""",
            """#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

#ifndef __CPROVER
void __CPROVER_assume(bool condition);
void __CPROVER_assert(bool condition, const char *message);
#endif

#define __CPROVER_unique_domain(field, field_domain_array)                                  {                                                                                               size_t index;                                                                               __CPROVER_assume(index < (sizeof(field_domain_array) / sizeof(field_domain_array[0])));     __CPROVER_assume(!field_domain_array##_used[index]);                                        field_domain_array##_used[index] = true;                                                    field = field_domain_array[index];                                                      }

size_t __CPROVER_nondet_array_index(size_t length) {
    size_t index;
    __CPROVER_assume(index < length);
    return index;
}

#define __CPROVER_nondet_index(array)                                  __CPROVER_nondet_array_index(sizeof(array) / sizeof(array[0]))

#define __CPROVER_nondet_element(array)        (array)[__CPROVER_nondet_index(array)]

static size_t __CPROVER_index_tmp;

static size_t __CPROVER_index_dispatch(bool comparison) {
    __CPROVER_assume(comparison);
    return __CPROVER_index_tmp;
}

#define __CPROVER_index(array, value)     __CPROVER_index_dispatch(array[__CPROVER_index_tmp = __CPROVER_nondet_index(array)] == value)

struct House {
    int house_number;
    const char * name;
    const char * smoothie;
    const char * lunch;
    const char * phone;
    const char * car;
    const char * house_style;
};

static int House_house_number[] = {1, 2, 3, 4, 5, 6};
static bool House_house_number_used[6];
static const char * House_name[] = {"Alice", "Eric", "Peter", "Carol", "Bob", "Arnold"};
static bool House_name_used[6];
static const char * House_smoothie[] = {"watermelon", "blueberry", "desert", "cherry", "dragonfruit", "lime"};
static bool House_smoothie_used[6];
static const char * House_lunch[] = {"stew", "pizza", "grilled cheese", "stir fry", "soup", "spaghetti"};
static bool House_lunch_used[6];
static const char * House_phone[] = {"google pixel 6", "iphone 13", "xiaomi mi 11", "huawei p50", "samsung galaxy s21", "oneplus 9"};
static bool House_phone_used[6];
static const char * House_car[] = {"tesla model 3", "honda civic", "toyota camry", "ford f150", "chevrolet silverado", "bmw 3 series"};
static bool House_car_used[6];
static const char * House_house_style[] = {"craftsman", "ranch", "modern", "victorian", "mediterranean", "colonial"};
static bool House_house_style_used[6];

static void init_House(struct House * instance) {
    __CPROVER_unique_domain(instance->house_number, House_house_number);
    __CPROVER_unique_domain(instance->name, House_name);
    __CPROVER_unique_domain(instance->smoothie, House_smoothie);
    __CPROVER_unique_domain(instance->lunch, House_lunch);
    __CPROVER_unique_domain(instance->phone, House_phone);
    __CPROVER_unique_domain(instance->car, House_car);
    __CPROVER_unique_domain(instance->house_style, House_house_style);
}

struct PuzzleSolution {
    struct House houses[6];
};

static void init_PuzzleSolution(struct PuzzleSolution * instance) {
    for (size_t i = 0; i < sizeof(instance->houses) / sizeof(instance->houses[0]); ++i) {
        init_House(&instance->houses[i]);
    }
}

static void validate(struct PuzzleSolution solution) {
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

    def test_training_sequence_issue(self) -> None:
        self.__test_harness(
            """class Entity:
    house_number: Unique[Domain[int, range(1, 4)]]
    favorite_color: Unique[Domain[str, "blue", "teal", "pink"]]
    favorite_vegetable: Unique[Domain[str, "tomato", "carrot", "bell pepper"]]
    name: Unique[Domain[str, "john", "benjamin", "lisa"]]
    occupation: Unique[Domain[str, "teacher", "programmer", "musician"]]

class PuzzleSolution:
    entities: list[Entity, 3]

def validate(solution: PuzzleSolution) -> None:
    entity_0 = nondet(solution.entities)
    assume(entity_0.favorite_vegetable == "tomato")
    entity_1 = nondet(solution.entities)
    assume(entity_1.occupation == "programmer")
    assert abs(entity_0.house_number - entity_1.house_number) == 1
    __polymath_constraint_separator()
    entity_2 = nondet(solution.entities)
    assume(entity_2.favorite_vegetable == "carrot")
    entity_3 = nondet(solution.entities)
    assume(entity_3.favorite_vegetable == "bell pepper")
    assert entity_2.house_number == entity_3.house_number + 1
    __polymath_constraint_separator()
    entity_4 = nondet(solution.entities)
    assume(entity_4.favorite_vegetable == "carrot")
    entity_5 = nondet(solution.entities)
    assume(entity_5.name == "lisa")
    assert entity_4.house_number == entity_5.house_number
    __polymath_constraint_separator()
    entity_6 = nondet(solution.entities)
    assume(entity_6.favorite_color == "pink")
    entity_7 = nondet(solution.entities)
    assume(entity_7.name == "john")
    assert abs(entity_6.house_number - entity_7.house_number) == 1
    __polymath_constraint_separator()
    entity_8 = nondet(solution.entities)
    assume(entity_8.favorite_vegetable == "bell pepper")
    entity_9 = nondet(solution.entities)
    assume(entity_9.favorite_color == "teal")
    assert entity_8.house_number == entity_9.house_number - 1
    __polymath_constraint_separator()
    entity_10 = nondet(solution.entities)
    assume(entity_10.occupation == "musician")
    entity_11 = nondet(solution.entities)
    assume(entity_11.occupation == "programmer")
    assert entity_10.house_number == entity_11.house_number - 1
    __polymath_constraint_separator()
    entity_12 = nondet(solution.entities)
    assume(entity_12.favorite_color == "pink")
    entity_13 = nondet(solution.entities)
    assume(entity_13.occupation == "musician")
    assert abs(entity_12.house_number - entity_13.house_number) == 1
    __polymath_constraint_separator()
""",
            """#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

#ifndef __CPROVER
void __CPROVER_assume(bool condition);
void __CPROVER_assert(bool condition, const char *message);
#endif

#define __CPROVER_unique_domain(field, field_domain_array)                                  {                                                                                               size_t index;                                                                               __CPROVER_assume(index < (sizeof(field_domain_array) / sizeof(field_domain_array[0])));     __CPROVER_assume(!field_domain_array##_used[index]);                                        field_domain_array##_used[index] = true;                                                    field = field_domain_array[index];                                                      }

size_t __CPROVER_nondet_array_index(size_t length) {
    size_t index;
    __CPROVER_assume(index < length);
    return index;
}

#define __CPROVER_nondet_index(array)                                  __CPROVER_nondet_array_index(sizeof(array) / sizeof(array[0]))

#define __CPROVER_nondet_element(array)        (array)[__CPROVER_nondet_index(array)]

static size_t __CPROVER_index_tmp;

static size_t __CPROVER_index_dispatch(bool comparison) {
    __CPROVER_assume(comparison);
    return __CPROVER_index_tmp;
}

#define __CPROVER_index(array, value)     __CPROVER_index_dispatch(array[__CPROVER_index_tmp = __CPROVER_nondet_index(array)] == value)

struct Entity {
    int house_number;
    const char * favorite_color;
    const char * favorite_vegetable;
    const char * name;
    const char * occupation;
};

static int Entity_house_number[] = {1, 2, 3};
static bool Entity_house_number_used[3];
static const char * Entity_favorite_color[] = {"blue", "teal", "pink"};
static bool Entity_favorite_color_used[3];
static const char * Entity_favorite_vegetable[] = {"tomato", "carrot", "bell pepper"};
static bool Entity_favorite_vegetable_used[3];
static const char * Entity_name[] = {"john", "benjamin", "lisa"};
static bool Entity_name_used[3];
static const char * Entity_occupation[] = {"teacher", "programmer", "musician"};
static bool Entity_occupation_used[3];

static void init_Entity(struct Entity * instance) {
    __CPROVER_unique_domain(instance->house_number, Entity_house_number);
    __CPROVER_unique_domain(instance->favorite_color, Entity_favorite_color);
    __CPROVER_unique_domain(instance->favorite_vegetable, Entity_favorite_vegetable);
    __CPROVER_unique_domain(instance->name, Entity_name);
    __CPROVER_unique_domain(instance->occupation, Entity_occupation);
}

struct PuzzleSolution {
    struct Entity entities[3];
};

static void init_PuzzleSolution(struct PuzzleSolution * instance) {
    for (size_t i = 0; i < sizeof(instance->entities) / sizeof(instance->entities[0]); ++i) {
        init_Entity(&instance->entities[i]);
    }
}

static void validate(struct PuzzleSolution solution) {
    typeof(__CPROVER_nondet_element(solution.entities)) entity_0 = __CPROVER_nondet_element(solution.entities);
    __CPROVER_assume(entity_0.favorite_vegetable == "tomato");
    typeof(__CPROVER_nondet_element(solution.entities)) entity_1 = __CPROVER_nondet_element(solution.entities);
    __CPROVER_assume(entity_1.occupation == "programmer");
    __CPROVER_assume(__CPROVER_abs(entity_0.house_number - entity_1.house_number) == 1);
    ;
    typeof(__CPROVER_nondet_element(solution.entities)) entity_2 = __CPROVER_nondet_element(solution.entities);
    __CPROVER_assume(entity_2.favorite_vegetable == "carrot");
    typeof(__CPROVER_nondet_element(solution.entities)) entity_3 = __CPROVER_nondet_element(solution.entities);
    __CPROVER_assume(entity_3.favorite_vegetable == "bell pepper");
    __CPROVER_assume(entity_2.house_number == entity_3.house_number + 1);
    ;
    typeof(__CPROVER_nondet_element(solution.entities)) entity_4 = __CPROVER_nondet_element(solution.entities);
    __CPROVER_assume(entity_4.favorite_vegetable == "carrot");
    typeof(__CPROVER_nondet_element(solution.entities)) entity_5 = __CPROVER_nondet_element(solution.entities);
    __CPROVER_assume(entity_5.name == "lisa");
    __CPROVER_assume(entity_4.house_number == entity_5.house_number);
    ;
    typeof(__CPROVER_nondet_element(solution.entities)) entity_6 = __CPROVER_nondet_element(solution.entities);
    __CPROVER_assume(entity_6.favorite_color == "pink");
    typeof(__CPROVER_nondet_element(solution.entities)) entity_7 = __CPROVER_nondet_element(solution.entities);
    __CPROVER_assume(entity_7.name == "john");
    __CPROVER_assume(__CPROVER_abs(entity_6.house_number - entity_7.house_number) == 1);
    ;
    typeof(__CPROVER_nondet_element(solution.entities)) entity_8 = __CPROVER_nondet_element(solution.entities);
    __CPROVER_assume(entity_8.favorite_vegetable == "bell pepper");
    typeof(__CPROVER_nondet_element(solution.entities)) entity_9 = __CPROVER_nondet_element(solution.entities);
    __CPROVER_assume(entity_9.favorite_color == "teal");
    __CPROVER_assume(entity_8.house_number == entity_9.house_number - 1);
    ;
    typeof(__CPROVER_nondet_element(solution.entities)) entity_10 = __CPROVER_nondet_element(solution.entities);
    __CPROVER_assume(entity_10.occupation == "musician");
    typeof(__CPROVER_nondet_element(solution.entities)) entity_11 = __CPROVER_nondet_element(solution.entities);
    __CPROVER_assume(entity_11.occupation == "programmer");
    __CPROVER_assume(entity_10.house_number == entity_11.house_number - 1);
    ;
    typeof(__CPROVER_nondet_element(solution.entities)) entity_12 = __CPROVER_nondet_element(solution.entities);
    __CPROVER_assume(entity_12.favorite_color == "pink");
    typeof(__CPROVER_nondet_element(solution.entities)) entity_13 = __CPROVER_nondet_element(solution.entities);
    __CPROVER_assume(entity_13.occupation == "musician");
    __CPROVER_assume(__CPROVER_abs(entity_12.house_number - entity_13.house_number) == 1);
    ;
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

    def __test_harness(self, code: str, expected_harness: str) -> None:
        source_tree: Module = parse_module(code)
        harness: str = LogicPyCHarnessGenerator.generate(source_tree)
        print(harness)
        #self.assertEqual(expected_harness, harness)
        
    def test_parseur() -> None:
        e = CBMCSearchEngineStrategy()
        
