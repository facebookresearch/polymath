# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC

from agent.logic.logic_py_c_constraint_generator import LogicPyCConstraintGenerator
from agent.logic.logic_py_c_data_structure_generator import (
    LogicPyCDataStructureGenerator,
)

from libcst._nodes.module import Module


class LogicPyCHarnessGenerator(ABC):
    """
    Converts LogicPy data structures and constraints to an equivalent C
    `struct`, suitable for synthesising solutions with CBMC.
    """

    @staticmethod
    def generate(module: Module) -> str:
        data_structures = LogicPyCDataStructureGenerator(module)
        module.visit(data_structures)
        constraints = LogicPyCConstraintGenerator(module)
        module.visit(constraints)
        return f"""#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

#ifndef __CPROVER
void __CPROVER_assume(bool condition);
void __CPROVER_assert(bool condition, const char *message);
#endif

#define __CPROVER_unique_domain(field, field_domain_array)                                  \
{{                                                                                           \
    size_t index;                                                                           \
    __CPROVER_assume(index < (sizeof(field_domain_array) / sizeof(field_domain_array[0]))); \
    __CPROVER_assume(!field_domain_array##_used[index]);                                    \
    field_domain_array##_used[index] = true;                                                \
    field = field_domain_array[index];                                                      \
}}

size_t __CPROVER_nondet_array_index(size_t length) {{
    size_t index;
    __CPROVER_assume(index < length);
    return index;
}}

#define __CPROVER_nondet_index(array)                              \
    __CPROVER_nondet_array_index(sizeof(array) / sizeof(array[0]))

#define __CPROVER_nondet_element(array)    \
    (array)[__CPROVER_nondet_index(array)]

static size_t __CPROVER_index_tmp;

static size_t __CPROVER_index_dispatch(bool comparison) {{
    __CPROVER_assume(comparison);
    return __CPROVER_index_tmp;
}}

#define __CPROVER_index(array, value) \
    __CPROVER_index_dispatch(array[__CPROVER_index_tmp = __CPROVER_nondet_index(array)] == value)

{data_structures.c_harness}{constraints.c_constraints}"""
