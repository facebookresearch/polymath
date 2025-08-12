# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC

from agent.logic.logic_py_smt_constraint_generator import LogicPySMTConstraintGenerator
from agent.logic.logic_py_smt_data_structure_generator import (
    LogicPySMTDataStructureGenerator,
)

from libcst import MetadataWrapper


class LogicPySMTHarnessGenerator(ABC):
    """
    Converts LogicPy data structures and constraints to an equivalent C
    `struct`, suitable for synthesising solutions with CBMC.
    """

    @staticmethod
    def generate(metadata: MetadataWrapper) -> str:
        data_structures = LogicPySMTDataStructureGenerator()
        metadata.visit(data_structures)
        constraints = LogicPySMTConstraintGenerator(
            data_structures.ancestors_per_class, data_structures.smt_attributes
        )
        metadata.visit(constraints)
        return f"""{data_structures.smt_harness}
{constraints.smt_harness}"""
