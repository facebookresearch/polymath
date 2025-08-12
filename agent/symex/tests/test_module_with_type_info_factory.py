# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Mapping
from unittest import IsolatedAsyncioTestCase

from agent.logic.z3_conclusion_check_engine_strategy import _PYTHON_CODE_PREFIX

from agent.symex.module_with_type_info_factory import ModuleWithTypeInfoFactory

from concurrency.async_pool import AsyncPool

from libcst import CSTNode, MetadataWrapper

from libcst.metadata.type_inference_provider import TypeInferenceProvider


_CODE: str = f"""
{_PYTHON_CODE_PREFIX}

class Person:
    name: str
    age: int

class Universe:
    people: list[Person]

def premise(universe: Universe) -> None:
    peter = some(universe.people)
    assert peter.age > 10 or peter.age < 70
"""


class TestModuleWithTypeInfoFactory(IsolatedAsyncioTestCase):

    async def test_single(self) -> None:
        wrapper: MetadataWrapper = await ModuleWithTypeInfoFactory.create_module(_CODE)
        types: Mapping[CSTNode, str] = wrapper.resolve(TypeInferenceProvider)
        assert "typing.List[module.Person]" in str(types)

    async def test_multiple(self) -> None:
        pool = AsyncPool(10)
        for _ in range(10):
            await pool.submit(lambda: self.test_single())
        await pool.gather()
