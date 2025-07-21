# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

from agent.symex.scope import ScopeManager


class TestScopeManager(TestCase):

    def __init__(self, methodName="runTest") -> None:
        super().__init__(methodName)
        self.__manager = ScopeManager()

    def test_empty(self) -> None:
        self.assertEqual("foo", self.__manager.get_qualified_name("foo"))

    def test_missing(self) -> None:
        self.__manager.declare_variable("foo")
        self.__manager.begin_scope()
        self.__manager.declare_variable("bar")
        self.assertEqual("baz", self.__manager.get_qualified_name("baz"))

    def test_scope(self) -> None:
        self.__manager.declare_variable("foo")
        self.__manager.begin_scope()
        self.__manager.declare_variable("bar")
        self.assertEqual("foo", self.__manager.get_qualified_name("foo"))
        self.assertEqual("0::bar", self.__manager.get_qualified_name("bar"))

    def test_end_scope(self) -> None:
        self.__manager.declare_variable("foo")
        self.__manager.begin_scope()
        self.__manager.declare_variable("bar")
        self.__manager.begin_scope()
        self.__manager.declare_variable("bar")
        self.__manager.end_scope()
        self.assertEqual("0::bar", self.__manager.get_qualified_name("bar"))

    def test_nested_end(self) -> None:
        self.__manager.declare_variable("bar")
        self.assertEqual("bar", self.__manager.get_qualified_name("bar"))
        self.__manager.begin_scope()
        self.__manager.declare_variable("bar")
        self.assertEqual("0::bar", self.__manager.get_qualified_name("bar"))
        self.__manager.begin_scope()
        self.__manager.declare_variable("bar")
        self.assertEqual("0::0::bar", self.__manager.get_qualified_name("bar"))
        self.__manager.end_scope()
        self.assertEqual("0::bar", self.__manager.get_qualified_name("bar"))
        self.__manager.begin_scope()
        self.__manager.declare_variable("bar")
        self.assertEqual("0::1::bar", self.__manager.get_qualified_name("bar"))
        self.__manager.end_scope()
        self.assertEqual("0::bar", self.__manager.get_qualified_name("bar"))
        self.__manager.end_scope()
        self.assertEqual("bar", self.__manager.get_qualified_name("bar"))
        self.__manager.begin_scope()
        self.__manager.declare_variable("bar")
        self.assertEqual("1::bar", self.__manager.get_qualified_name("bar"))
