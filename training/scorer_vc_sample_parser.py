# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Tuple


class ScorerVCSampleParser(ABC):
    """ """

    @abstractmethod
    def parse(self, line: str) -> Tuple[str, str, str]:
        """ """
        ...
