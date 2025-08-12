# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class FinishReason(Enum):
    """
    Chat completions completion reason, indication successful or failed
    completions.
    """

    STOPPED = 1  # Successful completion
    MAX_OUTPUT_TOKENS = 2  # Token limit reached, will try to continue where we left off
    RETRYABLE_ERROR = 3  # Error in completion API, should retry.
