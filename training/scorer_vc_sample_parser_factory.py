# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from plugin.plugin_loader import PluginLoader
from training.scorer_vc_sample_parser import ScorerVCSampleParser


def create_scorer_vc_sample_parser() -> ScorerVCSampleParser:
    """ """
    plugin: PluginLoader[ScorerVCSampleParser] = PluginLoader("scorer_vc_sample_parser")
    if not plugin.is_configured:
        raise ValueError("scorer_vc_sample_parser plug-in is required.")

    return plugin.factory()
