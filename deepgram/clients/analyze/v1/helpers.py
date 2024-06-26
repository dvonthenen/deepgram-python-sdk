# Copyright 2023-2024 Deepgram SDK contributors. All Rights Reserved.
# Use of this source code is governed by a MIT license that can be found in the LICENSE file.
# SPDX-License-Identifier: MIT

from .options import AnalyzeSource


def is_buffer_source(provided_source: AnalyzeSource) -> bool:
    """
    Check if the provided source is a buffer source.
    """
    return "buffer" in provided_source


def is_readstream_source(provided_source: AnalyzeSource) -> bool:
    """
    Check if the provided source is a readstream source.
    """
    return "stream" in provided_source


def is_url_source(provided_source: AnalyzeSource) -> bool:
    """
    Check if the provided source is a url source.
    """
    return "url" in provided_source
