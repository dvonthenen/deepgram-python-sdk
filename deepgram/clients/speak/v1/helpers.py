# Copyright 2024 Deepgram SDK contributors. All Rights Reserved.
# Use of this source code is governed by a MIT license that can be found in the LICENSE file.
# SPDX-License-Identifier: MIT

from .options import SpeakSource


def is_text_source(provided_source: SpeakSource) -> bool:
    """
    Check if the provided source is a text source.
    """
    return "text" in provided_source


def is_readstream_source(provided_source: SpeakSource) -> bool:
    """
    Check if the provided source is a readstream source.
    """
    return "stream" in provided_source
