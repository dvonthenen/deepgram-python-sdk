# Copyright 2024 Deepgram SDK contributors. All Rights Reserved.
# Use of this source code is governed by a MIT license that can be found in the LICENSE file.
# SPDX-License-Identifier: MIT

import pytest
import hashlib

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    SpeakOptions,
)

MODEL = "aura-asteria-en"

# response constants
TEXT1 = "Hello, world."

# Create a list of tuples to store the key-value pairs
input_output = [
    (
        TEXT1,
        SpeakOptions(model=MODEL, encoding="linear16", sample_rate=24000),
        {
            "content_type": ["audio/wav"],
            "model_name": ["aura-asteria-en"],
            "characters": ["13"],
        },
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("text, options, expected_output", input_output)
async def test_unit_mock_async_speak_rest(text, options, expected_output):
    # Save the options
    input_sha256sum = hashlib.sha256(text.encode()).hexdigest()
    option_sha256sum = hashlib.sha256(options.to_json().encode()).hexdigest()

    unique = f"{option_sha256sum}-{input_sha256sum}"

    # Create a Deepgram client
    config: DeepgramClientOptions = DeepgramClientOptions(
        url="api.mock.deepgram.com",
    )
    deepgram: DeepgramClient = DeepgramClient("mock-api-key", config)

    # input text
    input_text = {"text": text}

    # Send the URL to Deepgram
    response = await deepgram.speak.asyncrest.v("1").stream_memory(input_text, options)
    # TODO: how do I test: response.stream_memory.getbuffer()

    # Check the response
    for key, value in expected_output.items():
        actual = response.eval(key)

        assert actual != "", f"Test ID: {unique} - Key {key} found"
