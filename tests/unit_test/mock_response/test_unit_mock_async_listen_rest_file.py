# Copyright 2024 Deepgram SDK contributors. All Rights Reserved.
# Use of this source code is governed by a MIT license that can be found in the LICENSE file.
# SPDX-License-Identifier: MIT

import json
import pytest
import hashlib

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    PrerecordedOptions,
    FileSource,
)

# response constants
FILE1 = "preamble-rest.wav"
FILE1_SMART_FORMAT = "*"
FILE1_SUMMARIZE1 = "*"

# Create a list of tuples to store the key-value pairs
input_output = [
    (
        FILE1,
        PrerecordedOptions(model="nova-2", smart_format=True),
        {"results.channels.0.alternatives.0.transcript": [FILE1_SMART_FORMAT]},
    ),
    (
        FILE1,
        PrerecordedOptions(model="nova-2", smart_format=True, summarize="v2"),
        {
            "results.channels.0.alternatives.0.transcript": [FILE1_SMART_FORMAT],
            "results.summary.short": [
                FILE1_SUMMARIZE1,
            ],
        },
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("filename, options, expected_output", input_output)
async def test_unit_mock_async_listen_rest_file(filename, options, expected_output):
    # options
    filenamestr = json.dumps(filename)
    input_sha256sum = hashlib.sha256(filenamestr.encode()).hexdigest()
    option_sha256sum = hashlib.sha256(options.to_json().encode()).hexdigest()

    unique = f"{option_sha256sum}-{input_sha256sum}"

    # Create a Deepgram client
    config: DeepgramClientOptions = DeepgramClientOptions(
        url="api.mock.deepgram.com",
    )
    deepgram: DeepgramClient = DeepgramClient("mock-api-key", config)

    # file buffer
    with open(f"tests/unit_test/{filename}", "rb") as file:
        buffer_data = file.read()

    payload: FileSource = {
        "buffer": buffer_data,
    }

    # make request
    response = await deepgram.listen.asyncrest.v("1").transcribe_file(payload, options)

    # Check the response
    for key, value in expected_output.items():
        actual = response.eval(key)

        assert actual != "", f"Test ID: {unique} - Key {key} found"
