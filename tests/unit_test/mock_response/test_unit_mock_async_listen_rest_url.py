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
)

# response constants
URL1 = {
    "url": "https://static.deepgram.com/examples/Bueller-Life-moves-pretty-fast.wav"
}
URL1_SMART_FORMAT1 = "*"
URL1_SUMMARIZE1 = "*"

# Create a list of tuples to store the key-value pairs
input_output = [
    (
        URL1,
        PrerecordedOptions(model="nova-2", smart_format=True),
        {"results.channels.0.alternatives.0.transcript": [URL1_SMART_FORMAT1]},
    ),
    (
        URL1,
        PrerecordedOptions(model="nova-2", smart_format=True, summarize="v2"),
        {
            "results.channels.0.alternatives.0.transcript": [URL1_SMART_FORMAT1],
            "results.summary.short": [URL1_SUMMARIZE1],
        },
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("url, options, expected_output", input_output)
async def test_unit_mock_async_listen_rest_url(url, options, expected_output):
    # options
    urlstr = json.dumps(url)
    input_sha256sum = hashlib.sha256(urlstr.encode()).hexdigest()
    option_sha256sum = hashlib.sha256(options.to_json().encode()).hexdigest()

    unique = f"{option_sha256sum}-{input_sha256sum}"

    # Create a Deepgram client
    config: DeepgramClientOptions = DeepgramClientOptions(
        url="api.mock.deepgram.com",
    )
    deepgram: DeepgramClient = DeepgramClient("mock-api-key", config)

    # make request
    response = await deepgram.listen.asyncrest.v("1").transcribe_url(url, options)

    # Check the response
    for key, value in expected_output.items():
        actual = response.eval(key)

        assert actual != "", f"Test ID: {unique} - Key {key} found"
