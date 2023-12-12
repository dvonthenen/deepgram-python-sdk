# Copyright 2023 Deepgram SDK contributors. All Rights Reserved.
# Use of this source code is governed by a MIT license that can be found in the LICENSE file.
# SPDX-License-Identifier: MIT

import logging, verboselogs
import inspect

from ...abstract_sync_client import AbstractSyncRestClient
from ..errors import DeepgramTypeError
from ..helpers import is_buffer_source, is_readstream_source, is_url_source
from ..source import UrlSource, FileSource

from .options import PrerecordedOptions
from .response import AsyncPrerecordedResponse, PrerecordedResponse


class PreRecordedClient(AbstractSyncRestClient):
    """
    A client class for handling pre-recorded audio data.
    Provides methods for transcribing audio from URLs and files.
    """

    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(config.verbose)
        self.config = config
        super().__init__(config)

    """
    Transcribes audio from a URL source.

    Args:
        source (UrlSource): The URL source of the audio to transcribe.
        options (PrerecordedOptions): Additional options for the transcription (default is None).
        endpoint (str): The API endpoint for the transcription (default is "v1/listen").

    Returns:
        SyncPrerecordedResponse: An object containing the transcription result.

    Raises:
        DeepgramApiError: Raised for known API errors.
        DeepgramUnknownApiError: Raised for unknown API errors.
        Exception: For any other unexpected exceptions.
    """

    def transcribe_url(
        self,
        source: UrlSource,
        options: PrerecordedOptions = None,
        endpoint: str = "v1/listen",
    ) -> PrerecordedResponse:
        self.logger.debug("PreRecordedClient.transcribe_url ENTER")
        url = f"{self.config.url}/{endpoint}"
        if options is not None and "callback" in options:
            self.logger.debug("PreRecordedClient.transcribe_url LEAVE")
            return self.transcribe_url_callback(
                source, options["callback"], options, endpoint
            )
        if is_url_source(source):
            body = source
        else:
            self.logger.error("Unknown transcription source type")
            self.logger.debug("PreRecordedClient.transcribe_url LEAVE")
            raise DeepgramTypeError("Unknown transcription source type")

        self.logger.info("url: %s", url)
        self.logger.info("source: %s", source)
        self.logger.info("options: %s", options)
        if isinstance(options, PrerecordedOptions):
            self.logger.info("PrerecordedOptions switching class -> json")
            options = options.to_json()
        res = PrerecordedResponse.from_json(self.post(url, options, json=body))
        self.logger.verbose("result: %s", res)
        self.logger.notice("transcribe_url succeeded")
        self.logger.debug("PreRecordedClient.transcribe_url LEAVE")
        return res

    """
    Transcribes audio from a URL source and sends the result to a callback URL.

    Args:
        source (UrlSource): The URL source of the audio to transcribe.
        callback (str): The callback URL where the transcription results will be sent.
        options (PrerecordedOptions): Additional options for the transcription (default is None).
        endpoint (str): The API endpoint for the transcription (default is "v1/listen").

    Returns:
        AsyncPrerecordedResponse: An object containing the request_id or an error message.

    Raises:
        DeepgramApiError: Raised for known API errors.
        DeepgramUnknownApiError: Raised for unknown API errors.
        Exception: For any other unexpected exceptions.
    """

    def transcribe_url_callback(
        self,
        source: UrlSource,
        callback: str,
        options: PrerecordedOptions = None,
        endpoint: str = "v1/listen",
    ) -> AsyncPrerecordedResponse:
        self.logger.debug("PreRecordedClient.transcribe_url_callback ENTER")

        url = f"{self.config.url}/{endpoint}"
        if options is None:
            options = {}
        options["callback"] = callback
        if is_url_source(source):
            body = source
        else:
            self.logger.error("Unknown transcription source type")
            self.logger.debug("PreRecordedClient.transcribe_url_callback LEAVE")
            raise DeepgramTypeError("Unknown transcription source type")

        self.logger.info("url: %s", url)
        self.logger.info("source: %s", source)
        self.logger.info("options: %s", options)
        if isinstance(options, PrerecordedOptions):
            self.logger.info("PrerecordedOptions switching class -> json")
            options = options.to_json()
        json = self.post(url, options, json=body)
        self.logger.info("json: %s", json)
        res = AsyncPrerecordedResponse.from_json(json)
        self.logger.verbose("result: %s", res)
        self.logger.notice("transcribe_url_callback succeeded")
        self.logger.debug("PreRecordedClient.transcribe_url_callback LEAVE")
        return res

    """
    Transcribes audio from a local file source.

    Args:
        source (FileSource): The local file source of the audio to transcribe.
        options (PrerecordedOptions): Additional options for the transcription (default is None).
        endpoint (str): The API endpoint for the transcription (default is "v1/listen").

    Returns:
        SyncPrerecordedResponse: An object containing the transcription result or an error message.

    Raises:

        DeepgramApiError: Raised for known API errors.
        DeepgramUnknownApiError: Raised for unknown API errors.
        Exception: For any other unexpected exceptions.
    """

    def transcribe_file(
        self,
        source: FileSource,
        options: PrerecordedOptions = None,
        endpoint: str = "v1/listen",
    ) -> PrerecordedResponse:
        self.logger.debug("PreRecordedClient.transcribe_file ENTER")

        url = f"{self.config.url}/{endpoint}"
        if is_buffer_source(source):
            body = source["buffer"]
        elif is_readstream_source(source):
            body = source["stream"]
        else:
            self.logger.error("Unknown transcription source type")
            self.logger.debug("PreRecordedClient.transcribe_file LEAVE")
            raise DeepgramTypeError("Unknown transcription source type")

        self.logger.info("url: %s", url)
        self.logger.info("options: %s", options)
        if isinstance(options, PrerecordedOptions):
            self.logger.info("PrerecordedOptions switching class -> json")
            options = options.to_json()
        json = self.post(url, options, content=body)
        self.logger.info("json: %s", json)
        res = PrerecordedResponse.from_json(json)
        self.logger.verbose("result: %s", res)
        self.logger.notice("transcribe_file succeeded")
        self.logger.debug("PreRecordedClient.transcribe_file LEAVE")
        return res

    """
    Transcribes audio from a local file source and sends the result to a callback URL.

    Args:
        source (FileSource): The local file source of the audio to transcribe.
        callback (str): The callback URL where the transcription results will be sent.
        options (PrerecordedOptions): Additional options for the transcription (default is None).
        endpoint (str): The API endpoint for the transcription (default is "v1/listen").

    Returns:
        AsyncPrerecordedResponse: An object containing the request_id or an error message.

    Raises:
        DeepgramApiError: Raised for known API errors.
        DeepgramUnknownApiError: Raised for unknown API errors.
        Exception: For any other unexpected exceptions.
    """

    def transcribe_file_callback(
        self,
        source: FileSource,
        callback: str,
        options: PrerecordedOptions = None,
        endpoint: str = "v1/listen",
    ) -> AsyncPrerecordedResponse:
        self.logger.debug("PreRecordedClient.transcribe_file_callback ENTER")

        url = f"{self.config.url}/{endpoint}"
        if options is None:
            options = {}
        options["callback"] = callback
        if is_buffer_source(source):
            body = source["buffer"]
        elif is_readstream_source(source):
            body = source["stream"]
        else:
            self.logger.error("Unknown transcription source type")
            self.logger.debug("PreRecordedClient.transcribe_file_callback LEAVE")
            raise DeepgramTypeError("Unknown transcription source type")

        self.logger.info("url: %s", url)
        self.logger.info("options: %s", options)
        if isinstance(options, PrerecordedOptions):
            self.logger.info("PrerecordedOptions switching class -> json")
            options = options.to_json()
        json = self.post(url, options, json=body)
        self.logger.info("json: %s", json)
        res = AsyncPrerecordedResponse.from_json(json)
        self.logger.verbose("result: %s", res)
        self.logger.notice("transcribe_file_callback succeeded")
        self.logger.debug("PreRecordedClient.transcribe_file_callback LEAVE")
        return res