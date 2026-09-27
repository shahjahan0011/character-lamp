"""Shared test fixtures. Automated tests must never make a real Gemini
API call (or need a camera/microphone/speaker/display/API key) -- this
autouse fixture is a global safety net for that, on top of individual
tests patching more specifically where the call site matters. Any test
whose SceneObserver background thread actually tries to reach
vision.describe_scene without an explicit, more specific patch gets a
fast, local, no-network failure instead of a real (or even attempted)
outbound call.
"""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _block_real_gemini_calls():
    def _raise(*args, **kwargs):
        raise RuntimeError("blocked: automated tests must not call the real Gemini API")

    # vision.py/tts.py each did `from .gemini_client import get_client`, a
    # separate name binding in their own module namespace -- patching only
    # gemini_client.get_client would silently miss both (a classic
    # patching gotcha), so each call site needs its own patch target.
    with patch("src.speech.gemini_client.get_client", side_effect=_raise), \
         patch("src.speech.vision.get_client", side_effect=_raise), \
         patch("src.speech.tts.get_client", side_effect=_raise):
        yield
