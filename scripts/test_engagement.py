"""Live test: opens the lamp in a GUI window, your webcam, and your
microphone. Sit in frame -> music stops, a chime plays, it turns toward
you, nods, flashes then brightens, and starts listening. Say something ->
local silence detection commits your utterance to a persistent Gemini
Live session; a short "thinking" cue (blue light + hum + pondering
gesture) plays until the reply's audio starts arriving, then it speaks
while reacting with whatever gesture it calls (nod/shake_head/excited/
curious/think) -- measured live, first audio lands ~1.7-2.2s after your
utterance is committed, not the 30-90+ seconds the old per-utterance
request/response pipeline took. Look away at any point -> it stops
listening, dims, returns home, and the idle music resumes; a reply still
in flight keeps playing into the room rather than being dropped. Right
after it engages, it also takes one look at whatever's in front of the
camera and remembers it (demo moment 4) -- ask it later what it's seen.

Requires GEMINI_API_KEY in a local .env (copy .env.example, add your key
from https://aistudio.google.com/apikey -- no credit card needed) for
conversation and scene memory. Engagement, motion, light, and music all
work without a key.

Usage: .venv/bin/python scripts/test_engagement.py
Ctrl+C to quit. macOS will prompt for camera and microphone permission the
first time.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

from src.body.executor import ActionExecutor, ExecutorHooks
from src.body.sim import LampSimulator
from src.character.fsm import CharacterOrchestrator
from src.character.memory import SceneMemory
from src.perception.engagement import EngagementWatcher
from src.speech.audio_mixer import AudioMixer
from src.speech.error_speech import ErrorSpeech
from src.speech.live_capture import LiveMicStreamer
from src.speech.live_client import LIVE_MODEL, OUTPUT_SAMPLE_RATE, GeminiLiveClient
from src.speech.playback import make_audio_hooks


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("No GEMINI_API_KEY found -- conversation and scene memory will be")
        print("disabled (add .env to enable). Engagement, motion, light, and")
        print("music all work without a key.")

    sim = LampSimulator(gui=True, clean_gui=True)
    mixer = AudioMixer(sample_rate=OUTPUT_SAMPLE_RATE)
    mixer.start()
    # on_speak (the standalone "speak" Action, unrelated to live conversation)
    # still uses the older tts.py request/response call; on_play_sound/
    # on_music_on/on_music_off are overridden with the mixer's versions so
    # SFX/music/(live speech, fed directly -- see fsm.py) all share one
    # output stream instead of competing sd.play() calls.
    hooks_dict = make_audio_hooks(speak=bool(api_key))
    hooks_dict.update(mixer.make_hooks())
    executor = ActionExecutor(sim, hooks=ExecutorHooks(**hooks_dict))
    watcher = EngagementWatcher()

    live_client = None
    mic_streamer = None
    error_speech = None
    if api_key:
        live_client = GeminiLiveClient(
            LIVE_MODEL, api_key, on_audio_chunk=mixer.enqueue_speech_pcm16, on_debug=print
        )
        mic_streamer = LiveMicStreamer(live_client, on_debug=print)
        # Pre-synthesized now, not on demand -- see error_speech.py's
        # docstring for why (the moment this is needed may be the moment a
        # fresh network call is least likely to succeed quickly).
        error_speech = ErrorSpeech(mixer, on_debug=print)
        error_speech.preload()

    print("Starting camera and microphone...")
    watcher.start()
    if mic_streamer is not None:
        mic_streamer.start()
        print("Connecting to Gemini Live...")
        if live_client.wait_until_connected(timeout_s=15.0):
            print("Connected.")
        else:
            print("Still connecting in the background (will keep retrying)...")
    print("Watching. Look at your webcam to engage the lamp; look away to disengage.")
    print("While engaged, speak -- it only listens while it's paying attention to you.")

    memory = SceneMemory()
    orchestrator = CharacterOrchestrator(
        executor,
        watcher,
        live_client=live_client,
        audio_mixer=mixer,
        memory=memory,
        error_speech=error_speech,
        on_debug=print,
    )
    try:
        orchestrator.run_forever(poll_hz=10.0)
    finally:
        watcher.stop()
        if mic_streamer is not None:
            mic_streamer.stop()
        if live_client is not None:
            live_client.close()
        mixer.close()
        sim.close()


if __name__ == "__main__":
    main()
