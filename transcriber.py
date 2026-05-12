import logging
import os

# Bypass SSL verification for Hugging Face model downloads (corporate proxy)
import httpx
from huggingface_hub.utils import set_client_factory
set_client_factory(lambda: httpx.Client(verify=False))

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Load once at module level — expensive to reload per request
# "small" model: good balance of speed and accuracy for Indian English
# Change to "medium" for better accuracy at the cost of speed
_model = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        model_size = os.environ.get("WHISPER_MODEL", "small")
        logger.info(f"Loading Whisper model: {model_size}")
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")
        logger.info("Whisper model loaded")
    return _model


def transcribe(audio_path: str) -> str:
    """
    Transcribe an audio file to text.
    Returns empty string if transcription fails or produces no output.
    """
    model = get_model()
    segments, info = model.transcribe(
        audio_path,
        language="en",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()
    logger.info(f"Transcribed ({info.duration:.1f}s): {text[:100]}")
    return text
