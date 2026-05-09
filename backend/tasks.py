"""RoboMunch backend tasks.

OOP wrappers around HuggingFace pipelines / diffusers — adapted from the
EE471 Week 10 BaseTask pattern. Each task lazy-loads its model on the first
.run() call and caches it for subsequent requests.
"""
from abc import ABC, abstractmethod
from typing import List, Dict

import torch
from PIL import Image
from transformers import pipeline


class BaseTask(ABC):
    task_name: str = ""
    default_model: str | None = None

    def __init__(self, model: str | None = None):
        self.model = model or self.default_model
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            kwargs = {"model": self.model} if self.model else {}
            self._pipe = pipeline(self.task_name, **kwargs)
        return self._pipe

    @abstractmethod
    def run(self, *args, **kwargs):
        ...


class ImageGenerationTask(BaseTask):
    """Stable Diffusion text-to-image via diffusers."""

    default_model = "runwayml/stable-diffusion-v1-5"

    def __init__(self, model: str | None = None):
        super().__init__(model)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32

    def _load(self):
        if self._pipe is None:
            from diffusers import StableDiffusionPipeline

            pipe = StableDiffusionPipeline.from_pretrained(
                self.model,
                torch_dtype=self._dtype,
                safety_checker=None,
            )
            pipe = pipe.to(self._device)
            if self._device == "cuda":
                pipe.enable_attention_slicing()
            self._pipe = pipe
        return self._pipe

    def run(self, prompt: str, steps: int = 25, guidance: float = 7.5) -> Image.Image:
        pipe = self._load()
        result = pipe(
            prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
        )
        return result.images[0]


class ChatTask(BaseTask):
    """Conversational reply using SmolLM2-360M-Instruct."""

    task_name = "text-generation"
    default_model = "HuggingFaceTB/SmolLM2-360M-Instruct"

    SYSTEM_PROMPT = (
        "You are RoboMunch, a creative digital art prompt generator inspired "
        "by Edvard Munch. When the user asks for an art prompt, idea, or "
        "description, you MUST reply with ONE vivid imaginative scene "
        "description in under 20 words. Do not explain yourself. Do not ask "
        "questions back. Do not refuse. Just output the scene description as "
        "a single sentence.\n\n"
        "Examples:\n"
        "User: Generate me a prompt to draw a digital art image. Keep it within 20 words.\n"
        "RoboMunch: A surreal portrait of a neon-lit cyborg gazelle leaping through a swirling galaxy of digital watercolor wildflowers.\n\n"
        "User: Give me a cool digital paint idea.\n"
        "RoboMunch: An ancient dragon of stained glass perched on a clocktower beneath aurora-streaked midnight skies.\n\n"
        "User: Write a prompt for an ordinary digital image.\n"
        "RoboMunch: A lone red umbrella floating above a rainy Tokyo street drenched in cinematic neon reflections.\n\n"
        "For other questions, reply briefly and helpfully in one or two sentences."
    )

    def _load(self):
        if self._pipe is None:
            device = 0 if torch.cuda.is_available() else -1
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self._pipe = pipeline(
                self.task_name,
                model=self.model,
                torch_dtype=dtype,
                device=device,
            )
        return self._pipe

    def run(self, message: str, history: List[Dict[str, str]] | None = None) -> str:
        pipe = self._load()
        history = history or []
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        for h in history[-10:]:
            role = h.get("role")
            content = h.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})

        out = pipe(
            messages,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.3,
        )
        # transformers returns generated_text as a list of message dicts
        # when given chat-style input
        generated = out[0]["generated_text"]
        if isinstance(generated, list):
            return generated[-1]["content"].strip()
        return str(generated).strip()


class SpeechRecognitionTask(BaseTask):
    """Whisper ASR — adapted from EE471 Week 10.

    Uses PyAV to decode the input audio (webm/opus, ogg, mp3, wav, ...) to a
    16 kHz mono float32 numpy array, then feeds the raw samples to the
    HuggingFace pipeline. Avoids the need for a system-installed ffmpeg.
    """

    task_name = "automatic-speech-recognition"
    default_model = "openai/whisper-base"
    target_sr = 16000

    def _load(self):
        if self._pipe is None:
            device = 0 if torch.cuda.is_available() else -1
            self._pipe = pipeline(
                self.task_name,
                model=self.model,
                device=device,
            )
        return self._pipe

    @staticmethod
    def _decode(audio_path: str, target_sr: int) -> "np.ndarray":
        import av
        import numpy as np

        with av.open(audio_path) as container:
            stream = next(s for s in container.streams if s.type == "audio")
            resampler = av.audio.resampler.AudioResampler(
                format="s16", layout="mono", rate=target_sr
            )
            chunks = []
            for frame in container.decode(stream):
                for r in resampler.resample(frame):
                    chunks.append(r.to_ndarray().flatten())
            # flush
            for r in resampler.resample(None):
                chunks.append(r.to_ndarray().flatten())

        if not chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(chunks).astype(np.float32) / 32768.0
        return audio

    def run(self, audio_path: str) -> str:
        audio = self._decode(audio_path, self.target_sr)
        result = self._load()(
            {"raw": audio, "sampling_rate": self.target_sr}
        )
        return result["text"].strip()
