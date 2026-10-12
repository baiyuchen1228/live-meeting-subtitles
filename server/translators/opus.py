"""English -> Chinese translation with Helsinki-NLP Opus-MT (Marian, ~300MB).

The model writes mostly Simplified Chinese, so ZH-TW output is converted
with OpenCC (s2twp: Taiwan standard characters and phrasing).
"""

import threading

try:
    from transformers import MarianMTModel, MarianTokenizer
    import torch
except ImportError:
    MarianMTModel = None
    torch = None

try:
    import opencc
except ImportError:
    opencc = None

from .base import Translator, TranslationError


class OpusTranslator(Translator):
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-zh", device: str | None = None):
        if MarianMTModel is None:
            raise ImportError("transformers required: pip install transformers sentencepiece torch")
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None
        self._to_traditional = opencc.OpenCC("s2twp") if opencc else None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"Opus-MT ({self.model_name.split('/')[-1]})"

    def _load(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            print(f"Loading translation model: {self.model_name}...")
            self._tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._model = MarianMTModel.from_pretrained(self.model_name, dtype=dtype).to(self.device).eval()
            # We pass max_new_tokens; drop the config's max_length so transformers
            # does not log a warning on every call
            self._model.generation_config.max_length = None

    def translate(self, text: str, target_lang: str, source_lang: str | None = None, fast: bool = False) -> str:
        if not text.strip():
            return ""
        if not target_lang.upper().startswith("ZH"):
            raise TranslationError(f"Opus-MT en-zh only translates to Chinese, not {target_lang}")
        self._load()
        try:
            inputs = self._tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
            # Marian loops on very short inputs ("Oh." -> "呵呵。 呵呵。 呵呵。"):
            # bound the output by the input length and forbid repeated 3-grams
            max_new = min(256, 3 * inputs["input_ids"].shape[1] + 8)
            with torch.no_grad():
                generated = self._model.generate(
                    **inputs, max_new_tokens=max_new, num_beams=1 if fast else 4, no_repeat_ngram_size=3
                )
            out = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        except Exception as e:
            raise TranslationError(f"Opus-MT translation error: {e}") from e
        if target_lang.upper() == "ZH-TW":
            if self._to_traditional is None:
                raise TranslationError("opencc required for ZH-TW: pip install opencc-python-reimplemented")
            out = self._to_traditional.convert(out)
        return out
