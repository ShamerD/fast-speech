import unicodedata

import torch
import torchaudio

from src.utils import DATA_DIR


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        super().__init__(root=DATA_DIR, download=True)
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        waveform_length = torch.tensor([waveform.shape[-1]]).int()

        transcript = self._remove_accents(transcript)
        tokens, token_lengths = self._tokenizer(transcript)

        return waveform, waveform_length, transcript, tokens, token_lengths

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result

    @staticmethod
    def _remove_accents(text: str):
        """
        :param text: single text possibly containing non-ascii symbols (eg. Müller)
        :return: normalized text with accents removed

        LJSpeech dataset contains 19 entries with accented text.
        They can cause trouble, because 2 tokenizers (dataset's and aligner's) handle accented texts differently:
        dataset's tokenizer simply ignores them, aligner's treat them as <unk>.
        This leads to wrong alignment if not runtime error (if entry is batch's longest text)
        """
        return "".join([c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)])


if __name__ == "__main__":
    # download check
    lj_speech = LJSpeechDataset()
    print(lj_speech[0])
