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
        waveforn_length = torch.tensor([waveform.shape[-1]]).int()

        tokens, token_lengths = self._tokenizer(transcript)

        return waveform, waveforn_length, transcript, tokens, token_lengths

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result


if __name__ == "__main__":
    # download check
    lj_speech = LJSpeechDataset()
    print(lj_speech[0])