from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TokenizerSpec:
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    byte_offset: int = 3
    mask_id: int = 259

    @property
    def data_vocab_size(self) -> int:
        return self.mask_id

    @property
    def vocab_size(self) -> int:
        return self.mask_id + 1


class ByteTokenizer:
    """A deterministic byte-level tokenizer with an absorbing mask token.

    Token layout:
    - 0: PAD
    - 1: BOS
    - 2: EOS
    - 3..258: raw UTF-8 bytes
    - 259: MASK, the absorbing state used by the diffusion process
    """

    def __init__(self, spec: TokenizerSpec | None = None) -> None:
        self.spec = spec or TokenizerSpec()

    @property
    def pad_id(self) -> int:
        return self.spec.pad_id

    @property
    def bos_id(self) -> int:
        return self.spec.bos_id

    @property
    def eos_id(self) -> int:
        return self.spec.eos_id

    @property
    def mask_id(self) -> int:
        return self.spec.mask_id

    @property
    def vocab_size(self) -> int:
        return self.spec.vocab_size

    @property
    def data_vocab_size(self) -> int:
        return self.spec.data_vocab_size

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(byte + self.spec.byte_offset for byte in text.encode("utf-8"))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(
        self,
        ids: Iterable[int],
        *,
        skip_special: bool = True,
        stop_at_eos: bool = True,
    ) -> str:
        raw = bytearray()
        for token in ids:
            token = int(token)
            if token == self.eos_id and stop_at_eos:
                break
            if token in {self.pad_id, self.bos_id, self.eos_id, self.mask_id}:
                if skip_special:
                    continue
                if token == self.mask_id:
                    raw.extend(b"<mask>")
                continue
            byte = token - self.spec.byte_offset
            if 0 <= byte <= 255:
                raw.append(byte)
        return raw.decode("utf-8", errors="replace")

    def special_token_name(self, token_id: int) -> str | None:
        names = {
            self.pad_id: "<pad>",
            self.bos_id: "<bos>",
            self.eos_id: "<eos>",
            self.mask_id: "<mask>",
        }
        return names.get(int(token_id))
