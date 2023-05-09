from dataclasses import dataclass
from typing import Optional, Union

from torch.utils.data import Dataset
from transformers.tokenization_utils_base import (PaddingStrategy,
                                                  PreTrainedTokenizerBase)


class BaseDataset(Dataset):
    def __init__(self, df) -> None:
        super().__init__()
        self.pairs = list(zip(df["prompt"], df["text"], df["summary"]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        prompt, source, text = self.pairs[index]
        return prompt, source, text


@dataclass
class DataCollator:
    """
    Expects a list of texts corresponding to a sequence of [question, answer, question, answer, ...] pairs.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    input_length: Optional[int] = None
    output_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):

        batch = self.tokenizer.batch_encode_plus(
            [f"""{i[0].strip()} \n\n {i[1].strip()}""" for i in features],
            return_tensors="pt",
            max_length=self.input_length,
            padding=True,
            truncation=True,
            pad_to_multiple_of=8,
        )
        target = self.tokenizer.batch_encode_plus(
            [i[2].strip() for i in features],
            return_tensors="pt",
            max_length=self.output_length,
            padding=True,
            truncation=True,
            pad_to_multiple_of=8,
        )

        batch["labels"] = target["input_ids"]
        batch["target_mask"] = target["attention_mask"]

        return batch
