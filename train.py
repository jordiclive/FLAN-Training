import argparse
import gc
import glob
import logging
import os
import time
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from lightning_base import BaseTransformer, generic_train
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right
from utils import (ROUGE_KEYS, _strtobool, calculate_rouge,
                   label_smoothed_nll_loss, read_yamls)

from data import BaseDataset, DataCollator

logger = logging.getLogger(__name__)


class ClassificationTransformer(BaseTransformer):
    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        super().__init__(hparams)
        self.dataset_size = len(
            pd.read_parquet(Path(self.hparams.data_path).joinpath("train.parquet"))
        )
        self.model_name_or_path = self.hparams.model_name_or_path
        self.metric_names = ROUGE_KEYS
        self.decoder_start_token_id = None
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eval_beams = (
            self.model.config.num_beams
            if self.hparams.eval_beams is None
            else self.hparams.eval_beams
        )
        assert (
            self.eval_beams >= 1
        ), f"got self.eval_beams={self.eval_beams}. Need an integer > 1"

        if self.hparams.eval_max_gen_length is not None:
            self.eval_max_length = self.hparams.eval_max_gen_length
        else:
            self.eval_max_length = self.model.config.max_length
        self.val_metric = (
            self.default_val_metric
            if self.hparams.val_metric is None
            else self.hparams.val_metric
        )

        self.eval_min_length = self.hparams.eval_min_length
        rank_zero_info(
            "for decoding, eval_max_length={}, "
            "eval_min_length={}, eval_beams={}".format(
                self.eval_max_length, self.eval_min_length, self.eval_beams
            )
        )
        if not self.hparams.grad_checkpoint and self.hparams.freeze_embeds:
            rank_zero_info("FREEZING embeddings")
            self.freeze_embeds()

    def freeze_params(self, model):
        """Set requires_grad=False for each of model.parameters()"""
        for par in model.parameters():
            par.requires_grad = False

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        self.freeze_params(self.model.shared)
        for d in [self.model.encoder, self.model.decoder]:
            self.freeze_params(d.embed_tokens)
            self.freeze_params(self.model.shared)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def _step(self, batch: dict) -> Tuple:
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]

        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(tgt_ids)
        else:
            decoder_input_ids = shift_tokens_right(tgt_ids, self.pad_token_id)

        outputs = self(
            src_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        )

        lm_logits = outputs[0]
        if self.hparams.label_smoothing == 0:
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

            loss = ce_loss_fct(
                lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1)
            )
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs,
                tgt_ids,
                self.hparams.label_smoothing,
                ignore_index=self.pad_token_id,
            )
        return (loss,)

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:

        loss = self._step(batch)[0]
        self.log(
            "train_loss",
            float(loss),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return {"loss": loss}

    def get_dataloader(
        self, mode: str, batch_size: int, shuffle: bool = False
    ) -> DataLoader:
        "Load datasets. Called after prepare data."
        rank_zero_info(f"batch_size: {batch_size}")

        if mode == "dev":
            data = pd.read_parquet(Path(self.hparams.data_path).joinpath("val.parquet"))

        if mode == "train":
            data = pd.read_parquet(
                Path(self.hparams.data_path).joinpath("train.parquet")
            )

        data = BaseDataset(data)
        collate_fn = DataCollator(
            self.tokenizer,
            padding=True,
            input_length=self.hparams.max_seq_length,
            output_length=self.hparams.max_target_length,
        )

        return DataLoader(
            data,
            collate_fn=collate_fn,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
        )

    def validation_step(self, batch, batch_idx):
        if self.hparams.skip_val:
            return {"loss": 1}
        if self.hparams.hf_checkpoint:
            save_path = Path(self.output_dir).joinpath("checkpoint-curr-best")
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            raise ValueError("just saving")
        return self._generative_step(batch, batch_idx)

    def _generative_step(
        self, batch: dict, batch_idx=None, dataloader_idx=None
    ) -> dict:

        loss = self._step(batch)[0]

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if self.hparams.generate_during_val:
            if batch_idx == 0:

                generated_ids = self.model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=True,
                    length_penalty=self.hparams.length_penalty,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=self.eval_beams,
                    min_length=self.eval_min_length,
                    max_length=self.eval_max_length,
                    no_repeat_ngram_size=self.hparams.no_repeat_ngram_size,
                    # repetition_penalty = self.hparams.repetition_penalty,
                )
                preds: List[str] = self.ids_to_clean_text(generated_ids)
                target: List[str] = self.ids_to_clean_text(batch["labels"])
                rouge: Dict = self.calc_generative_metrics(preds, target)
                self.log(
                    self.val_metric,
                    rouge[self.val_metric],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )
        return {"loss": loss}

    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_rouge(preds, target)

    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))

    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)

        effective_batch_size = (
            self.hparams.train_batch_size
            * self.hparams.gradient_accumulation_steps
            * num_devices
        )

        return (
            self.dataset_size / effective_batch_size
        ) * self.hparams.num_train_epochs

    @rank_zero_only
    def save_hf(self, path):

        rank_zero_info(path)
        save_path = Path(self.hparams.save_path).joinpath(path)
        self.save_path = save_path
        self.model.save_pretrained(save_path, torch_dtype=self.dtype)
        self.tokenizer.save_pretrained(save_path)

    @staticmethod
    def flatten(all_g, col):
        l = [x[col] for x in all_g]
        flat_list = [item for sublist in l for item in sublist]
        return flat_list

    def validation_epoch_end(self, outputs: list) -> dict:
        if self.hparams.skip_val:
            return 0
        gc.collect()
        torch.cuda.empty_cache()


def argument_parsing(notebook=False, notebook_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="""
        Multiple configs can be passed to set different options.
        For example, run as:

           ./train.py --configs galactica-125m webgpt_dataset_only per_digit_tokens
    """,
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--rng_seed", type=int, help="rng seed")

    if notebook:
        args, remaining = parser.parse_known_args(notebook_args)
    else:
        args, remaining = parser.parse_known_args()

    # Config from YAML
    conf = {}
    configs = read_yamls("configs/")
    conf.update(configs["defaults"])
    try:
        for name in args.configs:
            if "," in name:
                for n in name.split(","):
                    conf.update(configs[n])
            else:
                conf.update(configs[name])
    except KeyError as e:
        print(f'Error: Could not find the config "{e.args[0]}" in config.yaml')
        exit(1)

    conf["local_rank"] = args.local_rank
    conf["deepspeed"] = args.deepspeed
    if args.rng_seed is not None:
        conf["rng_seed"] = args.rng_seed

    # get the world size in deeepspeed
    if conf["deepspeed"]:
        conf["world_size"] = int(os.getenv("WORLD_SIZE", default="1"))
    else:
        conf["world_size"] = 1

    # Override config from command-line
    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = _strtobool
        parser.add_argument(f"--{key}", type=type_, default=value)
        # Allow --no-{key}  to remove it completely
        parser.add_argument(f"--no-{key}", dest=key, action="store_const", const=None)

    return parser.parse_args(remaining)


def main():
    args = argument_parsing()

    if args.local:
        args.gpus = 0
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices
    if args.offline or args.local:
        os.environ["WANDB_API_KEY"] = args.wandb_key
        os.environ["WANDB_MODE"] = "offline"
    else:
        os.environ["WANDB_API_KEY"] = args.wandb_key

    if args.output_dir is None:

        args.output_dir = os.path.join(
            args.output_dir,
            f"{time.strftime('%Y%m%d_%H%M')}",
        )
        try:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
        except:
            pass
    model = ClassificationTransformer(args)

    trainer = generic_train(model, args)

    if args.do_predict:
        checkpoints = list(
            sorted(
                glob.glob(
                    os.path.join(args.output_dir, "checkpoint-epoch=*.ckpt"),
                    recursive=True,
                )
            )
        )
        model = model.load_from_checkpoint(checkpoints[-1])
        return trainer.test(model)


if __name__ == "__main__":
    main()
