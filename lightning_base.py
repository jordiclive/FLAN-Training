import argparse
import json
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from transformers import ( AutoConfig, AutoModel,
                          AutoModelForPreTraining,
                          AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoModelWithLMHead,
                          AutoTokenizer, PretrainedConfig, PreTrainedTokenizer)
from torch.optim import AdamW
from transformers.optimization import (
    Adafactor, get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup)
from transformers.utils.versions import require_version

logger = logging.getLogger(__name__)

require_version("pytorch_lightning>=1.0.4")

MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelWithLMHead,
    "summarization": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
}

# update this and the import above to support new schedulers from transformers.optimization
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"


class BaseTransformer(pl.LightningModule):
    def __init__(
        self,
        hparams: argparse.Namespace,
        config=None,
        tokenizer=None,
        num_labels=None,
        model=None,
        **config_kwargs,
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()

        self.save_hyperparameters(hparams)
        self.step_count = 0
        self.output_dir = Path(self.hparams.output_dir)
        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None
        self.model_name_or_path = self.hparams.model_name_or_path

        if config is None:
            self.config = AutoConfig.from_pretrained(
                self.hparams.config_name
                if self.hparams.config_name
                else self.hparams.model_name_or_path,
                **({"num_labels": num_labels} if num_labels is not None else {}),
                cache_dir=cache_dir,
                **config_kwargs,
            )
        else:
            self.config: PretrainedConfig = config

        extra_model_params = (
            "encoder_layerdrop",
            "decoder_layerdrop",
            "dropout",
            "attention_dropout",
        )
        for p in extra_model_params:
            if getattr(self.hparams, p, None):
                assert hasattr(
                    self.config, p
                ), f"model config doesn't have a `{p}` attribute"
                setattr(self.config, p, getattr(self.hparams, p))
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.model_name_or_path,
                cache_dir=cache_dir,
            )
        else:
            self.tokenizer: PreTrainedTokenizer = tokenizer
        if self.hparams.local:
            self.torchtype = torch.float32
        elif self.hparams.precision == "bf16":
            self.torchtype = torch.bfloat16
        else:
            self.torchtype = torch.float16

        if model is None:
            if self.hparams.grad_checkpoint:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.hparams.model_name_or_path,
                    from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
                    config=self.config,
                    cache_dir=cache_dir,
                    torch_dtype=self.torchtype,
                )
                self.model.gradient_checkpointing_enable()
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.hparams.model_name_or_path,
                    from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
                    config=self.config,
                    cache_dir=cache_dir,
                    torch_dtype=self.torchtype,
                )

        else:
            self.model = model
        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.max_target_length,
            "test": self.hparams.max_target_length,
        }

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        scheduler = get_schedule_func(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps(),
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.opt = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
        )

        scheduler = self.get_lr_scheduler()

        return [self.opt], [scheduler]

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        return self.validation_end(outputs)

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False):
        raise NotImplementedError("You must implement this for your task")

    def train_dataloader(self):
        return self.get_dataloader("train", self.hparams.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.hparams.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.hparams.eval_batch_size, shuffle=False)


def generic_train(
    model: BaseTransformer,
    args: argparse.Namespace,
    early_stopping_callback=None,
    logger=True,  # can pass WandbLogger() here
    extra_callbacks=[],
    checkpoint_callback=None,
    logging_callback=None,
    **extra_train_kwargs,
):
    pl.seed_everything(args.seed)

    odir = Path(model.hparams.output_dir)
    odir.mkdir(exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=odir,
        monitor="val_loss_epoch",
        mode="min",
        save_top_k=1,
    )

    train_params = {}

    train_params["accumulate_grad_batches"] = model.hparams.gradient_accumulation_steps
    train_params["precision"] = args.precision
    train_params["profiler"] = extra_train_kwargs.get("profiler", None)
    train_params["gradient_clip_val"] = args.gradient_clip_val
    train_params["val_check_interval"] = args.val_check_interval
    train_params["num_sanity_val_steps"] = args.num_sanity_val_steps
    train_params["max_epochs"] = args.num_train_epochs

    if model.hparams.local:
        train_params["precision"] = 32
        train_params["num_sanity_val_steps"] = 10
        train_params["gpus"] = 0
        train_params["strategy"] = None
    else:
        train_params["accelerator"] = "gpu"
        train_params["devices"] = args.devices

    if args.limit_val_batches is not None:
        train_params["limit_val_batches"] = args.limit_val_batches

    if args.resume_from_checkpoint is not None:
        train_params["resume_from_checkpoint"] = args.resume_from_checkpoint
    if args.debug_mode:
        train_params["limit_train_batches"] = 100
        train_params["limit_val_batches"] = 30

    if not args.local:
        with open(args.deepspeed_config, "r") as f:
            deepspeed_config = json.load(f)
        train_params["strategy"] = DeepSpeedStrategy(config=deepspeed_config)
    if args.logger_name == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        if args.id is not None:
            id_ = args.id
        else:
            id_ = wandb.util.generate_id()
        logger = WandbLogger(
            id=id_, name=args.wb_name, project=args.wb_project, entity=args.wb_entity
        )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        **train_params,
    )
    trainer.fit(model)

    return trainer
