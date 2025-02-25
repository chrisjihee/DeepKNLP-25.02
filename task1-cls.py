import logging
import os
from pathlib import Path
from time import sleep
from typing import List, Dict, Mapping, Any

import torch
import typer
from chrisbase.data import AppTyper, JobTimer, ProjectEnv
from chrisbase.io import LoggingFormat, make_dir, files
from chrisbase.util import mute_tqdm_cls, tupled
from flask import Flask, request, jsonify, render_template
from flask_classful import FlaskView, route
from lightning import LightningModule
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from DeepKNLP.arguments import DataFiles, DataOption, ModelOption, ServerOption, HardwareOption, PrintingOption, LearningOption
from DeepKNLP.arguments import TrainerArguments, TesterArguments, ServerArguments
from DeepKNLP.cls import ClassificationDataset, NsmcCorpus
from DeepKNLP.helper import CheckpointSaver, epsilon, data_collator, fabric_barrier
from DeepKNLP.metrics import accuracy

logger = logging.getLogger(__name__)
main = AppTyper()


class NSMCModel(LightningModule):
    def __init__(self, args: TrainerArguments | TesterArguments | ServerArguments):
        super().__init__()
        self.args: TrainerArguments | TesterArguments | ServerArguments = args
        self.data: NsmcCorpus = NsmcCorpus(args)

        assert self.data.num_labels > 0, f"Invalid num_labels: {self.data.num_labels}"
        self.lm_config: PretrainedConfig = AutoConfig.from_pretrained(
            args.model.pretrained,
            num_labels=self.data.num_labels
        )
        self.lm_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            args.model.pretrained,
            use_fast=True,
        )
        self.lang_model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            args.model.pretrained,
            config=self.lm_config,
        )

    def to_checkpoint(self) -> Dict[str, Any]:
        return {
            "lang_model": self.lang_model.state_dict(),
            "args_prog": self.args.prog,
        }

    def from_checkpoint(self, ckpt_state: Dict[str, Any]):
        self.lang_model.load_state_dict(ckpt_state["lang_model"])
        self.args.prog = ckpt_state["args_prog"]
        self.eval()

    def load_checkpoint_file(self, checkpoint_file):
        assert Path(checkpoint_file).exists(), f"Model file not found: {checkpoint_file}"
        self.fabric.print(f"Loading model from {checkpoint_file}")
        self.from_checkpoint(self.fabric.load(checkpoint_file))

    def load_last_checkpoint_file(self, checkpoints_glob):
        checkpoint_files = files(checkpoints_glob)
        assert checkpoint_files, f"No model file found: {checkpoints_glob}"
        self.load_checkpoint_file(checkpoint_files[-1])

    def configure_optimizers(self):
        return AdamW(self.lang_model.parameters(), lr=self.args.learning.learning_rate)

    def train_dataloader(self):
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug
        train_dataset = ClassificationDataset("train", data=self.data, tokenizer=self.lm_tokenizer)
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset, replacement=False),
                                      num_workers=self.args.hardware.cpu_workers,
                                      batch_size=self.args.hardware.train_batch,
                                      collate_fn=data_collator,
                                      drop_last=False)
        self.fabric.print(f"Created train_dataset providing {len(train_dataset)} examples")
        self.fabric.print(f"Created train_dataloader providing {len(train_dataloader)} batches")
        return train_dataloader

    def val_dataloader(self):
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug
        val_dataset = ClassificationDataset("valid", data=self.data, tokenizer=self.lm_tokenizer)
        val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset),
                                      num_workers=self.args.hardware.cpu_workers,
                                      batch_size=self.args.hardware.infer_batch,
                                      collate_fn=data_collator,
                                      drop_last=False)
        self.fabric.print(f"Created val_dataset providing {len(val_dataset)} examples")
        self.fabric.print(f"Created val_dataloader providing {len(val_dataloader)} batches")
        return val_dataloader

    def test_dataloader(self):
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug
        test_dataset = ClassificationDataset("test", data=self.data, tokenizer=self.lm_tokenizer)
        test_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset),
                                    num_workers=self.args.hardware.cpu_workers,
                                    batch_size=self.args.hardware.infer_batch,
                                    collate_fn=data_collator,
                                    drop_last=False)
        self.fabric.print(f"Created test_dataset providing {len(test_dataset)} examples")
        self.fabric.print(f"Created test_dataloader providing {len(test_dataloader)} batches")
        return test_dataloader

    def training_step(self, inputs, batch_idx):
        outputs: SequenceClassifierOutput = self.lang_model(**inputs)
        labels: torch.Tensor = inputs["labels"]
        preds: torch.Tensor = outputs.logits.argmax(dim=-1)
        acc: torch.Tensor = accuracy(preds=preds, labels=labels)
        return {
            "loss": outputs.loss,
            "acc": acc,
        }

    @torch.no_grad()
    def validation_step(self, inputs, batch_idx):
        outputs: SequenceClassifierOutput = self.lang_model(**inputs)
        labels: List[int] = inputs["labels"].tolist()
        preds: List[int] = outputs.logits.argmax(dim=-1).tolist()
        return {
            "loss": outputs.loss,
            "preds": preds,
            "labels": labels
        }

    @torch.no_grad()
    def test_step(self, inputs, batch_idx):
        return self.validation_step(inputs, batch_idx)

    @torch.no_grad()
    def infer_one(self, text: str):
        inputs = self.lm_tokenizer(
            tupled(text),
            max_length=self.args.model.seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        outputs: SequenceClassifierOutput = self.lang_model(**inputs)
        prob = outputs.logits.softmax(dim=1)
        pred = "긍정 (positive)" if torch.argmax(prob) == 1 else "부정 (negative)"
        positive_prob = round(prob[0][1].item(), 4)
        negative_prob = round(prob[0][0].item(), 4)
        return {
            'sentence': text,
            'prediction': pred,
            'positive_data': f"긍정 {positive_prob * 100:.1f}%",
            'negative_data': f"부정 {negative_prob * 100:.1f}%",
            'positive_width': f"{positive_prob * 100:.2f}%",
            'negative_width': f"{negative_prob * 100:.2f}%",
        }

    def run_server(self, server: Flask, *args, **kwargs):
        NSMCModel.WebAPI.register(route_base='/', app=server, init_argument=self)
        server.run(*args, **kwargs)

    class WebAPI(FlaskView):
        def __init__(self, model: "NSMCModel"):
            self.model = model

        @route('/')
        def index(self):
            return render_template(self.model.args.server.page)

        @route('/api', methods=['POST'])
        def api(self):
            response = self.model.infer_one(text=request.json)
            return jsonify(response)


def train_loop(
        model: NSMCModel,
        optimizer: OptimizerLRScheduler,
        dataloader: DataLoader,
        val_dataloader: DataLoader,
        checkpoint_saver: CheckpointSaver | None = None,
):
    fabric = model.fabric
    fabric.barrier()
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug
    num_batch = len(dataloader)
    print_interval = model.args.printing.print_rate_on_training * num_batch - epsilon if model.args.printing.print_step_on_training < 1 else model.args.printing.print_step_on_training
    check_interval = model.args.learning.check_rate_on_training * num_batch - epsilon
    model.args.prog.global_step = 0
    model.args.prog.global_epoch = 0.0
    for epoch in range(model.args.learning.num_epochs):
        progress = mute_tqdm_cls(bar_size=30, desc_size=8)(range(num_batch), unit=f"x{dataloader.batch_size}b", desc="training")
        for i, batch in enumerate(dataloader, start=1):
            model.train()
            model.args.prog.global_step += 1
            model.args.prog.global_epoch = model.args.prog.global_step / num_batch
            optimizer.zero_grad()
            outputs = model.training_step(batch, i)
            fabric.backward(outputs["loss"])
            optimizer.step()
            progress.update()
            fabric.barrier()
            with torch.no_grad():
                model.eval()
                metrics: Mapping[str, Any] = {
                    "step": round(fabric.all_gather(torch.tensor(model.args.prog.global_step * 1.0)).mean().item()),
                    "epoch": round(fabric.all_gather(torch.tensor(model.args.prog.global_epoch)).mean().item(), 4),
                    "loss": fabric.all_gather(outputs["loss"]).mean().item(),
                    "acc": fabric.all_gather(outputs["acc"]).mean().item(),
                }
                fabric.log_dict(metrics=metrics, step=metrics["step"])
                if i % print_interval < 1:
                    fabric.print(f"(Ep {model.args.prog.global_epoch:4.2f}) {progress}"
                                 f" | {model.args.printing.tag_format_on_training.format(**metrics)}")
                if model.args.prog.global_step % check_interval < 1:
                    val_loop(model, val_dataloader, checkpoint_saver)
        fabric_barrier(fabric, "[after-epoch]", c='=')
    fabric_barrier(fabric, "[after-train]")


@torch.no_grad()
def val_loop(
        model: NSMCModel,
        dataloader: DataLoader,
        checkpoint_saver: CheckpointSaver | None = None,
):
    fabric = model.fabric
    fabric.barrier()
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug
    num_batch = len(dataloader)
    print_interval = model.args.printing.print_rate_on_validate * num_batch - epsilon if model.args.printing.print_step_on_validate < 1 else model.args.printing.print_step_on_validate
    preds: List[int] = []
    labels: List[int] = []
    losses: List[torch.Tensor] = []
    progress = mute_tqdm_cls(bar_size=20, desc_size=8)(range(num_batch), unit=f"x{dataloader.batch_size}b", desc="checking")
    for i, batch in enumerate(dataloader, start=1):
        outputs = model.validation_step(batch, i)
        preds.extend(outputs["preds"])
        labels.extend(outputs["labels"])
        losses.append(outputs["loss"])
        progress.update()
        if i < num_batch and i % print_interval < 1:
            fabric.print(f"(Ep {model.args.prog.global_epoch:4.2f}) {progress}")
    fabric.barrier()
    all_preds: torch.Tensor = fabric.all_gather(torch.tensor(preds)).flatten()
    all_labels: torch.Tensor = fabric.all_gather(torch.tensor(labels)).flatten()
    metrics: Mapping[str, Any] = {
        "step": round(fabric.all_gather(torch.tensor(model.args.prog.global_step * 1.0)).mean().item()),
        "epoch": round(fabric.all_gather(torch.tensor(model.args.prog.global_epoch)).mean().item(), 4),
        "val_loss": fabric.all_gather(torch.stack(losses)).mean().item(),
        "val_acc": accuracy(all_preds, all_labels).item(),
    }
    fabric.log_dict(metrics=metrics, step=metrics["step"])
    fabric.print(f"(Ep {model.args.prog.global_epoch:4.2f}) {progress}"
                 f" | {model.args.printing.tag_format_on_validate.format(**metrics)}")
    fabric_barrier(fabric, "[after-check]")
    if checkpoint_saver:
        checkpoint_saver.save_checkpoint(metrics=metrics, ckpt_state=model.to_checkpoint())


@torch.no_grad()
def test_loop(
        model: NSMCModel,
        dataloader: DataLoader,
        checkpoint_path: str | Path | None = None,
):
    if checkpoint_path:
        model.load_checkpoint_file(checkpoint_path)
    fabric = model.fabric
    fabric.barrier()
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug
    num_batch = len(dataloader)
    print_interval = model.args.printing.print_rate_on_evaluate * num_batch - epsilon if model.args.printing.print_step_on_evaluate < 1 else model.args.printing.print_step_on_evaluate
    preds: List[int] = []
    labels: List[int] = []
    losses: List[torch.Tensor] = []
    progress = mute_tqdm_cls(bar_size=20, desc_size=8)(range(num_batch), unit=f"x{dataloader.batch_size}b", desc="testing")
    for i, batch in enumerate(dataloader, start=1):
        outputs = model.test_step(batch, i)
        preds.extend(outputs["preds"])
        labels.extend(outputs["labels"])
        losses.append(outputs["loss"])
        progress.update()
        if i < num_batch and i % print_interval < 1:
            fabric.print(f"(Ep {model.args.prog.global_epoch:4.2f}) {progress}")
    fabric.barrier()
    all_preds: torch.Tensor = fabric.all_gather(torch.tensor(preds)).flatten()
    all_labels: torch.Tensor = fabric.all_gather(torch.tensor(labels)).flatten()
    metrics: Mapping[str, Any] = {
        "step": round(fabric.all_gather(torch.tensor(model.args.prog.global_step * 1.0)).mean().item()),
        "epoch": round(fabric.all_gather(torch.tensor(model.args.prog.global_epoch)).mean().item(), 4),
        "test_loss": fabric.all_gather(torch.stack(losses)).mean().item(),
        "test_acc": accuracy(all_preds, all_labels).item(),
    }
    fabric.log_dict(metrics=metrics, step=metrics["step"])
    fabric.print(f"(Ep {model.args.prog.global_epoch:4.2f}) {progress}"
                 f" | {model.args.printing.tag_format_on_evaluate.format(**metrics)}")
    fabric_barrier(fabric, "[after-test]")


@main.command()
def train(
        verbose: int = typer.Option(default=2),
        # env
        project: str = typer.Option(default="DeepKNLP"),
        job_name: str = typer.Option(default=None),
        job_version: int = typer.Option(default=None),
        debugging: bool = typer.Option(default=False),
        logging_file: str = typer.Option(default="logging.out"),
        argument_file: str = typer.Option(default="arguments.json"),
        # data
        data_home: str = typer.Option(default="data"),
        data_name: str = typer.Option(default="nsmc"),
        train_file: str = typer.Option(default="ratings_train.txt"),
        valid_file: str = typer.Option(default="ratings_valid.txt"),
        test_file: str = typer.Option(default="ratings_test.txt"),
        num_check: int = typer.Option(default=3),
        # model
        pretrained: str = typer.Option(default="beomi/KcELECTRA-base"),
        finetuning: str = typer.Option(default="output"),
        model_name: str = typer.Option(default=None),
        seq_len: int = typer.Option(default=128),  # TODO: -> 64, 128, 256, 512
        # hardware
        cpu_workers: int = typer.Option(default=min(os.cpu_count() / 2, 10)),
        train_batch: int = typer.Option(default=50),
        infer_batch: int = typer.Option(default=50),
        accelerator: str = typer.Option(default="cuda"),  # TODO: -> cuda, cpu, mps
        precision: str = typer.Option(default="16-mixed"),  # TODO: -> 32-true, bf16-mixed, 16-mixed
        strategy: str = typer.Option(default="ddp"),
        device: List[int] = typer.Option(default=[0]),  # TODO: -> [0], [0,1], [0,1,2,3]
        # printing
        print_rate_on_training: float = typer.Option(default=1 / 20),  # TODO: -> 1/10, 1/20, 1/40, 1/100
        print_rate_on_validate: float = typer.Option(default=1 / 2),  # TODO: -> 1/2, 1/3
        print_rate_on_evaluate: float = typer.Option(default=1 / 2),  # TODO: -> 1/2, 1/3
        print_step_on_training: int = typer.Option(default=-1),
        print_step_on_validate: int = typer.Option(default=-1),
        print_step_on_evaluate: int = typer.Option(default=-1),
        tag_format_on_training: str = typer.Option(default="st={step:d}, ep={epoch:.2f}, loss={loss:06.4f}, acc={acc:06.4f}"),
        tag_format_on_validate: str = typer.Option(default="st={step:d}, ep={epoch:.2f}, val_loss={val_loss:06.4f}, val_acc={val_acc:06.4f}"),
        tag_format_on_evaluate: str = typer.Option(default="st={step:d}, ep={epoch:.2f}, test_loss={test_loss:06.4f}, test_acc={test_acc:06.4f}"),
        # learning
        learning_rate: float = typer.Option(default=5e-5),
        random_seed: int = typer.Option(default=7),
        saving_mode: str = typer.Option(default="max val_acc"),
        num_saving: int = typer.Option(default=1),  # TODO: -> 2, 3
        num_epochs: int = typer.Option(default=2),  # TODO: -> 2, 3
        check_rate_on_training: float = typer.Option(default=1 / 5),  # TODO: -> 1/5, 1/10
        name_format_on_saving: str = typer.Option(default="ep={epoch:.1f}, loss={val_loss:06.4f}, acc={val_acc:06.4f}"),
):
    torch.set_float32_matmul_precision('high')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.getLogger("c10d-NullHandler").setLevel(logging.INFO)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)

    pretrained = Path(pretrained)
    args = TrainerArguments(
        env=ProjectEnv(
            project=project,
            job_name=job_name if job_name else pretrained.name,
            job_version=job_version,
            debugging=debugging,
            message_level=logging.DEBUG if debugging else logging.INFO,
            message_format=LoggingFormat.DEBUG_20 if debugging else LoggingFormat.CHECK_20,
        ),
        data=DataOption(
            home=data_home,
            name=data_name,
            files=DataFiles(
                train=train_file,
                valid=valid_file,
                test=test_file,
            ),
            num_check=num_check,
        ),
        model=ModelOption(
            pretrained=pretrained,
            finetuning=finetuning,
            name=model_name,
            seq_len=seq_len,
        ),
        hardware=HardwareOption(
            cpu_workers=cpu_workers,
            train_batch=train_batch,
            infer_batch=infer_batch,
            accelerator=accelerator,
            precision=precision,
            strategy=strategy,
            devices=device,
        ),
        printing=PrintingOption(
            print_rate_on_training=print_rate_on_training,
            print_rate_on_validate=print_rate_on_validate,
            print_rate_on_evaluate=print_rate_on_evaluate,
            print_step_on_training=print_step_on_training,
            print_step_on_validate=print_step_on_validate,
            print_step_on_evaluate=print_step_on_evaluate,
            tag_format_on_training=tag_format_on_training,
            tag_format_on_validate=tag_format_on_validate,
            tag_format_on_evaluate=tag_format_on_evaluate,
        ),
        learning=LearningOption(
            learning_rate=learning_rate,
            random_seed=random_seed,
            saving_mode=saving_mode,
            num_saving=num_saving,
            num_epochs=num_epochs,
            check_rate_on_training=check_rate_on_training,
            name_format_on_saving=name_format_on_saving,
        ),
    )
    finetuning_home = Path(f"{finetuning}/{data_name}")
    output_name = f"{args.tag}={args.env.job_name}={args.env.hostname}" if not args.model.name else args.model.name
    make_dir(finetuning_home / output_name)
    args.env.job_version = args.env.job_version if args.env.job_version else CSVLogger(finetuning_home, output_name).version
    args.prog.tb_logger = TensorBoardLogger(finetuning_home, output_name, args.env.job_version)  # tensorboard --logdir finetuning --bind_all
    args.prog.csv_logger = CSVLogger(finetuning_home, output_name, args.env.job_version, flush_logs_every_n_steps=1)
    sleep(0.3)
    fabric = Fabric(
        loggers=[args.prog.tb_logger, args.prog.csv_logger],
        devices=args.hardware.devices if args.hardware.accelerator in ["cuda", "gpu"] else args.hardware.cpu_workers if args.hardware.accelerator == "cpu" else "auto",
        strategy=args.hardware.strategy if args.hardware.accelerator in ["cuda", "gpu"] else "auto",
        precision=args.hardware.precision if args.hardware.accelerator in ["cuda", "gpu"] else None,
        accelerator=args.hardware.accelerator,
    )
    fabric.launch()
    fabric.barrier()
    job_versions = fabric.all_gather(torch.tensor(args.env.job_version))
    assert job_versions.min() == job_versions.max(), f"Job version must be same across all processes: {job_versions.tolist()}"
    sleep(fabric.global_rank * 0.3)
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug
    args.env.set_logging_home(args.prog.csv_logger.log_dir)
    args.env.set_logging_file(logging_file)
    args.env.set_argument_file(argument_file)
    args.prog.world_size = fabric.world_size
    args.prog.local_rank = fabric.local_rank
    args.prog.global_rank = fabric.global_rank
    fabric.seed_everything(args.learning.random_seed)
    fabric.barrier()

    with JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", rt=1, rb=1, rc='=',
                  args=args if (debugging or verbose > 1) and fabric.local_rank == 0 else None,
                  verbose=verbose > 0 and fabric.local_rank == 0,
                  mute_warning="lightning.fabric.loggers.csv_logs"):
        model = NSMCModel(args=args)
        optimizer = model.configure_optimizers()
        model, optimizer = fabric.setup(model, optimizer)
        fabric_barrier(fabric, "[after-model]", c='=')

        assert args.data.files.train, "No training file found"
        train_dataloader = model.train_dataloader()
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
        fabric_barrier(fabric, "[after-train_dataloader]", c='=')
        assert args.data.files.valid, "No validation file found"
        val_dataloader = model.val_dataloader()
        val_dataloader = fabric.setup_dataloaders(val_dataloader)
        fabric_barrier(fabric, "[after-val_dataloader]", c='=')
        checkpoint_saver = CheckpointSaver(
            fabric=fabric,
            output_home=model.args.env.logging_home,
            name_format=model.args.learning.name_format_on_saving,
            saving_mode=model.args.learning.saving_mode,
            num_saving=model.args.learning.num_saving,
        )
        train_loop(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            checkpoint_saver=checkpoint_saver,
        )

        if args.data.files.test:
            test_dataloader = model.test_dataloader()
            test_dataloader = fabric.setup_dataloaders(test_dataloader)
            fabric_barrier(fabric, "[after-test_dataloader]", c='=')
            test_loop(
                model=model,
                dataloader=test_dataloader,
                checkpoint_path=checkpoint_saver.best_model_path,
            )


@main.command()
def test(
        verbose: int = typer.Option(default=2),
        # env
        project: str = typer.Option(default="DeepKNLP"),
        job_name: str = typer.Option(default=None),
        job_version: int = typer.Option(default=None),
        debugging: bool = typer.Option(default=False),
        logging_file: str = typer.Option(default="logging.out"),
        argument_file: str = typer.Option(default="arguments.json"),
        # data
        data_home: str = typer.Option(default="data"),
        data_name: str = typer.Option(default="nsmc"),
        test_file: str = typer.Option(default="ratings_valid.txt"),
        num_check: int = typer.Option(default=3),
        # model
        pretrained: str = typer.Option(default="beomi/KcELECTRA-base"),
        finetuning: str = typer.Option(default="output"),
        model_name: str = typer.Option(default="train=*"),
        seq_len: int = typer.Option(default=128),  # TODO: -> 64, 128, 256, 512
        # hardware
        cpu_workers: int = typer.Option(default=min(os.cpu_count() / 2, 10)),
        infer_batch: int = typer.Option(default=10),
        accelerator: str = typer.Option(default="cuda"),  # TODO: -> cuda, cpu, mps
        precision: str = typer.Option(default=None),  # TODO: -> 32-true, bf16-mixed, 16-mixed
        strategy: str = typer.Option(default="auto"),
        device: List[int] = typer.Option(default=[0]),
        # printing
        print_rate_on_evaluate: float = typer.Option(default=1 / 10),  # TODO: -> 1/2, 1/3, 1/5, 1/10, 1/50, 1/100
        print_step_on_evaluate: int = typer.Option(default=-1),
        tag_format_on_evaluate: str = typer.Option(default="st={step:d}, ep={epoch:.2f}, test_loss={test_loss:06.4f}, test_acc={test_acc:06.4f}"),
):
    torch.set_float32_matmul_precision('high')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.getLogger("c10d-NullHandler").setLevel(logging.INFO)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)

    pretrained = Path(pretrained)
    args = TesterArguments(
        env=ProjectEnv(
            project=project,
            job_name=job_name if job_name else pretrained.name,
            job_version=job_version,
            debugging=debugging,
            message_level=logging.DEBUG if debugging else logging.INFO,
            message_format=LoggingFormat.DEBUG_20 if debugging else LoggingFormat.CHECK_20,
        ),
        data=DataOption(
            home=data_home,
            name=data_name,
            files=DataFiles(
                test=test_file,
            ),
            num_check=num_check,
        ),
        model=ModelOption(
            pretrained=pretrained,
            finetuning=finetuning,
            name=model_name,
            seq_len=seq_len,
        ),
        hardware=HardwareOption(
            cpu_workers=cpu_workers,
            infer_batch=infer_batch,
            accelerator=accelerator,
            precision=precision,
            strategy=strategy,
            devices=device,
        ),
        printing=PrintingOption(
            print_rate_on_evaluate=print_rate_on_evaluate,
            print_step_on_evaluate=print_step_on_evaluate,
            tag_format_on_evaluate=tag_format_on_evaluate,
        ),
    )
    finetuning_home = Path(f"{finetuning}/{data_name}")
    output_name = f"{args.tag}={args.env.job_name}={args.env.hostname}"
    make_dir(finetuning_home / output_name)
    args.env.job_version = args.env.job_version if args.env.job_version else CSVLogger(finetuning_home, output_name).version
    args.prog.csv_logger = CSVLogger(finetuning_home, output_name, args.env.job_version, flush_logs_every_n_steps=1)
    sleep(0.3)
    fabric = Fabric(
        devices=args.hardware.devices if args.hardware.accelerator in ["cuda", "gpu"] else args.hardware.cpu_workers if args.hardware.accelerator == "cpu" else "auto",
        strategy=args.hardware.strategy if args.hardware.accelerator in ["cuda", "gpu"] else "auto",
        precision=args.hardware.precision if args.hardware.accelerator in ["cuda", "gpu"] else None,
        accelerator=args.hardware.accelerator,
    )
    fabric.launch()
    fabric.barrier()
    job_versions = fabric.all_gather(torch.tensor(args.env.job_version))
    assert job_versions.min() == job_versions.max(), f"Job version must be same across all processes: {job_versions.tolist()}"
    sleep(fabric.global_rank * 0.3)
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug
    args.env.set_logging_home(args.prog.csv_logger.log_dir)
    args.env.set_logging_file(logging_file)
    args.env.set_argument_file(argument_file)
    args.prog.world_size = fabric.world_size
    args.prog.local_rank = fabric.local_rank
    args.prog.global_rank = fabric.global_rank
    fabric.barrier()

    with JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", rt=1, rb=1, rc='=',
                  args=args if (debugging or verbose > 1) and fabric.local_rank == 0 else None,
                  verbose=verbose > 0 and fabric.local_rank == 0,
                  mute_warning="lightning.fabric.loggers.csv_logs"):
        model = NSMCModel(args=args)
        model = fabric.setup(model)
        fabric_barrier(fabric, "[after-model]", c='=')

        assert args.data.files.test, "No test file found"
        test_dataloader = model.test_dataloader()
        test_dataloader = fabric.setup_dataloaders(test_dataloader)
        fabric_barrier(fabric, "[after-test_dataloader]", c='=')

        for checkpoint_path in files(finetuning_home / args.model.name / "**/*.ckpt"):
            test_loop(
                model=model,
                dataloader=test_dataloader,
                checkpoint_path=checkpoint_path,
            )


@main.command()
def serve(
        verbose: int = typer.Option(default=2),
        # env
        project: str = typer.Option(default="DeepKNLP"),
        job_name: str = typer.Option(default=None),
        job_version: int = typer.Option(default=None),
        debugging: bool = typer.Option(default=False),
        logging_file: str = typer.Option(default="logging.out"),
        argument_file: str = typer.Option(default="arguments.json"),
        # data
        data_name: str = typer.Option(default="nsmc"),
        # model
        pretrained: str = typer.Option(default="beomi/KcELECTRA-base"),
        finetuning: str = typer.Option(default="output"),
        model_name: str = typer.Option(default="train=*"),
        seq_len: int = typer.Option(default=128),  # TODO: -> 64, 128, 256, 512
        # server
        server_port: int = typer.Option(default=9164),
        server_host: str = typer.Option(default="0.0.0.0"),
        server_temp: str = typer.Option(default="templates"),
        server_page: str = typer.Option(default="serve_cls.html"),
):
    torch.set_float32_matmul_precision('high')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.getLogger("c10d-NullHandler").setLevel(logging.INFO)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)

    pretrained = Path(pretrained)
    args = ServerArguments(
        env=ProjectEnv(
            project=project,
            job_name=job_name if job_name else pretrained.name,
            job_version=job_version,
            debugging=debugging,
            message_level=logging.DEBUG if debugging else logging.INFO,
            message_format=LoggingFormat.DEBUG_20 if debugging else LoggingFormat.CHECK_20,
        ),
        data=DataOption(
            name=data_name,
        ),
        model=ModelOption(
            pretrained=pretrained,
            finetuning=finetuning,
            name=model_name,
            seq_len=seq_len,
        ),
        server=ServerOption(
            port=server_port,
            host=server_host,
            temp=server_temp,
            page=server_page,
        )
    )
    finetuning_home = Path(f"{finetuning}/{data_name}")
    output_name = f"{args.tag}={args.env.job_name}={args.env.hostname}"
    make_dir(finetuning_home / output_name)
    args.env.job_version = args.env.job_version if args.env.job_version else CSVLogger(finetuning_home, output_name).version
    args.prog.csv_logger = CSVLogger(finetuning_home, output_name, args.env.job_version, flush_logs_every_n_steps=1)
    fabric = Fabric(devices=1, accelerator="cpu")
    fabric.print = logger.info
    args.env.set_logging_home(args.prog.csv_logger.log_dir)
    args.env.set_logging_file(logging_file)
    args.env.set_argument_file(argument_file)

    with JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", rt=1, rb=1, rc='=',
                  args=args if (debugging or verbose > 1) else None, verbose=verbose > 0,
                  mute_warning="lightning.fabric.loggers.csv_logs"):
        model = NSMCModel(args=args)
        model = fabric.setup(model)
        model.load_last_checkpoint_file(finetuning_home / args.model.name / "**/*.ckpt")
        fabric_barrier(fabric, "[after-model]", c='=')

        model.run_server(server=Flask(output_name, template_folder=args.server.temp),
                         host=args.server.host, port=args.server.port, debug=debugging)


if __name__ == "__main__":
    main()
