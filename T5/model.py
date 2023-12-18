from abc import ABC
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from data_module import MyLightningDataModule
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)


class MyLightningModel(pl.LightningModule, ABC):
    def __init__(
            self,
            tokenizer,
            model,
            save_name: str = "Default",
            lr: float = 0.0001,
            output_dir: str = "outputs",
            save_only_last_epoch: bool = False,
    ):
        super().__init__()
        self.save_name = save_name
        self.lr = lr
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.average_training_loss = None
        self.average_validation_loss = None
        self.save_only_last_epoch = save_only_last_epoch

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        """ forward step """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        return output.loss, output.logits

    def training_step(self, batch, batch_size):
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log(
            "train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
        )
        return loss

    def validation_step(self, batch, batch_size):
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log(
            "val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
        )

    def test_step(self, batch, batch_size):
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler}
        }

    def training_epoch_end(self, training_step_outputs):
        path = f"{self.output_dir}/{self.save_name}"
        if self.save_only_last_epoch:
            if self.current_epoch == self.trainer.max_epochs - 1:
                self.tokenizer.save_pretrained(path)
                self.model.save_pretrained(path)
        else:
            self.tokenizer.save_pretrained(path)
            self.model.save_pretrained(path)


class MyT5Model:
    def __init__(self,
                 save_name: str = "Default") -> None:
        self.save_name = save_name
        self.model = None
        self.T5_Model = None
        self.tokenizer = None
        self.device = None
        self.data_module = None

    def from_pretrained(self, model_name="t5-base") -> None:
        self.tokenizer = T5Tokenizer.from_pretrained(f"{model_name}")
        self.model = T5ForConditionalGeneration.from_pretrained(
            f"{model_name}", return_dict=True
        )

    def train(
            self,
            train_df: pd.DataFrame,
            eval_df: pd.DataFrame,
            source_max_token_len: int = 512,
            target_max_token_len: int = 512,
            batch_size: int = 8,
            max_epochs: int = 5,
            use_gpu: bool = True,
            output_dir: str = "outputs",
            early_stopping_patience_epochs: int = 2,
            precision=32,
            logger="default",
            dataloader_num_workers: int = 2,
            save_only_last_epoch: bool = False,
    ):
        self.data_module = MyLightningDataModule(
            train_df=train_df,
            test_df=eval_df,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            source_max_token_len=source_max_token_len,
            target_max_token_len=target_max_token_len,
            num_workers=dataloader_num_workers,
        )

        self.T5_Model = MyLightningModel(
            save_name=self.save_name,
            tokenizer=self.tokenizer,
            model=self.model,
            output_dir=output_dir,
            save_only_last_epoch=save_only_last_epoch,
        )

        callbacks = [TQDMProgressBar(refresh_rate=5)]

        if early_stopping_patience_epochs > 0:
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=early_stopping_patience_epochs,
                verbose=True,
                mode="min",
            )
            callbacks.append(early_stop_callback)

        gpus = 1 if use_gpu else 0

        loggers = True if logger == "default" else logger

        trainer = pl.Trainer(
            default_root_dir='./',
            logger=loggers,
            enable_checkpointing=False,
            callbacks=callbacks,
            max_epochs=max_epochs,
            gpus=gpus,
            precision=precision,
            log_every_n_steps=1,
        )

        trainer.fit(self.T5_Model, self.data_module)

    def load_model(
            self,
            model_dir: str = "outputs",
            use_gpu: bool = False
    ):
        self.model = T5ForConditionalGeneration.from_pretrained(f"{model_dir}")
        self.tokenizer = T5Tokenizer.from_pretrained(f"{model_dir}")

        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise "exception ---> no gpu found. set use_gpu=False, to use CPU"
        else:
            self.device = torch.device("cpu")

    def predict(
            self,
            source_text: str,
            max_length: int = 256,
            num_return_sequences: int = 1,
            num_beams: int = 4,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 2.5,
            length_penalty: float = 0.6,
            early_stopping: bool = True,
            skip_special_tokens: bool = True,
            clean_up_tokenization_spaces: bool = True,
    ):
        input_ids = self.tokenizer.encode(
            source_text,
            return_tensors="pt",
            add_special_tokens=True
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)
            input_ids = input_ids.to(self.device)

        generated_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=num_beams,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
        )

        predictions_text = [
            self.tokenizer.decode(
                every_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            for every_ids in generated_ids
        ]
        return predictions_text
