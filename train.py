import numpy as np
import sys,os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import EncoderDecoderModel, BertConfig,EncoderDecoderConfig,BertTokenizer,BertModel
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from transformers import AdamW
from dataset import GenomicData
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from config import *

class TransformerGenomicModel(pl.LightningModule):
    def __init__(self, word_num, embedding_dim,batch_size):
        super().__init__()
        self.word_num = word_num
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.config_encoder = BertConfig(vocab_size=word_num, hidden_size=embedding_dim,
                                            num_hidden_layers=6,
                                            num_attention_heads=2,
                                            intermediate_size=512,
                                            output_hidden_states=False,
                                            output_attentions=False)#shape (bs, inp_len, inp_len)

        self.config_decoder = BertConfig(vocab_size=word_num, hidden_size=embedding_dim,
                                            num_hidden_layers=6,
                                            num_attention_heads=2,
                                            intermediate_size=512,
                                            output_hidden_states=True,
                                            output_attentions=False)#shape (bs, tar_len, tar_len)

        self.config = EncoderDecoderConfig.from_encoder_decoder_configs(self.config_encoder, self.config_decoder)
        self.encoder = BertModel(config=self.config_encoder)
        #self.seq2seq = EncoderDecoderModel(config=self.config)
        #self.fc1 = nn.Linear(word_num, 1)
        self.fc2 = nn.Linear(embedding_dim, 1)

    def forward(self,batch_inputs_embeds,batch_decoder_inputs_embeds,isSeq2Seq=False):
        if isSeq2Seq:
            x = self.seq2seq(inputs_embeds = batch_inputs_embeds, 
                                            decoder_inputs_embeds = batch_decoder_inputs_embeds)
            #outputs[0]: score before softmax with shape (batch_size, sequence_length, vocabulary_size)
            #outputs[1]: ((batch_size, sequence_length, embedding_dim),...)output embedding + output of each decoder layer
            x = self.fc1(x[0])    
        else:
            x = self.encoder(inputs_embeds=batch_inputs_embeds)
            x = self.fc2(x[0])
        output = F.relu(x)
        return output
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=LEARNING_RATE)
    def training_step(self,batch,batch_idx):
        batch_encoder_embeds, targets = batch
        batch_decoder_embeds = batch_encoder_embeds
        batch_pre = self.forward(batch_encoder_embeds,batch_decoder_embeds)
        loss_fn = nn.MSELoss(reduce=True, size_average=True)
        loss = loss_fn(batch_pre,targets)
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=False, logger=True, 
                                            on_step=True, on_epoch=True, sync_dist=True)
        return result

    def validation_step(self,batch,batch_idx):
        batch_encoder_embeds, targets = batch
        batch_decoder_embeds = batch_encoder_embeds
        batch_pre = self.forward(batch_encoder_embeds,batch_decoder_embeds)
        loss_fn = nn.MSELoss(reduce=True, size_average=True)
        loss = loss_fn(batch_pre,targets)
        result = pl.EvalResult(early_stop_on=loss, checkpoint_on=loss)
        result.log('val_loss', loss, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return result


    def setup(self,stage):
        dataset = GenomicData(100)
        train_size = int(0.95 * len(dataset))
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(dataset,
                                            [train_size, len(dataset) - train_size])

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=False,num_workers=14)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=14)

    # def test_dataloader(self):
    #     return DataLoader(self.dataset_test, batch_size=self.batch_size)                                                              
                                                                        





# model = TransformerGenomicModel(WORD_NUM,EMBEDDING_DIM)
# encoder_batch = torch.rand(2,3,4)
# decoder_batch = torch.rand(2,6,4)
# output = model(encoder_batch,decoder_batch)
# print (len(output))
# print (output)


def main(hparams):
    pl.seed_everything(hparams.seed)
    if hparams.train:
        model = TransformerGenomicModel(WORD_NUM,EMBEDDING_DIM,BATCH_SIZE)
        trainer = pl.Trainer(
            resume_from_checkpoint='checkpoint/epoch=43.ckpt',
            logger=pl_loggers.TensorBoardLogger(save_dir='logs',name='TensorBoard',version=3),
            checkpoint_callback=pl.callbacks.ModelCheckpoint(
                filepath='checkpoint', verbose=True, save_top_k=hparams.save_top_k),
            early_stop_callback=pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=True, mode='min'),
            default_root_dir=os.getcwd(),
            gpus=hparams.gpus,
            accumulate_grad_batches=2,
            distributed_backend='ddp',
            precision=16,
            log_gpu_memory='all')
        trainer.fit(model)
    else:
        model = TransformerGenomicModel(WORD_NUM,EMBEDDING_DIM,BATCH_SIZE)
        model.load_state_dict(torch.load('checkpoint/epoch=43.ckpt')['state_dict'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Reproduction")
    parser.add_argument('--gpus', type=int, default=1, help="How many gpus")
    parser.add_argument('--train', type=bool, default=False, help="train or load from pretrained model")
    parser.add_argument("--save_top_k", default=5, type=int,
                        help="The best k models according to the quantity monitored will be saved.")
    hparams = parser.parse_args()

    main(hparams)
