import dgl
import torch
from neuralkg_ind.eval_task import *
from .BaseLitModel import BaseLitModel
from neuralkg_ind.utils.tools import logging, log_metrics

class SNRILitModel(BaseLitModel):

    def __init__(self, model, args):
        super().__init__(model, args)
        self.args = args

    def forward(self, x):
        return self.model(x)
    
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--lr", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        return parser
    
    def training_step(self, batch, batch_idx):

        pos_sample = batch["positive_sample"]
        neg_sample = batch["negative_sample"]
        pos_label = batch["positive_label"]
        neg_label = batch["negative_label"]

        dgi_loss = 0
  
        pos_score, s_G_pos, s_g_pos = self.model((pos_sample, pos_label), is_return_emb=True)
        neg_score = self.model((neg_sample, neg_label))
        _, _, s_g_cor = self.model((pos_sample, pos_label), is_return_emb=True, cor_graph=True)
        pos_sample = dgl.batch(pos_sample)
        lbl_1 = torch.ones(pos_sample.batch_size)
        lbl_2 = torch.zeros(pos_sample.batch_size)
        lbl = torch.cat((lbl_1, lbl_2)).type_as(pos_score)
        logits = self.model.get_logits(s_G_pos, s_g_pos, s_g_cor)
        self.b_xent = torch.nn.BCEWithLogitsLoss()
        dgi_loss = self.b_xent(logits, lbl)            

        loss = self.loss(pos_score, neg_score) + dgi_loss

        self.log("Train|loss", loss,  on_step=False, on_epoch=True)

        logging.info("Train|loss: %.4f at epoch %d" %(loss, self.current_epoch+1))
        return loss
    
    def validation_step(self, batch, batch_idx):

        results = dict()
        score = classification(batch, self.model)
        results.update(score)
        results["pos_labels"] = batch["graph_pos_label"]
        results["neg_labels"] = batch["graph_neg_label"]
        return results
    
    def validation_epoch_end(self, results) -> None:
        outputs = self.get_auc(results, "Eval")

        if self.current_epoch!=0:
            logging.info("++++++++++++++++++++++++++start validating++++++++++++++++++++++++++")
            log_metrics(self.current_epoch+1, outputs)
            logging.info("++++++++++++++++++++++++++over validating+++++++++++++++++++++++++++")

        self.log_dict(outputs, prog_bar=True, on_epoch=True)
    
    def test_step(self, batch, batch_idx):

        results = dict()
        if self.args.eval_task == 'link_prediction':
            ranks = link_predict(batch, self.model, prediction='ind')
            results["count"] = torch.numel(ranks)
            results["mrr"] = torch.sum(1.0 / ranks).item()
            for k in self.args.calc_hits:
                results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= k])

        elif self.args.eval_task == 'triple_classification':
            score = classification(batch, self.model)
            results.update(score)
            results["pos_labels"] = batch["graph_pos_label"]
            results["neg_labels"] = batch["graph_neg_label"]
        return results

    def test_epoch_end(self, results) -> None:
        if self.args.eval_task == 'link_prediction':
            outputs = self.get_results(results, "Test")
        elif self.args.eval_task == 'triple_classification':
            outputs = self.get_auc(results, "Test")

        if self.current_epoch!=0:
            logging.info("++++++++++++++++++++++++++start Test++++++++++++++++++++++++++")
            log_metrics(self.current_epoch+1, outputs)
            logging.info("++++++++++++++++++++++++++over Test+++++++++++++++++++++++++++")

        self.log_dict(outputs, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        """Setting optimizer and lr_scheduler.

        Returns:
            optim_dict: Record the optimizer and lr_scheduler, type: dict.   
        """
        milestones = int(self.args.max_epochs)
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.args.lr, weight_decay=5e-4)
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestones], gamma=0.1)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict
