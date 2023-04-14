import torch
from neuralkg_ind.eval_task import *
from .BaseLitModel import BaseLitModel
from neuralkg_ind.utils.tools import logging, log_metrics

class indGNNLitModel(BaseLitModel):
    """Processing of inductive training, evaluation and testing.
    """

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
        """Getting samples, labels and training in inductive model.
        
        Args:
            batch: The training data.
            batch_idx: The dict_key in batch, type: list.

        Returns:
            loss: The training loss for back propagation.
        """
        pos_sample = batch["positive_sample"]
        neg_sample = batch["negative_sample"]
        pos_label = batch["positive_label"]
        neg_label = batch["negative_label"]

        pos_score = self.model((pos_sample, pos_label))
        neg_score = self.model((neg_sample, neg_label))

        loss = self.loss(pos_score, neg_score)

        self.log("Train|loss", loss,  on_step=False, on_epoch=True)

        logging.info("Train|loss: %.4f at epoch %d" %(loss, self.current_epoch+1))  #TODO: 把logging改到BaseLitModel里面
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Getting samples and validating in inductive model.
        
        Args:
            batch: The evalutaion data.
            batch_idx: The dict_key in batch, type: list.

        Returns:
            results: auc and auc_pr.
        """
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
        """Getting samples and test in inductive model.
        
        Args:
            batch: The evaluation data.
            batch_idx: The dict_key in batch, type: list.

        Returns:
            results: mrr and hits@1,5,10.
        """
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
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.l2)
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestones], gamma=0.1)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict
