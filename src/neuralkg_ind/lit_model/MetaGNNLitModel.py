import dgl
import torch
from neuralkg_ind.eval_task import *
from .BaseLitModel import BaseLitModel
from collections import defaultdict as ddict
from neuralkg_ind.utils.tools import logging, log_metrics, get_g_bidir

class MetaGNNLitModel(BaseLitModel):
    """Processing of meta task training, evaluation and testing.
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
        """Getting meta tasks batch and training in meta model.
        
        Args:
            batch: The training data.
            batch_idx: The dict_key in batch, type: list.

        Returns:
            loss: The training loss for back propagation.
        """
        batch_loss = 0

        batch_sup_g = dgl.batch([get_g_bidir(d[0], self.args) for d in batch])
        self.model.get_ent_emb(batch_sup_g)
        sup_g_list = dgl.unbatch(batch_sup_g)
        for batch_i, data in enumerate(batch):
            que_tri, que_neg_tail_ent, que_neg_head_ent = [d.type_as(data[0]) for d in data[1:]]
            ent_emb = sup_g_list[batch_i].ndata['h']
            # kge loss
            neg_tail_score = self.model((que_tri, que_neg_tail_ent), ent_emb, mode='tail_predict')
            neg_head_score = self.model((que_tri, que_neg_head_ent), ent_emb, mode='head_predict')
            neg_score = torch.cat([neg_tail_score, neg_head_score])
            pos_score = self.model(que_tri, ent_emb)

            kge_loss = self.loss(pos_score, neg_score)
            batch_loss += kge_loss

        loss = batch_loss / len(batch)

        self.log("Train|loss", loss,  on_step=True, on_epoch=False)

        logging.info("Train|loss: %.4f at step %d" %(loss, self.global_step+1))  #TODO: 把logging改到BaseLitModel里面
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Getting meta tasks batch and validating in meta model.
        
        Args:
            batch: The evalutaion data.
            batch_idx: The dict_key in batch, type: list.

        Returns:
            results: mrr and hits@1,5,10.
        """
        batch_sup_g = dgl.batch([get_g_bidir(d[0], self.args) for d in batch])
        self.model.get_ent_emb(batch_sup_g)
        sup_g_list = dgl.unbatch(batch_sup_g)

        for batch_i, data in enumerate(batch):
            que_dataloader = data[1]
            ent_emb = sup_g_list[batch_i].ndata['h']

            count = 0
            results = ddict(float)
            results["count"] = 0
            for evl_batch in que_dataloader:
                batch_new = {}

                pos_triple, tail_label, head_label = [b.type_as(data[0]) for b in evl_batch]
                batch_new['positive_sample'] = pos_triple
                batch_new['cand'] = 'all'
                batch_new['head_label'] = head_label
                batch_new['tail_label'] = tail_label
                batch_new['ent_emb'] = ent_emb
                ranks = link_predict(batch_new, self.model, prediction='all')
                ranks = ranks.float()
                count += torch.numel(ranks)
                results['mr'] += torch.sum(ranks).item()
                results['mrr'] += torch.sum(1.0 / ranks).item()
                for k in self.args.calc_hits:
                    results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])
                    
        results["count"] += count
        return results
    
    def validation_epoch_end(self, results) -> None:
        outputs = self.get_results(results, "Eval")
        
        logging.info("++++++++++++++++++++++++++start validating++++++++++++++++++++++++++")
        for metric in outputs:
            logging.info('%s: %.4f at step %d' % (metric, outputs[metric], self.global_step + 1))
        logging.info("++++++++++++++++++++++++++over validating+++++++++++++++++++++++++++")

        self.log_dict(outputs, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """Getting meta tasks batch and test in meta model.
        
        Args:
            batch: The evaluation data.
            batch_idx: The dict_key in batch, type: list.

        Returns:
            results: mrr and hits@1,5,10 or auc and auc_pr.
        """
        indtest_train_g = self.model.get_intest_train_g()
        ent_emb = self.model.get_ent_emb(indtest_train_g)

        if self.args.eval_task == 'link_prediction':
            results = ddict(float)
            pos_triple = batch['positive_sample']
            batch['cand'] = None
            batch['ent_emb'] = ent_emb

            b_range = torch.arange(pos_triple.size()[0], device=self.args.gpu)
            target_idx = torch.zeros(pos_triple.size()[0], device=self.args.gpu, dtype=torch.int64)
            pred = self.model.get_score(batch, mode='tail_predict')
            tail_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                            dim=1, descending=False)[b_range, target_idx]

            pred = self.model.get_score(batch, mode='head_predict')
            head_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                            dim=1, descending=False)[b_range, target_idx]

            ranks = torch.cat([tail_ranks, head_ranks])
            ranks = ranks.float()
            results["count"] = torch.numel(ranks)
            results['mr'] += torch.sum(ranks).item()
            results['mrr'] += torch.sum(1.0 / ranks).item()
            for k in [1, 5, 10]:
                results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])
        
        elif self.args.eval_task == 'triple_classification':
            results = dict()
            pos_sample = batch["positive_sample"]
            score_pos = self.model(pos_sample, ent_emb)
            score_pos = score_pos.squeeze(1).detach().cpu().tolist()
            results["pos_scores"] = score_pos
            results["pos_labels"] = batch["positive_label"]

            neg_sample = batch["negative_sample"]
            score_neg = self.model(neg_sample, ent_emb)
            score_neg = score_neg.squeeze(1).detach().cpu().tolist()
            results["neg_scores"] = score_neg
            results["neg_labels"] = batch["negative_label"]

        return results

    def test_epoch_end(self, results) -> None:
        if self.args.eval_task == 'link_prediction':
            outputs = self.get_results(results, "Test")
        elif self.args.eval_task == 'triple_classification':
            outputs = self.get_auc(results, "Test")
        # outputs = self.get_results(results, "Eval")

        logging.info("++++++++++++++++++++++++++start Test++++++++++++++++++++++++++")
        for metric in outputs:
            logging.info('%s: %.4f at step %d' % (metric, outputs[metric], self.global_step))
        logging.info("++++++++++++++++++++++++++over Test+++++++++++++++++++++++++++")

        self.log_dict(outputs, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        """Setting optimizer and lr_scheduler.

        Returns:
            optim_dict: Record the optimizer and lr_scheduler, type: dict.   
        """
        milestones = int(self.args.max_epochs)
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.args.lr)
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestones], gamma=0.1)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict
