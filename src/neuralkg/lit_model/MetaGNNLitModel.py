import dgl
import torch
from neuralkg.eval_task import *
from .BaseLitModel import BaseLitModel
from collections import defaultdict as ddict
from neuralkg.utils.tools import logging, log_metrics, get_g_bidir

class MetaGNNLitModel(BaseLitModel):

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

        logging.info("Train|loss: %.4f at s %d" %(loss, self.global_step+1))  #TODO: 把logging改到BaseLitModel里面
        return loss
    
    def validation_step(self, batch, batch_idx):
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
        
        if self.current_epoch!=0:
            logging.info("++++++++++++++++++++++++++start validating++++++++++++++++++++++++++")
            log_metrics(self.current_epoch+1, outputs)
            logging.info("++++++++++++++++++++++++++over validating+++++++++++++++++++++++++++")

        self.log_dict(outputs, prog_bar=True, on_epoch=True)
    


    def test_step(self, batch, batch_idx):
        ent_emb = self.model.get_ent_emb(self.model.indtest_train_g)

        batch_new = {}
        results = ddict(float)
        pos_triple = batch['positive_sample']
        tail_cand = batch['tail_cand']
        head_cand = batch['head_cand']
        batch['cand'] = None
        batch['ent_emb'] = ent_emb
        # head_label = torch.FloatTensor(np.zeros([tail_cand.shape[0], tail_cand.shape[1]], dtype=np.float32))
        # tail_label = torch.FloatTensor(np.zeros([head_cand.shape[0], head_cand.shape[1]], dtype=np.float32))

        # batch_new['positive_sample'] = pos_triple
        # batch_new['cand'] = None
        # batch_new['head_label'] = head_label
        # batch_new['tail_label'] = tail_label
        # batch_new['head_cand'] = head_cand
        # batch_new['tail_cand'] = tail_cand
        # batch_new['ent_emb'] = ent_emb
        # ranks = link_predict(batch_new, self.model, prediction='all')

        b_range = torch.arange(pos_triple.size()[0], device=self.args.gpu)
        target_idx = torch.zeros(pos_triple.size()[0], device=self.args.gpu, dtype=torch.int64)
        # tail prediction
        pred = self.model.get_score(batch, mode='tail_predict')
        tail_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                        dim=1, descending=False)[b_range, target_idx]
        # head prediction
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
        return results

    def test_epoch_end(self, results) -> None:
        outputs = self.get_results(results, "Eval")

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
