import dgl
import torch
from .BaseLitModel import BaseLitModel
from neuralkg_ind.eval_task import *

class SEGNNLitModel(BaseLitModel):
    def __init__(self, model, args, src_list, dst_list, rel_list):
        super().__init__(model, args)
        self.src_list = src_list
        self.dst_list = dst_list
        self.rel_list = rel_list
        self.kg = self.get_kg(src_list, dst_list, rel_list)
             
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        optimizer = self.optimizers()
        #optimizer = optimizer.optimizer
        optimizer.zero_grad()
        (head, rel, _), label, rm_edges= batch
        kg = self.get_kg(self.src_list, self.dst_list, self.rel_list)
        kg = kg.to(torch.device("cuda:0"))
        if self.args.rm_rate > 0:
            kg.remove_edges(rm_edges)
        score = self.model(head, rel, kg)
        loss = self.loss(score, label)
        self.manual_backward(loss)
        optimizer.step()
        sch = self.lr_schedulers()
        sch.step()
        
        return loss

    def validation_step(self, batch, batch_idx):
        # pos_triple, tail_label, head_label = batch
        results = dict()
        ranks = link_predict_SEGNN(batch, self.kg, self.model, prediction='tail')
        results["count"] =  torch.numel(ranks)
        #results['mr'] = results.get('mr', 0.) + ranks.sum().item()
        results['mrr'] = torch.sum(1.0 / ranks).item()
        for k in self.args.calc_hits:
            results['hits@{}'.format(k)] = torch.numel(ranks[ranks<=k])
        return results
    
    def validation_epoch_end(self, results) -> None:
        outputs = self.get_results(results, "Eval")
        # self.log("Eval|mrr", outputs["Eval|mrr"], on_epoch=True)
        self.log_dict(outputs, prog_bar=True, on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        results = dict()
        ranks   = link_predict_SEGNN(batch, self.kg, self.model, prediction='tail')
        results["count"] = torch.numel(ranks)
        #results['mr'] = results.get('MR', 0.) + ranks.sum().item()
        results['mrr'] = torch.sum(1.0 / ranks).item()
        for k in self.args.calc_hits:
            results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= k])
        return results
    

    def test_epoch_end(self, results) -> None:
        outputs = self.get_results(results, "Test")
        self.log_dict(outputs, prog_bar=True, on_epoch=True)
        
    def get_kg(self, src_list, dst_list, rel_list):
        n_ent = self.args.num_ent
        kg = dgl.graph((src_list, dst_list), num_nodes=n_ent)
        kg.edata['rel_id'] = rel_list
        return kg
    
    '''这里设置优化器和lr_scheduler'''
    def configure_optimizers(self):
        def lr_lambda(current_step):
            """
            Compute a ratio according to current step,
            by which the optimizer's lr will be mutiplied.
            :param current_step:
            :return:
            """
            assert current_step <= self.args.maxsteps
            if current_step < self.args.warm_up_steps:
                return current_step / self.args.warm_up_steps
            else:
                return (self.args.maxsteps - current_step) / (self.args.maxsteps - self.args.warm_up_steps)

        assert self.args.maxsteps >= self.args.warm_up_steps
        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = self.args.lr)
        #StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler':scheduler}
        
        return optim_dict
