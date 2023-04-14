import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

    def init_emb(self):
        raise NotImplementedError

    def score_func(self, head_emb, relation_emb, tail_emb):
        raise NotImplementedError

    def forward(self, triples, negs, mode):
        raise NotImplementedError

    def tri2emb(self, triples, negs=None, mode="single"):
        """Get embedding of triples.
        
        This function get the embeddings of head, relation, and tail
        respectively. each embedding has three dimensions.

        Args:
            triples (tensor): This tensor save triples id, which dimension is 
                [triples number, 3].
            negs (tensor, optional): This tenosr store the id of the entity to 
                be replaced, which has one dimension. when negs is None, it is 
                in the test/eval phase. Defaults to None.
            mode (str, optional): This arg indicates that the negative entity 
                will replace the head or tail entity. when it is 'single', it 
                means that entity will not be replaced. Defaults to 'single'.

        Returns:
            head_emb: Head entity embedding.
            relation_emb: Relation embedding.
            tail_emb: Tail entity embedding.
        """
        if mode == "single":
            head_emb = self.ent_emb(triples[:, 0]).unsqueeze(1)  # [bs, 1, dim]
            relation_emb = self.rel_emb(triples[:, 1]).unsqueeze(1)  # [bs, 1, dim]
            tail_emb = self.ent_emb(triples[:, 2]).unsqueeze(1)  # [bs, 1, dim]

        elif mode == "head-batch" or mode == "head_predict":
            if negs is None:  # 说明这个时候是在evluation，所以需要直接用所有的entity embedding
                head_emb = self.ent_emb.weight.data.unsqueeze(0)  # [1, num_ent, dim]
            else:
                head_emb = self.ent_emb(negs)  # [bs, num_neg, dim]

            relation_emb = self.rel_emb(triples[:, 1]).unsqueeze(1)  # [bs, 1, dim]
            tail_emb = self.ent_emb(triples[:, 2]).unsqueeze(1)  # [bs, 1, dim]

        elif mode == "tail-batch" or mode == "tail_predict": 
            head_emb = self.ent_emb(triples[:, 0]).unsqueeze(1)  # [bs, 1, dim]
            relation_emb = self.rel_emb(triples[:, 1]).unsqueeze(1)  # [bs, 1, dim]

            if negs is None:
                tail_emb = self.ent_emb.weight.data.unsqueeze(0)  # [1, num_ent, dim]
            else:
                tail_emb = self.ent_emb(negs)  # [bs, num_neg, dim]
        return head_emb, relation_emb, tail_emb
