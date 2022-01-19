import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from IPython import embed
class RugE_Loss(nn.Module):
    def __init__(self,args, model):
        super(RugE_Loss, self).__init__()
        self.args = args
        self.model = model

    def forward(self, pos_score, neg_score, rule, confidence, triple_num, pos_len):
        entroy = nn.BCELoss()

        # 这段代码写的太简陋了 先跑通再说
        pos_label = torch.ones([pos_len, 1])
        neg_label = torch.zeros([pos_len, self.args.num_neg])
        one = torch.ones([1])
        zero = torch.zeros([1])
        pos_label = Variable(pos_label).to(self.args.gpu, dtype=torch.float)
        neg_label = Variable(neg_label).to(self.args.gpu, dtype=torch.float)
        one = Variable(one).to(self.args.gpu, dtype=torch.float)
        zero = Variable(zero).to(self.args.gpu, dtype=torch.float)

        sigmoid_neg = torch.sigmoid(neg_score)
        sigmoid_pos = torch.sigmoid(pos_score)

        postive_loss = entroy(sigmoid_pos, pos_label)
        negative_loss = entroy(sigmoid_neg, neg_label)

        pi_gradient = dict()
        # 感觉应该放在这个大函数的外面，不然每次被清空也没什么用
        sigmoid_value = dict()
        # 在计算每个grounding rule中的unlable的三元组对应的类似gradient
        for i in range(len(rule[0])):
            if triple_num[i] == 2:
                p1_rule = rule[0][i]
                unlabel_rule = rule[1][i]

                if p1_rule not in sigmoid_value:
                    p1_rule_score = self.model(p1_rule.unsqueeze(0))
                    sigmoid_rule = torch.sigmoid(p1_rule_score)
                    sigmoid_value[p1_rule] = sigmoid_rule
                else:
                    sigmoid_rule = sigmoid_value[p1_rule]

                if unlabel_rule not in pi_gradient:
                    pi_gradient[unlabel_rule] = self.args.slackness_penalty * confidence[i] * sigmoid_rule
                else:
                    pi_gradient[unlabel_rule] += self.args.slackness_penalty * confidence[i] * sigmoid_rule

            elif triple_num[i] == 3:
                p1_rule = rule[0][i]
                p2_rule = rule[1][i]
                unlabel_rule = rule[2][i]

                if p1_rule not in sigmoid_value:
                    p1_rule_score = self.model(p1_rule.unsqueeze(0))
                    sigmoid_rule = torch.sigmoid(p1_rule_score)
                    sigmoid_value[p1_rule] = sigmoid_rule
                else:
                    sigmoid_rule = sigmoid_value[p1_rule]

                if p2_rule not in sigmoid_value:
                    p2_rule_score = self.model(p2_rule.unsqueeze(0))
                    sigmoid_rule2 = torch.sigmoid(p2_rule_score)
                    sigmoid_value[p2_rule] = sigmoid_rule
                else:
                    sigmoid_rule2 = sigmoid_value[p2_rule]

                if unlabel_rule not in pi_gradient:
                    pi_gradient[unlabel_rule] = self.args.slackness_penalty * confidence[i] * sigmoid_rule * sigmoid_rule2
                else:
                    pi_gradient[unlabel_rule] += self.args.slackness_penalty * confidence[i] * sigmoid_rule * sigmoid_rule2

        unlabel_loss = 0.
        unlabel_triples = []
        gradient = []
        # 对于pi_gradient中的每个三元组（不重复）的 根据公式计算s函数
        for unlabel_triple in pi_gradient.keys():
            unlabel_triples.append(unlabel_triple.cpu().numpy())
            gradient.append(pi_gradient[unlabel_triple].cpu().detach().numpy())
        unlabel_triples = torch.tensor(unlabel_triples).to(self.args.gpu)
        gradient = torch.tensor(gradient).to(self.args.gpu).view(-1, 1)
        unlabel_triple_score = self.model(unlabel_triples)
        unlabel_triple_score = torch.sigmoid(unlabel_triple_score)
        unlabel_scores = []
        for i in range(0, len(gradient)):
            unlabel_score = (torch.min(torch.max(unlabel_triple_score[i] + gradient[i], zero), one)).cpu().detach().numpy()
            unlabel_scores.append(unlabel_score[0])
        unlabel_scores = torch.tensor(unlabel_scores).to(self.args.gpu)
        unlabel_scores = unlabel_scores.unsqueeze(1)
        unlabel_loss = entroy(unlabel_triple_score, unlabel_scores)
        # for unlabel_triple in pi_gradient.keys():
        #     unlabelrule_score = model(unlabel_triple.unsqueeze(0))
        #     sigmoid_unlabelrule = torch.sigmoid(unlabelrule_score)
        #     unlabel_score = torch.min(torch.max(sigmoid_unlabelrule + args.slackness_penalty * pi_gradient[unlabel_triple], zero), one)
        #     loss_part = entroy(sigmoid_unlabelrule, unlabel_score.to(args.gpu).detach())
        #     unlabel_loss = unlabel_loss + loss_part
        # 所有的grounding的unlbeled的两个值sigmoid和s函数都存在list中，需要转成tensor，然后一起计算loss

        loss = postive_loss + negative_loss + unlabel_loss

        if self.args.weight_decay != 0.0:
            #Use L2 regularization for ComplEx_NNE_AER
            ent_emb_all = self.model.ent_emb(torch.arange(self.args.num_ent).to(self.args.gpu))
            rel_emb_all = self.model.rel_emb(torch.arange(self.args.num_rel).to(self.args.gpu))
            regularization = self.args.weight_decay * (
                ent_emb_all.norm(p = 2)**2 + rel_emb_all.norm(p=2)**2
            )

        # print(postive_loss)
        # print(negative_loss)
        # print(unlabel_loss)
        loss += regularization
        return loss






