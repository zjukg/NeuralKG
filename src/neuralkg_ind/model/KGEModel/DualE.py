import torch.nn as nn
import torch
from .model import Model
from numpy.random import RandomState
import numpy as np


class DualE(Model):
    """`Dual Quaternion Knowledge Graph Embeddings`_ (DualE), which introduces dual quaternions into knowledge graph embeddings.

    Attributes:
        args: Model configuration parameters.
        ent_emb: Entity embedding, shape:[num_ent, emb_dim * 8].
        rel_emb: Relation embedding, shape:[num_rel, emb_dim * 8].
    
    .. Dual Quaternion Knowledge Graph Embeddings: https://ojs.aaai.org/index.php/AAAI/article/view/16850
    """
    def __init__(self, args):
        super(DualE, self).__init__(args)
        self.args = args
        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim*8)
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim*8)
        self.criterion = nn.Softplus()
        self.fc = nn.Linear(100, 50, bias=False)
        self.ent_dropout = torch.nn.Dropout(0)
        self.rel_dropout = torch.nn.Dropout(0)
        self.bn = torch.nn.BatchNorm1d(self.args.emb_dim)
        self.init_weights()

    def init_weights(self):
        r, i, j, k,r_1,i_1,j_1,k_1 = self.quaternion_init(self.args.num_ent, self.args.emb_dim)
        r, i, j, k,r_1,i_1,j_1,k_1 = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k),\
                                    torch.from_numpy(r_1), torch.from_numpy(i_1), torch.from_numpy(j_1), torch.from_numpy(k_1)
        tmp_ent_emb = torch.cat((r, i, j, k,r_1,i_1,j_1,k_1),1)
        self.ent_emb.weight.data = tmp_ent_emb.type_as(self.ent_emb.weight.data)

        s, x, y, z,s_1,x_1,y_1,z_1 = self.quaternion_init(self.args.num_ent, self.args.emb_dim)
        s, x, y, z,s_1,x_1,y_1,z_1 = torch.from_numpy(s), torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z), \
                                    torch.from_numpy(s_1), torch.from_numpy(x_1), torch.from_numpy(y_1), torch.from_numpy(z_1)
        tmp_rel_emb = torch.cat((s, x, y, z,s_1,x_1,y_1,z_1),1)
        self.rel_emb.weight.data = tmp_rel_emb.type_as(self.ent_emb.weight.data)

    #Calculate the Dual Hamiltonian product
    def _omult(self, a_0, a_1, a_2, a_3, b_0, b_1, b_2, b_3, c_0, c_1, c_2, c_3, d_0, d_1, d_2, d_3):
        h_0=a_0*c_0-a_1*c_1-a_2*c_2-a_3*c_3
        h1_0=a_0*d_0+b_0*c_0-a_1*d_1-b_1*c_1-a_2*d_2-b_2*c_2-a_3*d_3-b_3*c_3
        h_1=a_0*c_1+a_1*c_0+a_2*c_3-a_3*c_2
        h1_1=a_0*d_1+b_0*c_1+a_1*d_0+b_1*c_0+a_2*d_3+b_2*c_3-a_3*d_2-b_3*c_2
        h_2=a_0*c_2-a_1*c_3+a_2*c_0+a_3*c_1
        h1_2=a_0*d_2+b_0*c_2-a_1*d_3-b_1*c_3+a_2*d_0+b_2*c_0+a_3*d_1+b_3*c_1
        h_3=a_0*c_3+a_1*c_2-a_2*c_1+a_3*c_0
        h1_3=a_0*d_3+b_0*c_3+a_1*d_2+b_1*c_2-a_2*d_1-b_2*c_1+a_3*d_0+b_3*c_0

        return  (h_0,h_1,h_2,h_3,h1_0,h1_1,h1_2,h1_3)

    #Normalization of relationship embedding
    def _onorm(self,r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8):
        denominator_0 = r_1 ** 2 + r_2 ** 2 + r_3 ** 2 + r_4 ** 2
        denominator_1 = torch.sqrt(denominator_0)
        deno_cross = r_5 * r_1 + r_6 * r_2 + r_7 * r_3 + r_8 * r_4

        r_5 = r_5 - deno_cross / denominator_0 * r_1
        r_6 = r_6 - deno_cross / denominator_0 * r_2
        r_7 = r_7 - deno_cross / denominator_0 * r_3
        r_8 = r_8 - deno_cross / denominator_0 * r_4

        r_1 = r_1 / denominator_1
        r_2 = r_2 / denominator_1
        r_3 = r_3 / denominator_1
        r_4 = r_4 / denominator_1
        return r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8

    #Calculate the inner product of the head entity and the relationship Hamiltonian product and the tail entity
    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        """Calculating the score of triples.
        
        The formula for calculating the score is :math:` <\boldsymbol{Q}_h \otimes \boldsymbol{W}_r^{\diamond}, \boldsymbol{Q}_t> `

        Args:
            head_emb: The head entity embedding with 8 dimensionalities.
            relation_emb: The relation embedding with 8 dimensionalities.
            tail_emb: The tail entity embedding with 8 dimensionalities.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples with regul_1 and regul_2
        """
        e_1_h,e_2_h,e_3_h,e_4_h,e_5_h,e_6_h,e_7_h,e_8_h = torch.chunk(head_emb, 8, dim=-1)
        e_1_t,e_2_t,e_3_t,e_4_t,e_5_t,e_6_t,e_7_t,e_8_t = torch.chunk(tail_emb, 8, dim=-1)
        r_1,r_2,r_3,r_4,r_5,r_6,r_7,r_8 = torch.chunk(relation_emb, 8, dim=-1)

        r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 = self._onorm(r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 )

        o_1, o_2, o_3, o_4, o_5, o_6, o_7, o_8 = self._omult(e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
                                                             r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8)


        score_r = (o_1 * e_1_t + o_2 * e_2_t + o_3 * e_3_t + o_4 * e_4_t
                   +  o_5 * e_5_t + o_6 * e_6_t + o_7 * e_7_t + o_8 * e_8_t)

        regul_1 = (torch.mean(torch.abs(e_1_h) ** 2)
                 + torch.mean(torch.abs(e_2_h) ** 2)
                 + torch.mean(torch.abs(e_3_h) ** 2)
                 + torch.mean(torch.abs(e_4_h) ** 2)
                 + torch.mean(torch.abs(e_5_h) ** 2)
                 + torch.mean(torch.abs(e_6_h) ** 2)
                 + torch.mean(torch.abs(e_7_h) ** 2)
                 + torch.mean(torch.abs(e_8_h) ** 2)
                 + torch.mean(torch.abs(e_1_t) ** 2)
                 + torch.mean(torch.abs(e_2_t) ** 2)
                 + torch.mean(torch.abs(e_3_t) ** 2)
                 + torch.mean(torch.abs(e_4_t) ** 2)
                 + torch.mean(torch.abs(e_5_t) ** 2)
                 + torch.mean(torch.abs(e_6_t) ** 2)
                 + torch.mean(torch.abs(e_7_t) ** 2)
                 + torch.mean(torch.abs(e_8_t) ** 2)
                 )
        regul_2 = (torch.mean(torch.abs(r_1) ** 2)
                  + torch.mean(torch.abs(r_2) ** 2)
                  + torch.mean(torch.abs(r_3) ** 2)
                  + torch.mean(torch.abs(r_4) ** 2)
                  + torch.mean(torch.abs(r_5) ** 2)
                  + torch.mean(torch.abs(r_6) ** 2)
                  + torch.mean(torch.abs(r_7) ** 2)
                  + torch.mean(torch.abs(r_8) ** 2))

        return (torch.sum(score_r, -1), regul_1, regul_2)

    def forward(self, triples, negs=None, mode='single'):
        if negs != None:
            head_emb, relation_emb, tail_emb = self.tri2emb(negs)
        else:
            head_emb, relation_emb, tail_emb = self.tri2emb(triples)
        score, regul_1, regul_2 = self.score_func(head_emb, relation_emb, tail_emb, mode)
        return (score, regul_1, regul_2)
    
    def get_score(self, batch, mode):
        """The functions used in the testing phase
        Args:
            batch: A batch of data.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """
        triples = batch["positive_sample"]
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
        
        e_1_h,e_2_h,e_3_h,e_4_h,e_5_h,e_6_h,e_7_h,e_8_h = torch.chunk(head_emb, 8, dim=-1)
        e_1_t,e_2_t,e_3_t,e_4_t,e_5_t,e_6_t,e_7_t,e_8_t = torch.chunk(tail_emb, 8, dim=-1)
        r_1,r_2,r_3,r_4,r_5,r_6,r_7,r_8 = torch.chunk(relation_emb, 8, dim=-1)

        r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 = self._onorm(r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 )

        o_1, o_2, o_3, o_4, o_5, o_6, o_7, o_8 = self._omult(e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
                                                             r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8)


        score_r = (o_1 * e_1_t + o_2 * e_2_t + o_3 * e_3_t + o_4 * e_4_t
                   +  o_5 * e_5_t + o_6 * e_6_t + o_7 * e_7_t + o_8 * e_8_t)

        return torch.sum(score_r, -1)




    def quaternion_init(self, in_features, out_features, criterion='he'):
        """
            Quaternion-valued weight initialization
            the initialization scheme is optional on these four datasets, 
            random initialization can get the same performance. This initialization 
            scheme might be useful for the case which needs fewer epochs.
        """
        fan_in = in_features
        fan_out = out_features

        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
        rng = RandomState(2020)

        # Generating randoms and purely imaginary quaternions :
        kernel_shape = (in_features, out_features)

        number_of_weights = np.prod(kernel_shape) # in_features*out_features
        v_i = np.random.uniform(0.0, 1.0, number_of_weights) #(low,high,size)
        v_j = np.random.uniform(0.0, 1.0, number_of_weights)
        v_k = np.random.uniform(0.0, 1.0, number_of_weights)

        # Purely imaginary quaternions unitary
        for i in range(0, number_of_weights):
            norm = np.sqrt(v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2) + 0.0001
            v_i[i] /= norm
            v_j[i] /= norm
            v_k[i] /= norm
        v_i = v_i.reshape(kernel_shape)
        v_j = v_j.reshape(kernel_shape)
        v_k = v_k.reshape(kernel_shape)

        modulus = rng.uniform(low=-s, high=s, size=kernel_shape)


        # Calculate the three parts about t
        kernel_shape1 = (in_features, out_features)
        number_of_weights1 = np.prod(kernel_shape1)
        t_i = np.random.uniform(0.0, 1.0, number_of_weights1)
        t_j = np.random.uniform(0.0, 1.0, number_of_weights1)
        t_k = np.random.uniform(0.0, 1.0, number_of_weights1)

        # Purely imaginary quaternions unitary
        for i in range(0, number_of_weights1):
            norm1 = np.sqrt(t_i[i] ** 2 + t_j[i] ** 2 + t_k[i] ** 2) + 0.0001
            t_i[i] /= norm1
            t_j[i] /= norm1
            t_k[i] /= norm1
        t_i = t_i.reshape(kernel_shape1)
        t_j = t_j.reshape(kernel_shape1)
        t_k = t_k.reshape(kernel_shape1)
        tmp_t = rng.uniform(low=-s, high=s, size=kernel_shape1)


        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
        phase1 = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape1)

        weight_r = modulus * np.cos(phase)
        weight_i = modulus * v_i * np.sin(phase)
        weight_j = modulus * v_j * np.sin(phase)
        weight_k = modulus * v_k * np.sin(phase)

        wt_i = tmp_t * t_i * np.sin(phase1)
        wt_j = tmp_t * t_j * np.sin(phase1)
        wt_k = tmp_t * t_k * np.sin(phase1)

        i_0=weight_r
        i_1=weight_i
        i_2=weight_j
        i_3=weight_k
        i_4=(-wt_i*weight_i-wt_j*weight_j-wt_k*weight_k)/2
        i_5=(wt_i*weight_r+wt_j*weight_k-wt_k*weight_j)/2
        i_6=(-wt_i*weight_k+wt_j*weight_r+wt_k*weight_i)/2
        i_7=(wt_i*weight_j-wt_j*weight_i+wt_k*weight_r)/2


        return (i_0,i_1,i_2,i_3,i_4,i_5,i_6,i_7)
    