import torch
from torch.nn import Module, Linear, LayerNorm, Dropout
from transformers import BertPreTrainedModel, LongformerModel
from transformers.modeling_bert import ACT2FN
from utils import extract_clusters, extract_mentions_to_predicted_clusters_from_clusters, mask_tensor, sim_matrix
from torch.nn.utils.rnn import pad_sequence
from attention import Attention
import torch.nn.functional as F
from consts import PADDING_VALUE, MAX_VALUE, LESS_VALUE, CONTINUE, RETAIN, SHIFT

class FullyConnectedLayer(Module):
    def __init__(self, config, input_dim, output_dim, dropout_prob):
        super(FullyConnectedLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        self.dense = Linear(self.input_dim, self.output_dim)
        self.layer_norm = LayerNorm(self.output_dim, eps=config.layer_norm_eps)
        self.activation_func = ACT2FN[config.hidden_act]
        self.dropout = Dropout(self.dropout_prob)

    def forward(self, inputs):
        temp = inputs
        temp = self.dense(temp)
        temp = self.activation_func(temp)
        temp = self.layer_norm(temp)
        temp = self.dropout(temp)
        return temp


class S2E(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.max_span_length = args.max_span_length
        self.top_lambda = args.top_lambda
        self.ffnn_size = args.ffnn_size
        self.do_mlps = self.ffnn_size > 0
        self.top_m = args.top_m
        self.ffnn_size = self.ffnn_size if self.do_mlps else config.hidden_size
        self.normalise_loss = args.normalise_loss
        self.t_sim = args.t_sim

        self.longformer = LongformerModel(config)

        self.start_mention_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, args.dropout_prob) if self.do_mlps else None
        self.end_mention_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, args.dropout_prob) if self.do_mlps else None
        self.start_coref_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, args.dropout_prob) if self.do_mlps else None
        self.end_coref_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, args.dropout_prob) if self.do_mlps else None

        self.start_ft_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, args.dropout_prob) if self.do_mlps else None
        self.end_ft_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, args.dropout_prob) if self.do_mlps else None

        self.token_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, args.dropout_prob) if self.do_mlps else None

        self.mention_start_classifier = Linear(self.ffnn_size, 1)
        self.mention_end_classifier = Linear(self.ffnn_size, 1)
        self.mention_s2e_classifier = Linear(self.ffnn_size, self.ffnn_size)

        self.antecedent_s2s_classifier = Linear(self.ffnn_size, self.ffnn_size)
        self.antecedent_e2e_classifier = Linear(self.ffnn_size, self.ffnn_size)
        self.antecedent_s2e_classifier = Linear(self.ffnn_size, self.ffnn_size)
        self.antecedent_e2s_classifier = Linear(self.ffnn_size, self.ffnn_size)

        self.sent_s2e_classifier = Linear(self.ffnn_size, self.ffnn_size)

        self.get_sent_attentions = Attention(self.ffnn_size, self.top_m, args.heads, args.dropout_prob)

        self.init_weights()

    def _mask_antecedent_logits(self, antecedent_logits, span_mask):
        # We now build the matrix for each pair of spans (i,j) - whether j is a candidate for being antecedent of i?
        antecedents_mask = torch.ones_like(antecedent_logits, dtype=self.dtype).tril(diagonal=-1)  # [batch_size, k, k]
        antecedents_mask = antecedents_mask * span_mask.unsqueeze(-1)  # [batch_size, k, k]
        antecedent_logits = mask_tensor(antecedent_logits, antecedents_mask)
        return antecedent_logits
        
    def _get_cluster_labels_after_pruning(self, span_starts, span_ends, all_clusters):
        """
        :param span_starts: [batch_size, max_k]
        :param span_ends: [batch_size, max_k]
        :param all_clusters: [batch_size, max_cluster_size, max_clusters_num, 2]
        :return: [batch_size, max_k, max_k + 1] - [b, i, j] == 1 if i is antecedent of j
        """
        batch_size, max_k = span_starts.size()
        new_cluster_labels = torch.zeros((batch_size, max_k, max_k + 1), device='cpu')
        all_clusters_cpu = all_clusters.cpu().numpy()
        for b, (starts, ends, gold_clusters) in enumerate(zip(span_starts.cpu().tolist(), span_ends.cpu().tolist(), all_clusters_cpu)):
            gold_clusters = extract_clusters(gold_clusters)
            mention_to_gold_clusters = extract_mentions_to_predicted_clusters_from_clusters(gold_clusters)
            gold_mentions = set(mention_to_gold_clusters.keys())
            for i, (start, end) in enumerate(zip(starts, ends)):
                if (start, end) not in gold_mentions:
                    continue
                for j, (a_start, a_end) in enumerate(list(zip(starts, ends))[:i]):
                    if (a_start, a_end) in mention_to_gold_clusters[(start, end)]:
                        new_cluster_labels[b, i, j] = 1
        new_cluster_labels = new_cluster_labels.to(self.device)
        no_antecedents = 1 - torch.sum(new_cluster_labels, dim=-1).bool().float()
        new_cluster_labels[:, :, -1] = no_antecedents
        return new_cluster_labels
        
    def _get_marginal_log_likelihood_loss(self, coref_logits, cluster_labels_after_pruning, span_mask):
        """
        :param coref_logits: [batch_size, max_k, max_k]
        :param cluster_labels_after_pruning: [batch_size, max_k, max_k]
        :param span_mask: [batch_size, max_k]
        :return:
        """
        gold_coref_logits = mask_tensor(coref_logits, cluster_labels_after_pruning)

        gold_log_sum_exp = torch.logsumexp(gold_coref_logits, dim=-1)  # [batch_size, max_k]
        all_log_sum_exp = torch.logsumexp(coref_logits, dim=-1)  # [batch_size, max_k]

        gold_log_probs = gold_log_sum_exp - all_log_sum_exp
        losses = - gold_log_probs
        losses = losses * span_mask
        per_example_loss = torch.sum(losses, dim=-1)  # [batch_size]
        if self.normalise_loss:
            per_example_loss = per_example_loss / losses.size(-1)
        loss = per_example_loss.mean()
        return loss

    def _calc_coref_logits(self, top_k_start_coref_reps, top_k_end_coref_reps):
        # s2s
        temp = self.antecedent_s2s_classifier(top_k_start_coref_reps)  # [batch_size, max_k, dim]
        top_k_s2s_coref_logits = torch.matmul(temp,
                                            top_k_start_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # e2e
        temp = self.antecedent_e2e_classifier(top_k_end_coref_reps)  # [batch_size, max_k, dim]
        top_k_e2e_coref_logits = torch.matmul(temp,
                                            top_k_end_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # s2e
        temp = self.antecedent_s2e_classifier(top_k_start_coref_reps)  # [batch_size, max_k, dim]
        top_k_s2e_coref_logits = torch.matmul(temp,
                                            top_k_end_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # e2s
        temp = self.antecedent_e2s_classifier(top_k_end_coref_reps)  # [batch_size, max_k, dim]
        top_k_e2s_coref_logits = torch.matmul(temp,
                                            top_k_start_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # sum all terms
        coref_logits = top_k_s2e_coref_logits + top_k_e2s_coref_logits + top_k_s2s_coref_logits + top_k_e2e_coref_logits  # [batch_size, max_k, max_k]
        return coref_logits

    def _get_adjacency_matrix(self, cp_reps, cb_reps, cp_mask, cb_mask):
        
        cb_reps, cp_reps = cb_reps.squeeze(-2), cp_reps.squeeze(-2)    # [batch_size,sent_len,dim]
        sim_cb_cp, sim_cb_cb = sim_matrix(cb_reps, cp_reps), sim_matrix(cb_reps, cb_reps) # [batch_size, sent_len, sent_len]
        
        cb_cb = torch.where(sim_cb_cb>=self.t_sim, CONTINUE, SHIFT)
        try:
            cb_cb = (cb_cb * cb_mask.unsqueeze(-2)) * cb_mask.unsqueeze(-1).int()
        except IndexError:
            print('cb_mask:',cb_mask)
        
        cb_cp = torch.where(sim_cb_cp>=self.t_sim, CONTINUE, RETAIN) * cp_mask.unsqueeze(-2)
        
        adjacency_matrix = torch.where((cb_cb==CONTINUE) & (cb_cp==RETAIN), cb_cp, cb_cb)  # [batch_size, sent_len, sent_len]
        
        bs, num_sents, _ = adjacency_matrix.shape
        identity = torch.eye(num_sents).reshape((1, num_sents, num_sents))
        identity = identity.repeat(bs, 1, 1)
        
        adjacency_matrix = adjacency_matrix.float().to(self.device) + identity.float().to(self.device)    
        adjacency_matrix[adjacency_matrix == 0] = float('-inf')
        
        adjacency_matrix = F.softmax(adjacency_matrix, dim=-1)
        
        return adjacency_matrix
    
    def _get_cbs(self, sent_topm_reps, sent_topm_mask, topk_sent_reps):
        
        sent_reps_from_sec = topk_sent_reps[:,1:,:].unsqueeze(-2)   # [batch_size,sent_len-1,1,dim]
        sent_topm_reps_to_last = sent_topm_reps[:,:-1,:,:]   # [batch_size,sent_len-1,topm,dim]
        sent_topm_mask_to_last = (sent_topm_mask[:,:-1,:].float()-1)*MAX_VALUE  # [batch_size,sent_len-1, topm,dim]
        bz,_,_,dim = sent_topm_reps_to_last.size()
        
        sim = F.cosine_similarity(sent_topm_reps_to_last, sent_reps_from_sec, -1) # [batch_size,sent_len-1,topm]
        sim = sim + sent_topm_mask_to_last # [batch_size,sent_len-1,topm]
        _, cbs = torch.topk(sim,dim=-1, k=1)  # [batch_size, sent_len-1, 1]
        
        cb_mask = torch.gather(sent_topm_mask[:,:-1,:], -1, cbs)     # [batch_size, sent_len-1, 1]
        cb_mask = torch.cat((torch.zeros((bz,1,1), device=self.device).int(), cb_mask.int()), 1).squeeze()  # [batch_size, sent_len]
        
        cb_reps = torch.gather(sent_topm_reps_to_last, 2, cbs.unsqueeze(-1).expand(-1,-1,-1,dim))    # [batch_size,sent_len-1,1,dim]
        cb_reps = torch.cat((torch.zeros((bz,1,1,dim), device=self.device),cb_reps),1) # [batch_size,sent_len,1,dim]
        
        return cb_reps, cb_mask
    
    def _get_sent_sent_logits_byGraph(self, sent_topm_reps, sent_topm_mask, topk_start_sent_reps, topk_end_sent_reps, topk_sent_reps_indices, topk_sent_reps):
        
        bz,topk,dim = topk_start_sent_reps.size()
        
        sent_attentions = self.get_sent_attentions(sent_topm_reps, sent_topm_mask) # [batch_size, sent_len, topm]
        
        #
        _, cf = torch.sort(sent_attentions, -1, True)   # [batch_size, sent_len, topm]
        cf_mask = torch.gather(sent_topm_mask, -1, cf)     # [batch_size, sent_len, topm]
        cp_reps = torch.gather(sent_topm_reps, 2, (cf[:,:,0].unsqueeze(-1)).unsqueeze(-1).expand(-1,-1,-1,dim)) # [batch_size,sent_len,1,dim]
        
        cb_reps, cb_mask = self._get_cbs(sent_topm_reps, sent_topm_mask, topk_sent_reps)    # [batch_size,sent_len,1,dim]
        
        adjacency_matrix = self._get_adjacency_matrix(cp_reps, cb_reps, cf_mask[:,:,0], cb_mask) # [batch_size,sent_len, sent_len]
        row = topk_sent_reps_indices.unsqueeze(-1)
        col = topk_sent_reps_indices.unsqueeze(-2)
        topk_adjacency_matrix = adjacency_matrix[(torch.arange(bz).unsqueeze(-1)).unsqueeze(-1).expand(bz, topk, topk),row,col] # [batch_size,max_k,max_k]
        
        temp = self.sent_s2e_classifier(topk_start_sent_reps)
        temp1 = torch.matmul(topk_adjacency_matrix, temp)
        topk_sent_sent_logits = torch.matmul(temp1, topk_end_sent_reps.permute([0, 2, 1]))   # [batch_size, max_k, max_k]
        
        return topk_sent_sent_logits
    
    def _get_sent_topm_reps(self,sent_topm_sentindices, mention_start_ids, mention_end_ids, start_ft_reps, end_ft_reps):
     
        dim, topk, sent_len = start_ft_reps.size(2), mention_start_ids.size(1), sent_topm_sentindices.size(1)
        
        sent_topm_sentindices = torch.where(sent_topm_sentindices <PADDING_VALUE, sent_topm_sentindices, torch.tensor(topk-1, device=self.device)) # [batch_size, sent_len, topm]
        
        sent_topm_start_ids = torch.gather(mention_start_ids.unsqueeze(1).expand(-1,sent_len,-1), 2, sent_topm_sentindices)
        sent_topm_end_ids = torch.gather(mention_end_ids.unsqueeze(1).expand(-1,sent_len,-1), 2, sent_topm_sentindices)

        sent_topm_start_reps = torch.gather(start_ft_reps.unsqueeze(1).expand(-1,sent_len,-1, -1), 2, sent_topm_start_ids.unsqueeze(-1).expand(-1,-1,-1,dim))
        sent_topm_end_reps = torch.gather(end_ft_reps.unsqueeze(1).expand(-1,sent_len,-1, -1), 2, sent_topm_end_ids.unsqueeze(-1).expand(-1,-1,-1,dim))
        
        sent_topm_reps = sent_topm_start_reps + sent_topm_end_reps # [batch_size,sent_len,topm,dim]

        return sent_topm_reps

    def _get_topm_sentindices(self, topk_mention_logits, topk_sent_indices, topk_sent_reps_indices):
        
        bz,topk = topk_sent_reps_indices.shape

        topk_sentindices = torch.arange(topk).repeat(bz,1) # [batch_size, max_k]
        
        topk_sent_sentindices_list = [[topk_sentindices[j][topk_sent_reps_indices[j] == i] for i in range(len(topk_sent_indices))] for j in range(bz)]
        sent_topm_indices_list = [[torch.sort(torch.sort(topk_mention_logits[j][topk_sent_reps_indices[j] == i], descending=True).indices[:self.top_m]).values for i in range(len(topk_sent_indices))] for j in range(bz)]
	
        sent_topm_sentindices = torch.stack([pad_sequence([torch.gather(topk_sent_sentindices_list[j][i].to(self.device).clone(), 0, sent_topm_indices_list[j][i].to(self.device).clone()) for i in range(len(topk_sent_indices))], batch_first=True, padding_value=PADDING_VALUE) for j in range(bz)]) # [batch_size,sent_len,topm]

        return sent_topm_sentindices # [batch_size,sent_len,topm]
    
    def _calc_sent_sent_logits(self, mention_start_ids, mention_end_ids, topk_mention_logits, start_ft_reps, end_ft_reps, sent_reps, inverse_indices):
        
        topk_start_sent_reps, topk_end_sent_reps, topk_sent_reps, topk_sent_indices, topk_sent_reps_indices = self._get_topk_sent_reps(mention_start_ids, mention_end_ids, sent_reps, inverse_indices)
        
        # get index of topm of each sentence in topk indices form
        sent_topm_sentindices = self._get_topm_sentindices(topk_mention_logits, topk_sent_indices, topk_sent_reps_indices) # [batch_size,sent_len,topm]
        
        # 
        sent_topm_mask = sent_topm_sentindices.le(LESS_VALUE)  # [batch_size,sent_len, topm]
        sent_topm_reps = self._get_sent_topm_reps(sent_topm_sentindices, mention_start_ids, mention_end_ids, start_ft_reps, end_ft_reps) # [batch_size,sent_len, topm, dim]
        
        topk_sent_sent_logits = self._get_sent_sent_logits_byGraph(sent_topm_reps, sent_topm_mask, topk_start_sent_reps, topk_end_sent_reps, topk_sent_reps_indices, topk_sent_reps)
        
        return topk_sent_sent_logits

    def _get_topk_sent_reps(self, mention_start_ids, mention_end_ids, sent_reps, inverse_indices):
        
        topk_start_inverse_indices = torch.gather(inverse_indices, 1, mention_start_ids) # [batch_size, max_k]
        topk_end_inverse_indices = torch.gather(inverse_indices, 1, mention_end_ids) # [batch_size, max_k]
        
        bz,_,dim = sent_reps.size()
        topk_start_sent_reps = torch.gather(sent_reps, 1, topk_start_inverse_indices.unsqueeze(-1).expand(-1,-1,dim)) # [batch_size, max_k, dim]
        topk_end_sent_reps = torch.gather(sent_reps, 1, topk_end_inverse_indices.unsqueeze(-1).expand(-1,-1,dim)) # [batch_size, max_k, dim]
        
        #
        topk_sent_indices, topk_sent_reps_indices = torch.unique(topk_start_inverse_indices, sorted=True, return_inverse=True) # [sent_len], [batch_size,max_k]
        topk_sent_reps = torch.gather(sent_reps, 1, (topk_sent_indices.unsqueeze(0)).unsqueeze(-1).expand(bz,-1,dim)) # [batch_size, max_k, dim]
        
        return topk_start_sent_reps, topk_end_sent_reps, topk_sent_reps, topk_sent_indices, topk_sent_reps_indices
        
    def _get_span_mask(self, batch_size, k, max_k):
        """
        :param batch_size: int
        :param k: tensor of size [batch_size], with the required k for each example
        :param max_k: int
        :return: [batch_size, max_k] of zero-ones, where 1 stands for a valid span and 0 for a padded span
        """
        size = (batch_size, max_k)
        idx = torch.arange(max_k, device=self.device).unsqueeze(0).expand(size)
        len_expanded = k.unsqueeze(1).expand(size)
        return (idx < len_expanded).int()
    
    def _prune_topk_mentions(self, mention_logits, attention_mask, sentence_ids):
        """
        :param mention_logits: Shape [batch_size, seq_len, seq_len]
        :param attention_mask: [batch_size, seq_len]
        :param top_lambda:
        :return:
        """
        batch_size, seq_length, _ = mention_logits.size()
        actual_seq_lengths = torch.sum(attention_mask, dim=-1)  # [batch_size]

        k = (actual_seq_lengths * self.top_lambda).int()  # [batch_size]
        max_k = int(torch.max(k))  # This is the k for the largest input in the batch, we will need to pad
        
        _, topk_1d_indices = torch.topk(mention_logits.view(batch_size, -1), dim=-1, k=max_k)  # [batch_size, max_k]
        span_mask = self._get_span_mask(batch_size, k, max_k)  # [batch_size, max_k]
        topk_1d_indices = (topk_1d_indices * span_mask) + (1 - span_mask) * ((seq_length ** 2) - 1)  # We take different k for each example
        sorted_topk_1d_indices, _ = torch.sort(topk_1d_indices, dim=-1)  # [batch_size, max_k]
        
        topk_mention_start_ids = sorted_topk_1d_indices // seq_length  # [batch_size, max_k]
        topk_mention_end_ids = sorted_topk_1d_indices % seq_length  # [batch_size, max_k]

        topk_mention_logits = mention_logits[torch.arange(batch_size).unsqueeze(-1).expand(batch_size, max_k),
                                            topk_mention_start_ids, topk_mention_end_ids]  # [batch_size, max_k]

        topk_inter_mention_logits = topk_mention_logits.unsqueeze(-1) + topk_mention_logits.unsqueeze(-2)  # [batch_size, max_k, max_k]

        return topk_mention_start_ids, topk_mention_end_ids, span_mask, topk_inter_mention_logits, topk_mention_logits

    def _calc_sent_reps(self, token_reps, sentence_ids):
        
        bz = token_reps.size(0)
        
        unique_sent_ids, inverse_indices = torch.unique(sentence_ids, sorted=True, return_inverse=True)
        sent_reps = torch.stack([torch.stack([torch.mean(token_reps[j][sentence_ids[j] == i][:], dim=0) for i in unique_sent_ids]) for j in range(bz)]) # [batch_size,sent_len,dim]
        
        return sent_reps, inverse_indices

    def _get_mention_mask(self, mention_logits, sentence_ids):
        """
        Returns a tensor of size [batch_size, seq_len, seq_len] where valid spans
        (start <= end < start + max_span_length) are 1 and the rest are 0
        :param mention_logits: Either the span mention logits, size [batch_size, seq_len, seq_len]
        """
        mention_mask = torch.ones_like(mention_logits, dtype=self.dtype)
        mention_mask = mention_mask.triu(diagonal=0)
        mention_mask = mention_mask.tril(diagonal=self.max_span_length - 1).to(torch.bool)

        start_sent_idx = torch.unsqueeze(sentence_ids, 2)   # [batch_size, seq_len, 1]
        end_sent_idx = torch.unsqueeze(sentence_ids, 1)     # [batch_size, 1, seq_len]
        sentence_mask = (start_sent_idx == end_sent_idx)   # [batch_size, seq_len, seq_len]

        mention_mask = (mention_mask & sentence_mask).to(torch.float)

        return mention_mask

    def _calc_mention_logits(self, start_mention_reps, end_mention_reps, sentence_ids):
        start_mention_logits = self.mention_start_classifier(start_mention_reps).squeeze(-1)  # [batch_size, seq_len]
        end_mention_logits = self.mention_end_classifier(end_mention_reps).squeeze(-1)  # [batch_size, seq_len]

        temp = self.mention_s2e_classifier(start_mention_reps)  # [batch_size, seq_len, dim]
        joint_mention_logits = torch.matmul(temp,
                                            end_mention_reps.permute([0, 2, 1]))  # [batch_size, seq_len, seq_len]

        mention_logits = joint_mention_logits + start_mention_logits.unsqueeze(-1) + end_mention_logits.unsqueeze(-2)

        mention_mask = self._get_mention_mask(mention_logits, sentence_ids)  # [batch_size, seq_len, seq_len]
        mention_logits = mask_tensor(mention_logits, mention_mask)  # [batch_size, seq_len, seq_len]
         
        return mention_logits

    def forward(self, input_ids, attention_mask=None, sentence_ids=None, gold_clusters=None, return_all_outputs=False):
        outputs = self.longformer(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # [batch_size, seq_len, dim]

        # Compute representations
        start_mention_reps = self.start_mention_mlp(sequence_output) if self.do_mlps else sequence_output
        end_mention_reps = self.end_mention_mlp(sequence_output) if self.do_mlps else sequence_output
        # mention scores        
        mention_logits = self._calc_mention_logits(start_mention_reps, end_mention_reps, sentence_ids)
        
        # Compute sentence representations
        token_reps = self.token_mlp(sequence_output) if self.do_mlps else sequence_output   # [batch_size, seq_len, dim]
        sent_reps, inverse_indices = self._calc_sent_reps(token_reps, sentence_ids)     # [batch_size,sent_len,dim], [batch_size,seq_len]

        # prune mentions
        mention_start_ids, mention_end_ids, span_mask, topk_inter_mention_logits, topk_mention_logits = self._prune_topk_mentions(mention_logits, attention_mask, sentence_ids)
        
        # Compute representations for focus tokens
        start_ft_reps = self.start_ft_mlp(sequence_output) if self.do_mlps else sequence_output
        end_ft_reps = self.end_ft_mlp(sequence_output) if self.do_mlps else sequence_output
        # sentence-sentence scores
        sent_sent_logits = self._calc_sent_sent_logits(mention_start_ids, mention_end_ids, topk_mention_logits, start_ft_reps, end_ft_reps, sent_reps, inverse_indices)
        
        # Compute representations
        start_coref_reps = self.start_coref_mlp(sequence_output) if self.do_mlps else sequence_output
        end_coref_reps = self.end_coref_mlp(sequence_output) if self.do_mlps else sequence_output
        
        batch_size, _, dim = start_coref_reps.size()
        max_k = mention_start_ids.size(-1)
        size = (batch_size, max_k, dim)
        
        # top k representations
        topk_start_coref_reps = torch.gather(start_coref_reps, dim=1, index=mention_start_ids.unsqueeze(-1).expand(size))
        topk_end_coref_reps = torch.gather(end_coref_reps, dim=1, index=mention_end_ids.unsqueeze(-1).expand(size))
        # coref scores
        coref_logits = self._calc_coref_logits(topk_start_coref_reps, topk_end_coref_reps)
        
        # 
        final_logits = topk_inter_mention_logits + coref_logits + sent_sent_logits
        final_logits = self._mask_antecedent_logits(final_logits, span_mask)
        # adding zero logits for null span
        final_logits = torch.cat((final_logits, torch.zeros((batch_size, max_k, 1), device=self.device)), dim=-1)  # [batch_size, max_k, max_k + 1]

        if return_all_outputs:
            outputs = (mention_start_ids, mention_end_ids, final_logits, mention_logits)
        else:
            outputs = tuple()

        if gold_clusters is not None:
            losses = {}
            labels_after_pruning = self._get_cluster_labels_after_pruning(mention_start_ids, mention_end_ids, gold_clusters)
            loss = self._get_marginal_log_likelihood_loss(final_logits, labels_after_pruning, span_mask)
            losses.update({"loss": loss})
            outputs = (loss,) + outputs + (losses,)
            
        return outputs
