import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import entmax
import numpy as np
import itertools


def clamp(value, abs_max):
  value = max(-abs_max, value)
  value = min(abs_max, value)
  return value

# Adapted from
# https://github.com/tensorflow/tensor2tensor/blob/0b156ac533ab53f65f44966381f6e147c7371eee/tensor2tensor/layers/common_attention.py
def relative_attention_logits(query, key, relation):
  # We can't reuse the same logic as tensor2tensor because we don't share relation vectors across the batch.
  # In this version, relation vectors are shared across heads.
  # query: [batch, heads, num queries, depth].
  # key: [batch, heads, num kvs, depth].
  # relation: [batch, num queries, num kvs, depth].

  # qk_matmul is [batch, heads, num queries, num kvs]
  qk_matmul = torch.matmul(query, key.transpose(-2, -1))

  # q_t is [batch, num queries, heads, depth]
  q_t = query.permute(0, 2, 1, 3)

  # r_t is [batch, num queries, depth, num kvs]
  r_t = relation.transpose(-2, -1)

  #   [batch, num queries, heads, depth]
  # * [batch, num queries, depth, num kvs]
  # = [batch, num queries, heads, num kvs]
  # For each batch and query, we have a query vector per head.
  # We take its dot product with the relation vector for each kv.
  q_tr_t_matmul = torch.matmul(q_t, r_t)

  # qtr_t_matmul_t is [batch, heads, num queries, num kvs]
  q_tr_tmatmul_t = q_tr_t_matmul.permute(0, 2, 1, 3)

  # [batch, heads, num queries, num kvs]
  return (qk_matmul + q_tr_tmatmul_t) / math.sqrt(query.shape[-1])

  # Sharing relation vectors across batch and heads:
  # query: [batch, heads, num queries, depth].
  # key: [batch, heads, num kvs, depth].
  # relation: [num queries, num kvs, depth].
  #
  # Then take
  # key reshaped
  #   [num queries, batch * heads, depth]
  # relation.transpose(-2, -1)
  #   [num queries, depth, num kvs]
  # and multiply them together.
  #
  # Without sharing relation vectors across heads:
  # query: [batch, heads, num queries, depth].
  # key: [batch, heads, num kvs, depth].
  # relation: [batch, heads, num queries, num kvs, depth].
  #
  # Then take
  # key.unsqueeze(3)
  #   [batch, heads, num queries, 1, depth]
  # relation.transpose(-2, -1)
  #   [batch, heads, num queries, depth, num kvs]
  # and multiply them together:
  #   [batch, heads, num queries, 1, depth]
  # * [batch, heads, num queries, depth, num kvs]
  # = [batch, heads, num queries, 1, num kvs]
  # and squeeze
  # [batch, heads, num queries, num kvs]


def relative_attention_values(weight, value, relation):
  # In this version, relation vectors are shared across heads.
  # weight: [batch, heads, num queries, num kvs].
  # value: [batch, heads, num kvs, depth].
  # relation: [batch, num queries, num kvs, depth].

  # wv_matmul is [batch, heads, num queries, depth]
  wv_matmul = torch.matmul(weight, value)

  # w_t is [batch, num queries, heads, num kvs]
  w_t = weight.permute(0, 2, 1, 3)

  #   [batch, num queries, heads, num kvs]
  # * [batch, num queries, num kvs, depth]
  # = [batch, num queries, heads, depth]
  w_tr_matmul = torch.matmul(w_t, relation)

  # w_tr_matmul_t is [batch, heads, num queries, depth]
  w_tr_matmul_t = w_tr_matmul.permute(0, 2, 1, 3)

  return wv_matmul + w_tr_matmul_t


# Adapted from The Annotated Transformer
def clones(module_fn, N):
  return nn.ModuleList([module_fn() for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
  "Compute 'Scaled Dot Product Attention'"
  d_k = query.size(-1)
  scores = torch.matmul(query, key.transpose(-2, -1)) \
           / math.sqrt(d_k)
  if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)
  p_attn = F.softmax(scores, dim=-1)
  if dropout is not None:
    p_attn = dropout(p_attn)
  # return torch.matmul(p_attn, value), scores.squeeze(1).squeeze(1)
  return torch.matmul(p_attn, value), p_attn


def sparse_attention(query, key, value, alpha, mask=None, dropout=None):
  "Compute 'Scaled Dot Product Attention'"
  d_k = query.size(-1)
  scores = torch.matmul(query, key.transpose(-2, -1)) \
           / math.sqrt(d_k)
  if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)
  if alpha == 2:
    p_attn = entmax.sparsemax(scores, -1)
  elif alpha == 1.5:
    p_attn = entmax.entmax15(scores, -1)
  else:
    raise NotImplementedError
  if dropout is not None:
    p_attn = dropout(p_attn)
  # return torch.matmul(p_attn, value), scores.squeeze(1).squeeze(1)
  return torch.matmul(p_attn, value), p_attn


# Adapted from The Annotated Transformers
class MultiHeadedAttention(nn.Module):
  def __init__(self, h, d_model, dropout=0.1):
    "Take in model size and number of heads."
    super(MultiHeadedAttention, self).__init__()
    assert d_model % h == 0
    # We assume d_v always equals d_k
    self.d_k = d_model // h
    self.h = h
    self.linears = clones(lambda: nn.Linear(d_model, d_model), 4)
    self.attn = None
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, query, key, value, mask=None):
    "Implements Figure 2"
    if mask is not None:
      # Same mask applied to all h heads.
      mask = mask.unsqueeze(1)
    nbatches = query.size(0)

    # 1) Do all the linear projections in batch from d_model => h x d_k
    query, key, value = \
      [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
       for l, x in zip(self.linears, (query, key, value))]

    # 2) Apply attention on all the projected vectors in batch.
    x, self.attn = attention(query, key, value, mask=mask,
                             dropout=self.dropout)

    # 3) "Concat" using a view and apply a final linear.
    x = x.transpose(1, 2).contiguous() \
      .view(nbatches, -1, self.h * self.d_k)
    if query.dim() == 3:
      x = x.squeeze(1)
    return self.linears[-1](x)


# Adapted from The Annotated Transformer
def attention_with_relations(query, key, value, relation_k, relation_v, mask=None, dropout=None):
  "Compute 'Scaled Dot Product Attention'"
  d_k = query.size(-1)
  scores = relative_attention_logits(query, key, relation_k)
  if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)
  p_attn_orig = F.softmax(scores, dim=-1)
  if dropout is not None:
    p_attn = dropout(p_attn_orig)
  return relative_attention_values(p_attn, value, relation_v), p_attn_orig


class PointerWithRelations(nn.Module):
  def __init__(self, hidden_size, num_relation_kinds, dropout=0.2):
    super(PointerWithRelations, self).__init__()
    self.hidden_size = hidden_size
    self.linears = clones(lambda: nn.Linear(hidden_size, hidden_size), 3)
    self.attn = None
    self.dropout = nn.Dropout(p=dropout)

    self.relation_k_emb = nn.Embedding(num_relation_kinds, self.hidden_size)
    self.relation_v_emb = nn.Embedding(num_relation_kinds, self.hidden_size)

  def forward(self, query, key, value, relation, mask=None):
    relation_k = self.relation_k_emb(relation)
    relation_v = self.relation_v_emb(relation)

    if mask is not None:
      mask = mask.unsqueeze(0)
    nbatches = query.size(0)

    query, key, value = \
      [l(x).view(nbatches, -1, 1, self.hidden_size).transpose(1, 2)
       for l, x in zip(self.linears, (query, key, value))]

    _, self.attn = attention_with_relations(
      query,
      key,
      value,
      relation_k,
      relation_v,
      mask=mask,
      dropout=self.dropout)

    return self.attn[0, 0]


# Adapted from The Annotated Transformer
class MultiHeadedAttentionWithRelations(nn.Module):
  def __init__(self, h, d_model, dropout=0.1):
    "Take in model size and number of heads."
    super(MultiHeadedAttentionWithRelations, self).__init__()
    assert d_model % h == 0
    # We assume d_v always equals d_k
    self.d_k = d_model // h
    self.h = h
    self.linears = clones(lambda: nn.Linear(d_model, d_model), 4)
    self.attn = None
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, query, key, value, relation_k, relation_v, mask=None):
    # query shape: [batch, num queries, d_model]
    # key shape: [batch, num kv, d_model]
    # value shape: [batch, num kv, d_model]
    # relations_k shape: [batch, num queries, num kv, (d_model // h)]
    # relations_v shape: [batch, num queries, num kv, (d_model // h)]
    # mask shape: [batch, num queries, num kv]
    if mask is not None:
      # Same mask applied to all h heads.
      # mask shape: [batch, 1, num queries, num kv]
      mask = mask.unsqueeze(1)
    nbatches = query.size(0)

    # 1) Do all the linear projections in batch from d_model => h x d_k
    query, key, value = \
      [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
       for l, x in zip(self.linears, (query, key, value))]

    # 2) Apply attention on all the projected vectors in batch.
    # x shape: [batch, heads, num queries, depth]
    x, self.attn = attention_with_relations(
      query,
      key,
      value,
      relation_k,
      relation_v,
      mask=mask,
      dropout=self.dropout)

    # 3) "Concat" using a view and apply a final linear.
    x = x.transpose(1, 2).contiguous() \
      .view(nbatches, -1, self.h * self.d_k)
    return self.linears[-1](x)


# Adapted from The Annotated Transformer
class Encoder(nn.Module):
  "Core encoder is a stack of N layers"

  def __init__(self, layer, layer_size, N, tie_layers=False):
    super(Encoder, self).__init__()
    if tie_layers:
      self.layer = layer()
      self.layers = [self.layer for _ in range(N)]
    else:
      self.layers = clones(layer, N)
    self.norm = nn.LayerNorm(layer_size)

    # TODO initialize using xavier

  def forward(self, x, relation, mask):
    "Pass the input (and mask) through each layer in turn."
    for layer in self.layers:
      x = layer(x, relation, mask)
    return self.norm(x)


# Adapted from The Annotated Transformer
class SublayerConnection(nn.Module):
  """
  A residual connection followed by a layer norm.
  Note for code simplicity the norm is first as opposed to last.
  """

  def __init__(self, size, dropout):
    super(SublayerConnection, self).__init__()
    self.norm = nn.LayerNorm(size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, sublayer):
    "Apply residual connection to any sublayer with the same size."
    return x + self.dropout(sublayer(self.norm(x)))


# Adapted from The Annotated Transformer
class EncoderLayer(nn.Module):
  "Encoder is made up of self-attn and feed forward (defined below)"

  def __init__(self, size, self_attn, feed_forward, num_relation_kinds, dropout):
    super(EncoderLayer, self).__init__()
    self.self_attn = self_attn
    self.feed_forward = feed_forward
    self.sublayer = clones(lambda: SublayerConnection(size, dropout), 2)
    self.size = size

    self.relation_k_emb = nn.Embedding(num_relation_kinds, self.self_attn.d_k)
    self.relation_v_emb = nn.Embedding(num_relation_kinds, self.self_attn.d_k)

  def forward(self, x, relation, mask):
    "Follow Figure 1 (left) for connections."
    relation_k = self.relation_k_emb(relation)
    relation_v = self.relation_v_emb(relation)

    x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, relation_k, relation_v, mask))
    return self.sublayer[1](x, self.feed_forward)


# Adapted from The Annotated Transformer
class PositionwiseFeedForward(nn.Module):
  "Implements FFN equation."

  def __init__(self, d_model, d_ff, dropout=0.1):
    super(PositionwiseFeedForward, self).__init__()
    self.w_1 = nn.Linear(d_model, d_ff)
    self.w_2 = nn.Linear(d_ff, d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    return self.w_2(self.dropout(F.relu(self.w_1(x))))

relation_ids = {
  ('qq_dist', -2): 0, ('qq_dist', -1): 1, ('qq_dist', 0): 2,
  ('qq_dist', 1): 3, ('qq_dist', 2): 4, 'qc_default': 5,
  'qt_default': 6, 'cq_default': 7, 'cc_default': 8,
  'cc_foreign_key_forward': 9, 'cc_foreign_key_backward': 10,
  'cc_table_match': 11, ('cc_dist', -2): 12, ('cc_dist', -1): 13,
  ('cc_dist', 0): 14, ('cc_dist', 1): 15, ('cc_dist', 2): 16,
  'ct_default': 17, 'ct_foreign_key': 18, 'ct_primary_key': 19,
  'ct_table_match': 20, 'ct_any_table': 21, 'tq_default': 22,
  'tc_default': 23, 'tc_primary_key': 24, 'tc_table_match': 25,
  'tc_any_table': 26, 'tc_foreign_key': 27, 'tt_default': 28,
  'tt_foreign_key_forward': 29, 'tt_foreign_key_backward': 30,
  'tt_foreign_key_both': 31, ('tt_dist', -2): 32, ('tt_dist', -1): 33,
  ('tt_dist', 0): 34, ('tt_dist', 1): 35, ('tt_dist', 2): 36, 'qcCEM': 37,
  'cqCEM': 38, 'qtTEM': 39, 'tqTEM': 40, 'qcCPM': 41, 'cqCPM': 42,
  'qtTPM': 43, 'tqTPM': 44, 'qcNUMBER': 45, 'cqNUMBER': 46, 'qcTIME': 47,
  'cqTIME': 48, 'qcCELLMATCH': 49, 'cqCELLMATCH': 50}

num_heads = 8
hidden_size = 1024
ff_size = 4096
dropout = 0.1
num_layers = 8
tie_layers = False
encoder = Encoder(
  lambda: EncoderLayer(
    hidden_size,
    MultiHeadedAttentionWithRelations(
            num_heads,
            hidden_size,
            dropout),
    PositionwiseFeedForward(
            hidden_size,
            ff_size,
            dropout),
        len(relation_ids),
        dropout),
    hidden_size,
    num_layers,
    tie_layers)


class RelationalTransformerUpdate(torch.nn.Module):

  def __init__(self, num_layers, num_heads, hidden_size,
               ff_size=None,
               dropout=0.1,
               merge_types=False,
               tie_layers=False,
               qq_max_dist=2,
               # qc_token_match=True,
               # qt_token_match=True,
               # cq_token_match=True,
               cc_foreign_key=True,
               cc_table_match=True,
               cc_max_dist=2,
               ct_foreign_key=True,
               ct_table_match=True,
               # tq_token_match=True,
               tc_table_match=True,
               tc_foreign_key=True,
               tt_max_dist=2,
               tt_foreign_key=True,
               sc_link=False,
               cv_link=False,
               ):
    super().__init__()
    self.num_heads = num_heads

    self.qq_max_dist = qq_max_dist
    # self.qc_token_match = qc_token_match
    # self.qt_token_match = qt_token_match
    # self.cq_token_match = cq_token_match
    self.cc_foreign_key = cc_foreign_key
    self.cc_table_match = cc_table_match
    self.cc_max_dist = cc_max_dist
    self.ct_foreign_key = ct_foreign_key
    self.ct_table_match = ct_table_match
    # self.tq_token_match = tq_token_match
    self.tc_table_match = tc_table_match
    self.tc_foreign_key = tc_foreign_key
    self.tt_max_dist = tt_max_dist
    self.tt_foreign_key = tt_foreign_key

    self.relation_ids = {}

    def add_relation(name):
      self.relation_ids[name] = len(self.relation_ids)

    def add_rel_dist(name, max_dist):
      for i in range(-max_dist, max_dist + 1):
        add_relation((name, i))

    add_rel_dist('qq_dist', qq_max_dist)

    add_relation('qc_default')
    # if qc_token_match:
    #    add_relation('qc_token_match')

    add_relation('qt_default')
    # if qt_token_match:
    #    add_relation('qt_token_match')

    add_relation('cq_default')
    # if cq_token_match:
    #    add_relation('cq_token_match')

    add_relation('cc_default')
    if cc_foreign_key:
      add_relation('cc_foreign_key_forward')
      add_relation('cc_foreign_key_backward')
    if cc_table_match:
      add_relation('cc_table_match')
    add_rel_dist('cc_dist', cc_max_dist)

    add_relation('ct_default')
    if ct_foreign_key:
      add_relation('ct_foreign_key')
    if ct_table_match:
      add_relation('ct_primary_key')
      add_relation('ct_table_match')
      add_relation('ct_any_table')

    add_relation('tq_default')
    # if cq_token_match:
    #    add_relation('tq_token_match')

    add_relation('tc_default')
    if tc_table_match:
      add_relation('tc_primary_key')
      add_relation('tc_table_match')
      add_relation('tc_any_table')
    if tc_foreign_key:
      add_relation('tc_foreign_key')

    add_relation('tt_default')
    if tt_foreign_key:
      add_relation('tt_foreign_key_forward')
      add_relation('tt_foreign_key_backward')
      add_relation('tt_foreign_key_both')
    add_rel_dist('tt_dist', tt_max_dist)

    # schema linking relations
    # forward_backward
    if sc_link:
      add_relation('qcCEM')
      add_relation('cqCEM')
      add_relation('qtTEM')
      add_relation('tqTEM')
      add_relation('qcCPM')
      add_relation('cqCPM')
      add_relation('qtTPM')
      add_relation('tqTPM')

    if cv_link:
      add_relation("qcNUMBER")
      add_relation("cqNUMBER")
      add_relation("qcTIME")
      add_relation("cqTIME")
      add_relation("qcCELLMATCH")
      add_relation("cqCELLMATCH")

    if merge_types:
      assert not cc_foreign_key
      assert not cc_table_match
      assert not ct_foreign_key
      assert not ct_table_match
      assert not tc_foreign_key
      assert not tc_table_match
      assert not tt_foreign_key

      assert cc_max_dist == qq_max_dist
      assert tt_max_dist == qq_max_dist

      add_relation('xx_default')
      self.relation_ids['qc_default'] = self.relation_ids['xx_default']
      self.relation_ids['qt_default'] = self.relation_ids['xx_default']
      self.relation_ids['cq_default'] = self.relation_ids['xx_default']
      self.relation_ids['cc_default'] = self.relation_ids['xx_default']
      self.relation_ids['ct_default'] = self.relation_ids['xx_default']
      self.relation_ids['tq_default'] = self.relation_ids['xx_default']
      self.relation_ids['tc_default'] = self.relation_ids['xx_default']
      self.relation_ids['tt_default'] = self.relation_ids['xx_default']

      if sc_link:
        self.relation_ids['qcCEM'] = self.relation_ids['xx_default']
        self.relation_ids['qcCPM'] = self.relation_ids['xx_default']
        self.relation_ids['qtTEM'] = self.relation_ids['xx_default']
        self.relation_ids['qtTPM'] = self.relation_ids['xx_default']
        self.relation_ids['cqCEM'] = self.relation_ids['xx_default']
        self.relation_ids['cqCPM'] = self.relation_ids['xx_default']
        self.relation_ids['tqTEM'] = self.relation_ids['xx_default']
        self.relation_ids['tqTPM'] = self.relation_ids['xx_default']
      if cv_link:
        self.relation_ids["qcNUMBER"] = self.relation_ids['xx_default']
        self.relation_ids["cqNUMBER"] = self.relation_ids['xx_default']
        self.relation_ids["qcTIME"] = self.relation_ids['xx_default']
        self.relation_ids["cqTIME"] = self.relation_ids['xx_default']
        self.relation_ids["qcCELLMATCH"] = self.relation_ids['xx_default']
        self.relation_ids["cqCELLMATCH"] = self.relation_ids['xx_default']

      for i in range(-qq_max_dist, qq_max_dist + 1):
        self.relation_ids['cc_dist', i] = self.relation_ids['qq_dist', i]
        self.relation_ids['tt_dist', i] = self.relation_ids['tt_dist', i]

    if ff_size is None:
      ff_size = hidden_size * 4
    self.encoder = Encoder(
      lambda: EncoderLayer(
        hidden_size,
        MultiHeadedAttentionWithRelations(
          num_heads,
          hidden_size,
          dropout),
        PositionwiseFeedForward(
          hidden_size,
          ff_size,
          dropout),
        len(self.relation_ids),
        dropout),
      hidden_size,
      num_layers,
      tie_layers)

    self.align_attn = PointerWithRelations(hidden_size,
                                                       len(self.relation_ids), dropout)

  def create_align_mask(self, num_head, q_length, c_length, t_length):
    # mask with size num_heads * all_len * all * len
    all_length = q_length + c_length + t_length
    mask_1 = torch.ones(num_head - 1, all_length, all_length, device=next(self.parameters()).device)
    mask_2 = torch.zeros(1, all_length, all_length, device=next(self.parameters()).device)
    for i in range(q_length):
      for j in range(q_length, q_length + c_length):
        mask_2[0, i, j] = 1
        mask_2[0, j, i] = 1
    mask = torch.cat([mask_1, mask_2], 0)
    return mask

  def forward_unbatched(self, desc, q_enc, c_enc, c_boundaries, t_enc, t_boundaries):
    # enc shape: total len x batch (=1) x recurrent size
    enc = torch.cat((q_enc, c_enc, t_enc), dim=0)

    # enc shape: batch (=1) x total len x recurrent size
    enc = enc.transpose(0, 1)

    # Catalogue which things are where
    relations = self.compute_relations(
      desc,
      enc_length=enc.shape[1],
      q_enc_length=q_enc.shape[0],
      c_enc_length=c_enc.shape[0],
      c_boundaries=c_boundaries,
      t_boundaries=t_boundaries)

    relations_t = torch.LongTensor(relations).to(next(self.parameters()).device)
    enc_new = self.encoder(enc, relations_t, mask=None)

    # Split updated_enc again
    c_base = q_enc.shape[0]
    t_base = q_enc.shape[0] + c_enc.shape[0]
    q_enc_new = enc_new[:, :c_base]
    c_enc_new = enc_new[:, c_base:t_base]
    t_enc_new = enc_new[:, t_base:]

    m2c_align_mat = self.align_attn(enc_new, enc_new[:, c_base:t_base], \
                                    enc_new[:, c_base:t_base], relations_t[:, c_base:t_base])
    m2t_align_mat = self.align_attn(enc_new, enc_new[:, t_base:], \
                                    enc_new[:, t_base:], relations_t[:, t_base:])
    return q_enc_new, c_enc_new, t_enc_new, (m2c_align_mat, m2t_align_mat)

  def compute_relations(self, desc, enc_length, q_enc_length, c_enc_length, c_boundaries, t_boundaries):
    sc_link = desc.get('sc_link', {'q_col_match': {}, 'q_tab_match': {}})
    cv_link = desc.get('cv_link', {'num_date_match': {}, 'cell_match': {}})

    # Catalogue which things are where
    loc_types = {}
    for i in range(q_enc_length):
      loc_types[i] = ('question',)

    c_base = q_enc_length
    for c_id, (c_start, c_end) in enumerate(zip(c_boundaries, c_boundaries[1:])):
      for i in range(c_start + c_base, c_end + c_base):
        loc_types[i] = ('column', c_id)
    t_base = q_enc_length + c_enc_length
    for t_id, (t_start, t_end) in enumerate(zip(t_boundaries, t_boundaries[1:])):
      for i in range(t_start + t_base, t_end + t_base):
        loc_types[i] = ('table', t_id)

    relations = np.empty((enc_length, enc_length), dtype=np.int64)

    for i, j in itertools.product(range(enc_length), repeat=2):
      def set_relation(name):
        relations[i, j] = self.relation_ids[name]

      i_type, j_type = loc_types[i], loc_types[j]
      if i_type[0] == 'question':
        if j_type[0] == 'question':
          set_relation(('qq_dist', clamp(j - i, self.qq_max_dist)))
        elif j_type[0] == 'column':
          # set_relation('qc_default')
          j_real = j - c_base
          if f"{i},{j_real}" in sc_link["q_col_match"]:
            set_relation("qc" + sc_link["q_col_match"][f"{i},{j_real}"])
          elif f"{i},{j_real}" in cv_link["cell_match"]:
            set_relation("qc" + cv_link["cell_match"][f"{i},{j_real}"])
          elif f"{i},{j_real}" in cv_link["num_date_match"]:
            set_relation("qc" + cv_link["num_date_match"][f"{i},{j_real}"])
          else:
            set_relation('qc_default')
        elif j_type[0] == 'table':
          # set_relation('qt_default')
          j_real = j - t_base
          if f"{i},{j_real}" in sc_link["q_tab_match"]:
            set_relation("qt" + sc_link["q_tab_match"][f"{i},{j_real}"])
          else:
            set_relation('qt_default')

      elif i_type[0] == 'column':
        if j_type[0] == 'question':
          # set_relation('cq_default')
          i_real = i - c_base
          if f"{j},{i_real}" in sc_link["q_col_match"]:
            set_relation("cq" + sc_link["q_col_match"][f"{j},{i_real}"])
          elif f"{j},{i_real}" in cv_link["cell_match"]:
            set_relation("cq" + cv_link["cell_match"][f"{j},{i_real}"])
          elif f"{j},{i_real}" in cv_link["num_date_match"]:
            set_relation("cq" + cv_link["num_date_match"][f"{j},{i_real}"])
          else:
            set_relation('cq_default')
        elif j_type[0] == 'column':
          col1, col2 = i_type[1], j_type[1]
          if col1 == col2:
            set_relation(('cc_dist', clamp(j - i, self.cc_max_dist)))
          else:
            set_relation('cc_default')
            if self.cc_foreign_key:
              if desc['foreign_keys'].get(str(col1)) == col2:
                set_relation('cc_foreign_key_forward')
              if desc['foreign_keys'].get(str(col2)) == col1:
                set_relation('cc_foreign_key_backward')
            if (self.cc_table_match and
                  desc['column_to_table'][str(col1)] == desc['column_to_table'][str(col2)]):
              set_relation('cc_table_match')

        elif j_type[0] == 'table':
          col, table = i_type[1], j_type[1]
          set_relation('ct_default')
          if self.ct_foreign_key and self.match_foreign_key(desc, col, table):
            set_relation('ct_foreign_key')
          if self.ct_table_match:
            col_table = desc['column_to_table'][str(col)]
            if col_table == table:
              if col in desc['primary_keys']:
                set_relation('ct_primary_key')
              else:
                set_relation('ct_table_match')
            elif col_table is None:
              set_relation('ct_any_table')

      elif i_type[0] == 'table':
        if j_type[0] == 'question':
          # set_relation('tq_default')
          i_real = i - t_base
          if f"{j},{i_real}" in sc_link["q_tab_match"]:
            set_relation("tq" + sc_link["q_tab_match"][f"{j},{i_real}"])
          else:
            set_relation('tq_default')
        elif j_type[0] == 'column':
          table, col = i_type[1], j_type[1]
          set_relation('tc_default')

          if self.tc_foreign_key and self.match_foreign_key(desc, col, table):
            set_relation('tc_foreign_key')
          if self.tc_table_match:
            col_table = desc['column_to_table'][str(col)]
            if col_table == table:
              if col in desc['primary_keys']:
                set_relation('tc_primary_key')
              else:
                set_relation('tc_table_match')
            elif col_table is None:
              set_relation('tc_any_table')
        elif j_type[0] == 'table':
          table1, table2 = i_type[1], j_type[1]
          if table1 == table2:
            set_relation(('tt_dist', clamp(j - i, self.tt_max_dist)))
          else:
            set_relation('tt_default')
            if self.tt_foreign_key:
              forward = table2 in desc['foreign_keys_tables'].get(str(table1), ())
              backward = table1 in desc['foreign_keys_tables'].get(str(table2), ())
              if forward and backward:
                set_relation('tt_foreign_key_both')
              elif forward:
                set_relation('tt_foreign_key_forward')
              elif backward:
                set_relation('tt_foreign_key_backward')
    return relations

  @classmethod
  def match_foreign_key(cls, desc, col, table):
    foreign_key_for = desc['foreign_keys'].get(str(col))
    if foreign_key_for is None:
      return False

    foreign_table = desc['column_to_table'][str(foreign_key_for)]
    return desc['column_to_table'][str(col)] == foreign_table


