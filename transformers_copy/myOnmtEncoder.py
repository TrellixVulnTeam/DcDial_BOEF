import torch.nn as nn
import math
import torch

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)

class EncoderBase(nn.Module):
    """
    Base encoder class. Specifies the interface used by different encoder types
    and required by :obj:`onmt.Models.NMTModel`.

    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
            C[Pos 1]
            D[Pos 2]
            E[Pos N]
          end
          F[Memory_Bank]
          G[Final]
          A-->C
          A-->D
          A-->E
          C-->F
          D-->F
          E-->F
          E-->G
    """

    def _check_args(self, src, lengths=None, hidden=None):
        _, n_batch, _ = src.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, src, lengths=None):
        """
        Args:
            src (:obj:`LongTensor`):
               padded sequences of sparse indices `[src_len x batch x nfeat]`
            lengths (:obj:`LongTensor`): length of each sequence `[batch]`


        Returns:
            (tuple of :obj:`FloatTensor`, :obj:`FloatTensor`):
                * final encoder state, used to initialize decoder
                * memory bank for attention, `[src_len x batch x hidden]`
        """
        raise NotImplementedError

class KNLTransformerEncoder(EncoderBase):
    """
    The Transformer encoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings

    Returns:
        (`FloatTensor`, `FloatTensor`):

        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout):
        super(KNLTransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.transformer = nn.ModuleList(
            [STransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src, inputs_history_mask):
        """ See :obj:`EncoderBase.forward()`"""
        out = src
        for i in range(self.num_layers):
            out = self.transformer[i](out, inputs_history_mask)
        out = self.layer_norm(out)

        return out

class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`#torch.Size([1, 1, 10])
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query), \
                                    self.linear_keys(query), \
                                    self.linear_values(query)
                key = shape(key)
                value = shape(value)
                device = key.device
                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"].to(device), key),
                        dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"].to(device), value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value
            elif type == "src":
                query = self.linear_query(query)
                if layer_cache["src_memory_keys"] is None:
                    key, value = self.linear_keys(key), \
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["src_memory_keys"], \
                                 layer_cache["src_memory_values"]
                layer_cache["src_memory_keys"] = key
                layer_cache["src_memory_values"] = value
            elif type == "knl":
                query = self.linear_query(query)
                if layer_cache["knl_memory_keys"] is None:
                    key, value = self.linear_keys(key), \
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["knl_memory_keys"], \
                                 layer_cache["knl_memory_values"]
                layer_cache["knl_memory_keys"] = key
                layer_cache["knl_memory_values"] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))
        #print("scores的形状是",scores.size(),scores)#torch.Size([1, 12, 8, 8])

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            #print("mask的形状是",mask.size(),mask)#torch.Size([1, 1, 1, 8])
            scores = scores.masked_fill(mask, -1e18)
            #print("scores之后的形状是",scores.size(),scores)#torch.Size([1, 12, 8, 8])

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = unshape(torch.matmul(drop_attn, value))

        output = self.final_linear(context)
        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return one attn
        top_attn = attn \
                       .view(batch_size, head_count,
                             query_len, key_len)[:, 0, :, :] \
            .contiguous()

        return output, top_attn

class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            d_model (int): the size of input for the first-layer of the FFN.
            d_ff (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Layer definition.

        Args:
            input: [ batch_size, input_len, model_dim ]


        Returns:
            output: [ batch_size, input_len, model_dim ]
        """
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

class STransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(STransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm(inputs)  # 归一化层
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)

class MyModelTransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(MyModelTransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.knowledge_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, src_mask, knl_bank, knl_mask):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm_1(inputs)
        query, _ = self.self_attn(input_norm, input_norm, input_norm,
                                  mask=src_mask)
        query = self.dropout(query) + inputs
        query_norm = self.layer_norm_2(query)
        knl_out, _ = self.knowledge_attn(knl_bank, knl_bank, query_norm,
                                         mask=knl_mask)
        knl_out = self.dropout(knl_out)+query

        return self.feed_forward(knl_out)

class MyModelTransformerEncoderLayer2(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(MyModelTransformerEncoderLayer2, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.knowledge_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, src_mask, knl_bank, knl_mask):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm_1(inputs)
        query, _ = self.self_attn(input_norm, input_norm, input_norm,
                                  mask=src_mask)
        query = self.dropout(query) + inputs
        query_norm = self.layer_norm_2(query)
        knl_out, _ = self.knowledge_attn(knl_bank, knl_bank, query_norm,
                                         mask=knl_mask)
        knl_out = self.dropout(knl_out) + query

        return self.feed_forward(knl_out)


class MyModelEncoder(EncoderBase):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout):
        super(MyModelEncoder, self).__init__()
        self.num_layers = num_layers

        self.knltransformer = KNLTransformerEncoder(num_layers, d_model, heads,
                                                    d_ff, dropout)
        #self.knltransformer.to("cuda")
        self.srctransformer = nn.ModuleList(
            [MyModelTransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])
        self.knltransformer2 = nn.ModuleList(
            [MyModelTransformerEncoderLayer2(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])

        self.layer_norm=nn.LayerNorm(d_model, eps=1e-6)
    def forward(self, src, knl, history,inputs_src_mask,inputs_history_mask,inputs_knl_mask):
        knl_bank_tgt= self.knltransformer(knl, inputs_knl_mask)
        out1= self.knltransformer(src, inputs_src_mask)
        if history.size()[-1]!=0:
            history_bank_tgt= self.knltransformer(history,inputs_history_mask)
            for i in range(self.num_layers):
                out1 = self.srctransformer[i](out1, inputs_src_mask, history_bank_tgt, inputs_history_mask)
            out1 = self.layer_norm(out1)
        out2=out1
        for i in range(self.num_layers):
            out2 = self.knltransformer2[i](out2, inputs_src_mask, knl_bank_tgt, inputs_knl_mask)
        out2 = self.layer_norm(out2)
        #print("src是",out,out.size())
        return out1,out2




class MyModelAddDecoder(EncoderBase):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout):
        super(MyModelAddDecoder, self).__init__()
        self.num_layers = num_layers

        """self.knltransformer = KNLTransformerEncoder(num_layers, d_model, heads,
                                                    d_ff, dropout)
        self.knltransformer.to("cuda")"""
        self.transformer1 = nn.ModuleList(
            [MyModelTransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])
        self.transformer2 = nn.ModuleList(
            [MyModelTransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])
        self.layer_norm=nn.LayerNorm(d_model, eps=1e-6)
    def forward(self, src, knl, history,inputs_src_mask,inputs_history_mask,inputs_knl_mask):
        """print("src是",src,src.size())
        print("knl是",knl,knl.size())
        print("history是",history,history.size())
        print("src_lengths是",src_lengths,src_lengths.size())
        print("knl_lengths是",knl_lengths,knl_lengths.size())
        print("history_lengths是",history_lengths,history_lengths.size())"""
        #history_bank_tgt= self.knltransformer(history,inputs_history_mask)
        #knl_bank_tgt= self.knltransformer(knl, inputs_knl_mask)
        #out= self.knltransformer(src, inputs_src_mask)
        out=src
        for i in range(self.num_layers):
            out = self.transformer1[i](out, inputs_src_mask, history, inputs_history_mask)
        out = self.layer_norm(out)
        for i in range(self.num_layers):
            out = self.transformer2[i](out, inputs_src_mask, knl, inputs_knl_mask)
        out = self.layer_norm(out)
        #print("src是",out,out.size())
        return out


