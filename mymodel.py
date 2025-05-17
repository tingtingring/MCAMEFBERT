import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from transformers import TrainingArguments, Trainer, logging
import torch



class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out, attention

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension.

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. compute self attention

        _x = x
        x, att = self.attention(q=x, k=x, v=x, mask=src_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x, att

class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)


class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x)
        for layer in self.layers:
            residual = x
            x = layer(x, src_mask)

        return x

class NCPEncoder(nn.Module):
    def __init__(self,kernel_num=256):
        super(NCPEncoder, self).__init__()

        self.ncp_dict = {
            0: [1, 1, 1],  # A
            1: [0, 1, 0],  # C
            2: [1, 0, 0],  # G
            3: [0, 0, 1]  # U/T
        }

        self.kernel_num = kernel_num

        # 卷积层
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=7, stride=3, padding=2)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # self.word_detection = nn.Sequential(
        #     nn.Conv1d(3, self.kernel_num, stride=3, kernel_size=7, padding=2),  # 修改了卷积核大小和步长
        #     nn.BatchNorm1d(kernel_num)
        # )


    def forward(self, x):
        """
        前向传播：
        1. 将输入序列进行NCP编码。
        2. 应用卷积层提取特征。
        :param x: Tensor of shape (batch_size, sequence_length)，值为 [0, 3]。
        :return: Tensor of shape (batch_size, output_length, out_channels)。
        """
        # NCP编码
        batch_size, seq_len = x.shape
        encoded = torch.zeros((batch_size, seq_len, 3), device=x.device)

        for nucleotide, vector in self.ncp_dict.items():
            encoded[x == nucleotide] = torch.tensor(vector, dtype=torch.float, device=x.device)

        # 调整维度以适应卷积层 (batch_size, embedding_dim, sequence_length)
        encoded = encoded.transpose(1, 2)

        # 应用卷积层
        x = F.relu(self.conv1(encoded))  #torch.Size([128, 64, 501])
        x = F.relu(self.conv2(x)) #torch.Size([128, 64, 250])
        x = self.maxpool(x)  #torch.Size([128, 64, 125])

        return x

class EIIP_Encoder(nn.Module):
    def __init__(self, embedding_dim=1,kernel_num=256):
        super(EIIP_Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.eiip_dict = {
            0: [0.1260],  # A
            1: [0.1340],  # C
            2: [0.0806],  # G
            3: [0.1335]   # U/T
        }

        self.kernel_num = kernel_num

        # 卷积层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=7, stride=3, padding=2)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # self.word_detection = nn.Sequential(
        #     nn.Conv1d(1, self.kernel_num, stride=3, kernel_size=7, padding=2),
        #     nn.BatchNorm1d(self.kernel_num),
        # )
    def forward(self, x):
        """
        前向传播：
        1. 将输入序列进行EIIP编码。
        2. 应用卷积层提取特征。
        :param x: Tensor of shape (batch_size, sequence_length)，值为 [0, 3]。
        :return: Tensor of shape (batch_size, output_length, out_channels)。
        """
        batch_size, seq_len = x.shape
        encoded = torch.zeros((batch_size, seq_len, self.embedding_dim), device=x.device)

        for nucleotide, vector in self.eiip_dict.items():
            encoded[x == nucleotide] = torch.tensor(vector, dtype=torch.float, device=x.device)

        # 调整维度以适应卷积层 (batch_size, embedding_dim, sequence_length)
        encoded = encoded.transpose(1, 2)

        # 应用卷积层
        x = F.relu(self.conv1(encoded))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)

        # x = F.relu(self.word_detection(encoded))

        return x

class ENAC_Encoder(nn.Module):
    def __init__(self, window_size=5,kernel_num=256):
        super(ENAC_Encoder, self).__init__()
        self.window_size = window_size

        self.kernel_num = kernel_num

        # 卷积层
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=7, stride=3, padding=2)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # self.detect_word = nn.Sequential(
        #     nn.Conv1d(4, self.kernel_num, stride=3, kernel_size=7, padding=2),
        #     nn.BatchNorm1d(self.kernel_num),
        # )

    def forward(self, x):
        """
        前向传播：
        1. 将输入序列进行ENAC编码。
        2. 应用卷积层提取特征。
        :param x: Tensor of shape (batch_size, sequence_length)，值为 [0, 3]。
        :return: Tensor of shape (batch_size, output_length, out_channels)。
        """
        batch_size, seq_len = x.shape
        encoded = torch.zeros((batch_size, seq_len, 4), device=x.device)

        # ENAC编码
        # 使用滑动窗口操作加速频率计算
        x_unfolded = x.unfold(dimension=1, size=self.window_size, step=1)  # 展开为滑动窗口
        counts = torch.nn.functional.one_hot(x_unfolded.to(torch.long), num_classes=4).sum(dim=2).float() / self.window_size
        encoded[:, self.window_size // 2:seq_len - self.window_size // 2] = counts

        # 调整维度以适应卷积层 (batch_size, 4, sequence_length)
        encoded = encoded.transpose(1, 2)

        # 应用卷积层
        # x = F.relu(self.detect_word(encoded))
        x = F.relu(self.conv1(encoded))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)

        return x

class Binary_Encoder(nn.Module):
    def __init__(self,kernel_num=256):
        super(Binary_Encoder, self).__init__()

        self.kernel_num = kernel_num

        # 卷积层
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=7, stride=3, padding=2)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # self.detect_word = nn.Sequential(
        #     nn.Conv1d(4, self.kernel_num, stride=3, kernel_size=7, padding=2),
        #     nn.BatchNorm1d(self.kernel_num),
        # )

    def forward(self, x):
        """
        前向传播：
        1. 将输入序列进行Binary/One-hot编码。
        2. 应用卷积层提取特征。
        :param x: Tensor of shape (batch_size, sequence_length)，值为 [0, 3]。
        :return: Tensor of shape (batch_size, output_length, out_channels)。
        """
        encoded = F.one_hot(x.to(torch.long), num_classes=4).float()  # One-hot 编码
        encoded = encoded.transpose(1, 2)  # 调整维度为 (batch_size, 4, sequence_length)

        x = F.relu(self.conv1(encoded))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)

        return x

class MultiChannelAttention(nn.Module):
    def __init__(self,in_channels=256*4):
        super(MultiChannelAttention, self).__init__()
        self.in_channles = in_channels

        self.out_channels1 = [128,64,64]

        self.kernel_size1 = [3, 5, 7]

        # 第一层卷积：将 DNA 序列的输入从 4 个通道扩展到较多的特征通道
        self.conv1 = nn.ModuleList([
            nn.Conv1d(in_channels=self.in_channles, out_channels=out_channels, kernel_size=ks, stride=1, padding=(ks - 1)//2)
            for out_channels, ks in zip(self.out_channels1, self.kernel_size1)
        ])

        self.channel_attention1 = ChannelAttention1D(256)

    def forward(self, inputs):

        conv1_outputs = []
        for conv in self.conv1:
            conv_output = F.relu(conv(inputs))
            conv1_outputs.append(conv_output)
        x = torch.cat(conv1_outputs, dim=1)  # 合并第一层卷积的输出
        x = self.channel_attention1(x)  # 应用通道注意力

        return x

class replace(nn.Module):
    def __init__(self,in_channels=256*4):
        super(replace, self).__init__()
        self.in_channles = in_channels

        self.out_channels1 = 256

        self.cnn = nn.Conv1d(in_channels=self.in_channles, out_channels=self.out_channels1, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):

        x = self.cnn(inputs)

        return x

class ChannelAttention1D(nn.Module):
    def __init__(self, num_channels, reduction_ratio=1):
        """
        1D Channel Attention Module with Residual Connection.

        Parameters:
        - num_channels: 输入特征的通道数
        - reduction_ratio: 压缩比率，用于减少全连接层的参数数量
        """
        super(ChannelAttention1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward propagation with residual connection.

        Parameters:
        - x: 输入特征，形状为 (batch_size, num_channels, seq_length)
        """
        # 全局平均池化，形状变为 (batch_size, num_channels, 1)
        avg_out = self.avg_pool(x).squeeze(-1)  # 去掉最后一个维度
        # 通过全连接层生成通道注意力权重
        attention_weights = self.fc(avg_out)
        # 将注意力权重应用于输入特征
        attention_output = x * attention_weights.unsqueeze(-1)  # 恢复最后一个维度
        # 添加残差连接
        return attention_output + x  # 残差连接


if __name__ == '__main__':
    TransformerEncoder = Encoder(enc_voc_size=1000,
                                    max_len=4048,
                                    d_model=512,
                                    ffn_hidden=2048,
                                    n_head=8,
                                    n_layers=6,
                                    drop_prob=0.1,
                                    device='cuda')
    TransformerEncoder.cuda()
    print(TransformerEncoder)

    input = torch.randint(0, 1000, (16, 2024))
    input = input.cuda()
    print(input)
    print(input.shape)

    import time

    start = time.time()
    output = TransformerEncoder(input, None)
    end = time.time()
    print('time : ', end - start)

    # print(output)
    # print(output.shape)