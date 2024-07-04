import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .metric_model import MetricModel
from core.utils import accuracy
from .proto_net import ProtoLayer
import math


###########################_______________"FCANET.py"_____________###########################################
##############################################################################################################

def get_freq_indices(method):#获取性能最好的频率
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    #num_freq = int(method[3:]) 是将字符串 method 中从第四个字符开始（索引3）到末尾的部分转换为整数。
    #(0,0)排在第一个，是频率图中效果最好的点，说明神经网络偏爱低频
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        #mapper_x = all_top_indices_x[:num_freq] 的作用是从列表 
        #all_top_indices_x 中取出前 num_freq 个元素，并赋值给 mapper_x。
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 1, 5, 0, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 4, 0, 5, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, sigma, k, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.sigma = sigma
        self.k = k
        self.dct_h = dct_h
        self.dct_w = dct_w
        # channel: 输入数据的通道数。
        # dct_h, dct_w: 使用的离散余弦变换（DCT）的高度和宽度。
        # sigma: 第一个线性层的比例因子。
        # k: 第二个线性层输出的维度的开方。
        # freq_sel_method: 选择频率的方法，默认为'top16'，即选择前16个顶部频率。

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 5) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 5) for temp_y in mapper_y]
        #self.num_split = len(mapper_x) 计算选择的频率数量。
        # mapper_x = [temp_x * (dct_h // 5) for temp_x in mapper_x] 和 mapper_y = [temp_y * (dct_w // 5) for temp_y in mapper_y] 
        # 将频率索引映射到实际的DCT网格中。
        # 这里的 dct_h // 5 和 dct_w // 5 是将标准化后的5x5频率空间映射到实际的DCT空间大小。

        # make the frequencies in different sizes are identical to a 5x5 frequency space
        # eg, (2,2) in 10x10 is identical to (1,1) in5x5

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        #SE_BLOCK的环节，一般σ = 0.2 and k = 3
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel*self.sigma), bias=False),#降维
            nn.ReLU(inplace=True),
            nn.Linear(int(channel*self.sigma), channel*self.k**2, bias=False),#升维
            nn.Sigmoid()
        )
        #经过这个全连接层 self.fc 的结果将是一个形状为 (n, c, self.k, self.k) 的张量，
        #其中 n 是批次大小，c 是通道数，self.k 是特定的维度大小，代表网络输出的维数
    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        #如果不相同，将其调整为预期的宽高self.dct_h和self.dct_w
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        #这个是余弦变换层，将图像压缩
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, self.k, self.k)
        # return x * y.expand_as(x)
        return y

class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0#保证通道能被特征图整除

        self.num_freq = len(mapper_x)#选择前n个特征图的最好频率编号

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter
    

################################_______"INSTA.py"________####################################################
######################################################################################################################
class INSTA(nn.Module):
    def __init__(self, c, spatial_size, sigma, k, args):
        """
        Initialize the INSTA network module.
        
        Parameters:
        - c: Number of channels in the input feature map. 特征图的通道数
        - spatial_size: The height and width of the input feature map.  特征图的长宽
        - sigma: A parameter possibly used for normalization or a scale parameter in attention mechanisms.
        - k: Kernel size for convolution operations and spatial attention.
        - args: Additional arguments for setup, possibly including hyperparameters or configuration options.
        """
        super().__init__()
        self.channel = c
        self.h1 = sigma
        self.h2 = k **2
        self.k = k
        # Standard 2D convolution for channel reduction or transformation.
        self.conv = nn.Conv2d(self.channel, self.h2, 1)
        # Batch normalization for the output of the spatial attention.
        self.fn_spatial = nn.BatchNorm2d(spatial_size**2)
        # Batch normalization for the output of the channel attention.
        self.fn_channel = nn.BatchNorm2d(self.channel)
        # Unfold operation for transforming feature map into patches.
        self.Unfold = nn.Unfold(kernel_size=self.k, padding=int((self.k+1)/2-1))
        self.spatial_size = spatial_size
        # Dictionary mapping channel numbers to width/height for MultiSpectralAttentionLayer.
        c2wh = dict([(512, 11), (640, self.spatial_size)])
        # MultiSpectralAttentionLayer for performing attention across spectral (frequency) components.
        self.channel_att = MultiSpectralAttentionLayer(c, c2wh[c], c2wh[c], sigma=self.h1, k=self.k, freq_sel_method='low16')
        self.args = args
        # Upper part of a Coordinate Learning Module (CLM), which modifies feature maps.
        self.CLM_upper = nn.Sequential(
            nn.Conv2d(c, c*2, 1),
            nn.BatchNorm2d(c*2),
            nn.ReLU(),
            nn.Conv2d(c*2, c*2, 1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU()
        )

        # Lower part of CLM, transforming the features back to original channel dimensions and applying sigmoid.
        self.CLM_lower = nn.Sequential(
            nn.Conv2d(c*2, c*2, 1),
            nn.BatchNorm2d(c*2),
            nn.ReLU(),
            nn.Conv2d(c*2, c, 1),
            nn.BatchNorm2d(c),
            nn.Sigmoid()  # Sigmoid activation to normalize the feature values between 0 and 1.
        )

    def CLM(self, featuremap):#处理特征图，提取出任务特征fϕ
        """
        The Coordinate Learning Module (CLM) that processes feature maps to adapt them spatially.
        
        Parameters:
        - featuremap: The input feature map to the CLM.
        
        Returns:
        - The adapted feature map processed through the CLM.
        """
        # Apply the upper CLM to modify and then aggregate features.
        adap = self.CLM_upper(featuremap)
        intermediate = adap.sum(dim=0)  # Summing features across the batch dimension.dim=0对第一维向量挤压
        adap_1 = self.CLM_lower(intermediate.unsqueeze(0))  # Applying the lower CLM. dim和unsqeeze相当于是逆操作
        #将第0维恢复成1，self.CLM_lower 可能期望接收一个具有批处理维度的输入。
        return adap_1

    def spatial_kernel_network(self, feature_map, conv):
        """
        Applies a convolution to the feature map to generate a spatial kernel,
        which will be used to modulate the spatial regions of the input features.
        
        Parameters:
        - feature_map: The feature map to process.
        - conv: The convolutional layer to apply.
        
        Returns:
        - The processed spatial kernel.
        """
        spatial_kernel = conv(feature_map)
        spatial_kernel = spatial_kernel.flatten(-2).transpose(-1, -2)#交换倒数第一维和倒数第二维，扁平化直到倒数第二维只剩下最后一个维度不展平
        #如果 spatial_kernel 的形状是 (batch_size, channels_out, height_out, width_out)，
        # 经过 flatten(-2) 后，形状将变为 (batch_size, channels_out, height_out * width_out)。-2为起始展平维度
        #transpose(-1, -2) 操作会交换张量的倒数第一和倒数第二个维度。在我们的情况下，
        # 这意味着将 height_out * width_out 这个新展平的维度与 channels_out 维度进行交换位置。
        size = spatial_kernel.size()#获取核的维度
        spatial_kernel = spatial_kernel.view(size[0], -1, self.k, self.k)
        #-1：这个值告诉 view() 函数根据原张量的大小和其他指定的维度，自动计算此位置的维度大小。
        # 在这里，它表示在不指定具体大小的情况下，由 PyTorch 自动计算其大小以保持总元素数量不变。
        spatial_kernel = self.fn_spatial(spatial_kernel)

        spatial_kernel = spatial_kernel.flatten(-2)
        return spatial_kernel

    def channel_kernel_network(self, feature_map):
        """
        Processes the feature map through a channel attention mechanism to modulate the channels
        based on their importance.
        
        Parameters:
        - feature_map: The feature map to process.
        
        Returns:
        - The channel-modulated feature map.
        """
        channel_kernel = self.channel_att(feature_map)
        channel_kernel = self.fn_channel(channel_kernel)
        channel_kernel = channel_kernel.flatten(-2)
        channel_kernel = channel_kernel.squeeze().view(channel_kernel.shape[0], self.channel, -1)
        return channel_kernel

    def unfold(self, x, padding, k):#手动截取
        """
        A manual implementation of the unfold operation, which extracts sliding local blocks from a batched input tensor.
        
        Parameters:
        - x: The input tensor.
        - padding: Padding to apply to the tensor.
        - k: Kernel size for the blocks to extract.
        
        Returns:
        - The unfolded tensor containing all local blocks.
        """
        x_padded = torch.cuda.FloatTensor(x.shape[0], x.shape[1], x.shape[2] + 2 * padding, x.shape[3] + 2 * padding).fill_(0)
        x_padded[:, :, padding:-padding, padding:-padding] = x
        x_unfolded = torch.cuda.FloatTensor(*x.shape, k, k).fill_(0)
        for i in range(int((self.k+1)/2-1), x.shape[2] + int((self.k+1)/2-1)): 
            for j in range(int((self.k+1)/2-1), x.shape[3] + int((self.k+1)/2-1)):
                x_unfolded[:, :, i - int(((self.k+1)/2-1)), j - int(((self.k+1)/2-1)), :, :] = x_padded[:, :, i-int(((self.k+1)/2-1)):i + int((self.k+1)/2), j - int(((self.k+1)/2-1)):j + int(((self.k+1)/2))]
        return x_unfolded

    def forward(self, x):
        """
        The forward method of INSTA, which combines the spatial and channel kernels to adapt the feature map,
        along with performing the unfolding operation to facilitate local receptive processing.
        
        Parameters:
        - x: The input tensor to the network.
        
        Returns:
        - The adapted feature map and the task-specific kernel used for adaptation.
        """
        spatial_kernel = self.spatial_kernel_network(x, self.conv).unsqueeze(-3)
        channel_kernenl = self.channel_kernel_network(x).unsqueeze(-2)
        kernel = spatial_kernel * channel_kernenl  # Combine spatial and channel kernels
        # Resize kernel and apply to the unfolded feature map
        kernel_shape = kernel.size()
        feature_shape = x.size()
        instance_kernel = kernel.view(kernel_shape[0], kernel_shape[1], feature_shape[-2], feature_shape[-1], self.k, self.k)
        task_s = self.CLM(x)  # Get task-specific representation
        spatial_kernel_task = self.spatial_kernel_network(task_s, self.conv).unsqueeze(-3)
        channel_kernenl_task = self.channel_kernel_network(task_s).unsqueeze(-2)
        task_kernel = spatial_kernel_task * channel_kernenl_task
        task_kernel_shape = task_kernel.size()
        task_kernel = task_kernel.view(task_kernel_shape[0], task_kernel_shape[1], feature_shape[-2], feature_shape[-1], self.k, self.k)
        kernel = task_kernel * instance_kernel
        unfold_feature = self.unfold(x, int((self.k+1)/2-1), self.k)  # Perform a custom unfold operation
        adapted_feauture = (unfold_feature * kernel).mean(dim=(-1, -2)).squeeze(-1).squeeze(-1)
        return adapted_feauture + x, task_kernel  # Return the normal training output and task-specific kernel


##################################_______"INSTA_PROTONET.py"__________#######################################
#############################################################################################################



class Insta_ProtoNet(MetricModel):
    def __init__(self,args,**kwargs):
        super(Insta_ProtoNet, self).__init__(**kwargs)
        self.args = args
        # from model.models.ddf import DDF
        if args.backbone_class == 'Res12':
            hdim = 640
            from ..backbone.resnet_12 import ResNet#这个地方路径要改
            self.emb_func = ResNet()
        elif args.backbone_class == 'Res18':
            hdim = 512
            from ..backbone.resnet_18 import ResNet
            self.emb_func = ResNet()
        else:
            raise ValueError('')
        
        self.INSTA = INSTA(640, 5, 0.2, 3, args=args)
        self.proto_layer=ProtoLayer()
        self.loss_func = nn.CrossEntropyLoss()

    def inner_loop(self, proto, support):
        #对使用一个最优化循环进行元学习微调
        
        # Clone and detach prototypes to prevent gradients from accumulating across episodes.
        #proto 是初始原型，通过 clone().detach() 克隆并分离，确保在优化过程中不会累积梯度。
        #然后将其转换为 nn.Parameter，使其能够被 PyTorch 的优化器优化。
        SFC = proto.clone().detach()
        SFC = nn.Parameter(SFC, requires_grad=True)

        # Initialize an SGD optimizer specifically for this inner loop.
        optimizer = torch.optim.SGD([SFC], lr=0.6, momentum=0.9, dampening=0.9, weight_decay=0)

        # Create labels for the support set, used in cross-entropy loss during fine-tuning.
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        label_shot = label_shot.type(torch.cuda.LongTensor)
        
        # Perform gradient steps to update the prototypes.
        #对每个批次反向传播
        with torch.enable_grad():
            for k in range(50):  # Number of gradient steps.
                rand_id = torch.randperm(self.args.way * self.args.shot).cuda()
                for j in range(0, self.args.way * self.args.shot, 4):
                    selected_id = rand_id[j: min(j + 4, self.args.way * self.args.shot)]
                    batch_shot = support[selected_id, :]
                    batch_label = label_shot[selected_id]
                    optimizer.zero_grad()
                    logits = self.classifier(batch_shot.detach(), SFC)
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)
                    loss = F.cross_entropy(logits, batch_label)
                    loss.backward()
                    optimizer.step()
        return SFC
    def classifier(self, query, proto):#计算欧氏距离
        logits = -torch.sum((proto.unsqueeze(0) - query.unsqueeze(1)) ** 2, 2) / self.args.temperature
        return logits.squeeze()
    def my_INSTA(self,support_feat, query_feat):#这里直接用的是libfewshot的split_instance函数传进去的
        
        emb_dim = support_feat.size()[-3:]#通道和宽高训练集和测试集都是一样的(C,H,W)
        channel_dim = emb_dim[0]#C
        num_samples = self.shot_num
        num_proto = self.way_num
        #现在support_feat形状：(way_num *shot_num,c,h,w)
        #原来support_feat形状   (1,shot, way,c，h,w)
        #传入INSTA的形状刚好是：(way_num *shot_num,c,h,w)，所以不用变形
        adapted_s, task_kernel = self.INSTA(support_feat)
        adapted_proto = adapted_s.view(num_samples, -1, *adapted_s.shape[1:]).mean(0)
        #(shot_num,x,)
        adapted_proto = nn.AdaptiveAvgPool2d(1)(adapted_proto).squeeze(-1).squeeze(-1)
        query = query.view(-1, *emb_dim)
        query_ = nn.AdaptiveAvgPool2d(1)((self.INSTA.unfold(query, int((task_kernel.shape[-1]+1)/2-1), task_kernel.shape[-1]) * task_kernel)).squeeze()
        query = query + query_
        adapted_q = nn.AdaptiveAvgPool2d(1)(query).squeeze(-1).squeeze(-1)
        
        return adapted_proto,adapted_q,channel_dim
   
    def set_forward(self, batch):
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )#划分为很多个episode=图像总数/(way*(支持集个数+查询集个数)
        feat = self.emb_func(image)#使用emb_func提取Image特征(B,C,H,W)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=3
        )#将特征划分为支持集和训练集的特征
        #划分完后supprot_feat/query_feat形状为(way_num *shot_num,c,h,w)/(way_num *query_num,c,h,w)

        support_feat,query_feat,channel_dim=self.my_INSTA(support_feat,query_feat)
        adapted_proto = self.inner_loop(adapted_proto, nn.AdaptiveAvgPool2d(1)(support_feat).squeeze().view(self.shot_num*self.query_num, channel_dim))
        #这个地方加了一个返回值channel_dim，当为测试阶段时使用，代替了if self.args.testing:
        output = self.proto_layer(
            query_feat, support_feat, self.way_num, self.shot_num, self.query_num
        ).reshape(episode_size * self.way_num * self.query_num, self.way_num)
        #查询集特征 query_feat、支持集特征 support_feat、类别数 self.way_num、
        #每类样本数 self.shot_num 和查询样本数 self.query_num
        acc = accuracy(output, query_target.reshape(-1))
        return output, acc
    
    def set_forward_loss(self, batch):
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )#划分为很多个episode=图像总数/(way*(支持集个数+查询集个数)
        feat = self.emb_func(image)#使用emb_func提取Image特征
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=3
        )#将特征划分为支持集和训练集的特征
        support_feat,query_feat,channel_dim=self.my_INSTA(support_feat,query_feat)
        output = self.proto_layer(
            query_feat, support_feat, self.way_num, self.shot_num, self.query_num
        ).reshape(episode_size * self.way_num * self.query_num, self.way_num)
        #查询集特征 query_feat、支持集特征 support_feat、类别数 self.way_num、
        #每类样本数 self.shot_num 和查询样本数 self.query_num
        loss = self.loss_func(output, query_target.reshape(-1))
        acc = accuracy(output, query_target.reshape(-1))

        return output, acc, loss