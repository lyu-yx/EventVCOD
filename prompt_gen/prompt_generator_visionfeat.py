# torch libraries
#from errno import EMEDIUMTYPE
from re import X
from turtle import forward
import torch
import torch.nn as nn
import sys
import torch.nn.functional as F
import numpy as np
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompt_gen.backbone.image_encoder import ImageEncoder, FpnNeck
from prompt_gen.backbone.hieradet import Hiera
from prompt_gen.backbone.position_encoding import PositionEmbeddingSine
from prompt_gen.kan_box_predictor import KANBBoxPredictorFPN, KANBBoxPredictorVisionFeat
# customized libraries


class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
'''a –> the negative slope of the rectifier used after this layer (only used with 'leaky_relu')
   mode –> either 'fan_in' (default) or 'fan_out'. 
Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. 
Choosing 'fan_out' preserves the magnitudes in the backwards pass.
'''

class DimensionalReduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DimensionalReduction, self).__init__()
        self.reduce = nn.Sequential(
            ConvBR(in_channel, out_channel, 3, padding=1),
            ConvBR(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class SoftGroupingStrategy(nn.Module):
    def __init__(self, in_channel, out_channel, N):
        super(SoftGroupingStrategy, self).__init__()

        # grouping method is the only difference here
        self.g_conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[0], bias=False)
        self.g_conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[1], bias=False)
        self.g_conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[2], bias=False)

    def forward(self, q):
        x1 = self.g_conv1(q)
        x2 = self.g_conv2(q)
        x3 = self.g_conv3(q)
        return self.g_conv1(q) + self.g_conv2(q) + self.g_conv3(q)


class FeatureGudianceFusion(nn.Module):
    def __init__(self, channel, M, N):
        super(FeatureGudianceFusion, self).__init__()
        self.M = M

        self.downsample2 = nn.Upsample(scale_factor=1 / 2, mode='bilinear', align_corners=True)
        self.downsample4 = nn.Upsample(scale_factor=1 / 4, mode='bilinear', align_corners=True)
        
        self.sgs3 = SoftGroupingStrategy(channel + 32, channel, N=N)
        self.sgs4 = SoftGroupingStrategy(channel + 32, channel, N=N)
        self.sgs5 = SoftGroupingStrategy(channel + 32, channel, N=N)

    def forward(self, xr3, xr4, xr5, xg):
        # transmit the gradient cues into the context embeddings
        q3 = self.feature_grouping(xr3, xg, M=self.M[0])
        q4 = self.feature_grouping(xr4, self.downsample2(xg), M=self.M[1])
        q5 = self.feature_grouping(xr5, self.downsample4(xg), M=self.M[2])

        # attention residual learning
        zt3 = xr3 + self.sgs3(q3)
        zt4 = xr4 + self.sgs4(q4)
        zt5 = xr5 + self.sgs5(q5)

        return zt3, zt4, zt5

    def feature_grouping(self, xr, xg, M):
        if M == 1:
            q = torch.cat(
                (xr, xg), 1)
        elif M == 2:
            xr_g = torch.chunk(xr, 2, dim=1)
            xg_g = torch.chunk(xg, 2, dim=1)
            q = torch.cat(
                (xr_g[0], xg_g[0], xr_g[1], xg_g[1]), 1)
        elif M == 4:
            xr_g = torch.chunk(xr, 4, dim=1)
            xg_g = torch.chunk(xg, 4, dim=1)
            q = torch.cat(
                (xr_g[0], xg_g[0], xr_g[1], xg_g[1], xr_g[2], xg_g[2], xr_g[3], xg_g[3]), 1)
        elif M == 8:
            xr_g = torch.chunk(xr, 8, dim=1)
            xg_g = torch.chunk(xg, 8, dim=1)
            q = torch.cat(
                (xr_g[0], xg_g[0], xr_g[1], xg_g[1], xr_g[2], xg_g[2], xr_g[3], xg_g[3],
                 xr_g[4], xg_g[4], xr_g[5], xg_g[5], xr_g[6], xg_g[6], xr_g[7], xg_g[7]), 1)
        elif M == 16:
            xr_g = torch.chunk(xr, 16, dim=1)
            xg_g = torch.chunk(xg, 16, dim=1)
            q = torch.cat(
                (xr_g[0], xg_g[0], xr_g[1], xg_g[1], xr_g[2], xg_g[2], xr_g[3], xg_g[3],
                 xr_g[4], xg_g[4], xr_g[5], xg_g[5], xr_g[6], xg_g[6], xr_g[7], xg_g[7],
                 xr_g[8], xg_g[8], xr_g[9], xg_g[9], xr_g[10], xg_g[10], xr_g[11], xg_g[11],
                 xr_g[12], xg_g[12], xr_g[13], xg_g[13], xr_g[14], xg_g[14], xr_g[15], xg_g[15]), 1)
        elif M == 32:
            xr_g = torch.chunk(xr, 32, dim=1)
            xg_g = torch.chunk(xg, 32, dim=1)
            q = torch.cat(
                (xr_g[0], xg_g[0], xr_g[1], xg_g[1], xr_g[2], xg_g[2], xr_g[3], xg_g[3],
                 xr_g[4], xg_g[4], xr_g[5], xg_g[5], xr_g[6], xg_g[6], xr_g[7], xg_g[7],
                 xr_g[8], xg_g[8], xr_g[9], xg_g[9], xr_g[10], xg_g[10], xr_g[11], xg_g[11],
                 xr_g[12], xg_g[12], xr_g[13], xg_g[13], xr_g[14], xg_g[14], xr_g[15], xg_g[15],
                 xr_g[16], xg_g[16], xr_g[17], xg_g[17], xr_g[18], xg_g[18], xr_g[19], xg_g[19],
                 xr_g[20], xg_g[20], xr_g[21], xg_g[21], xr_g[22], xg_g[22], xr_g[23], xg_g[23],
                 xr_g[24], xg_g[24], xr_g[25], xg_g[25], xr_g[26], xg_g[26], xr_g[27], xg_g[27],
                 xr_g[28], xg_g[28], xr_g[29], xg_g[29], xr_g[30], xg_g[30], xr_g[31], xg_g[31]), 1)
        else:
            raise Exception("Invalid Group Number!")

        return q


class EdgeEstimationModule(nn.Module):
    def __init__(self,):
        super(EdgeEstimationModule, self).__init__()
        self.reduce3 = DimensionalReduction(128, 64) #56 160 448    [128, 320, 512]
        self.reduce5 = DimensionalReduction(512, 256)
        self.block = nn.Sequential(
            ConvBR(64 + 256, 256, 3, padding=1),
            ConvBR(256, 128, 3, padding=1),
            nn.Conv2d(128, 1, 1))

    def forward(self, x3, x5):
        size = x3.size()[2:]
        x3 = self.reduce3(x3)
        x5 = self.reduce5(x5)
        x5 = F.interpolate(x5, size, mode='bilinear', align_corners=False)
        out = torch.cat((x5, x3), dim=1)
        out = self.block(out)

        return out



class FusionBlock(nn.Module):
    def __init__(self, channel):
        super(FusionBlock, self).__init__()
        self.conv_zt_e = ConvBR(channel, channel, 3, padding=1)
        self.conv_zt_e2 = ConvBR(channel * 2, channel * 2, 3, padding=1)

        self.conv_zt_u = ConvBR(channel, channel, 3, padding=1)
        self.conv_zt_u2 = ConvBR(channel * 2, channel * 2, 3, padding=1)
        self.conv_zt_dr = DimensionalReduction(8 * channel, channel)
    def forward(self, zt_e, zt_u):
        zt_e0 = self.conv_zt_e(zt_e)
        zt_e1 = torch.cat((zt_e0, zt_u), dim=1)
        zt_e2 = self.conv_zt_e2(zt_e1)

        zt_u0 = self.conv_zt_u(zt_u)
        zt_u1 = torch.cat((zt_u0, zt_e),dim=1)
        zt_u2 = self.conv_zt_u2(zt_u1)

        out = self.conv_zt_dr(torch.concat((zt_e1, zt_e2, zt_u1, zt_u2), dim=1))

        return out

class UncertaintyEdgeMutualFusion(nn.Module):
    def __init__(self, channel):
        super(UncertaintyEdgeMutualFusion, self).__init__()
        self.fb_4 = FusionBlock(channel)
        self.fb_6 = FusionBlock(channel)
        self.fb_8 = FusionBlock(channel)
        
        self.x8_cbr = ConvBR(channel, channel, 3, padding=1)
        self.x6_cbr = ConvBR(channel, channel, 3, padding=1)
        self.x4_cbr = ConvBR(channel, channel, 3, padding=1)
        

        self.upsample_8_6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_6_4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_ifb = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True) #iterative feedback block
        self.upsample_f1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_8_ifb = nn.Conv2d(channel * 2, channel, 8, 4, 2)
        self.conv_4_ifb = nn.Conv2d(channel * 2, channel, 1, 1)

        self.conv_86= ConvBR(channel * 2, channel, 3, padding=1)
        self.iter_864 = ConvBR(channel * 2, channel, 3, padding=1)
        self.out_cbr = ConvBR(channel * 2, channel, 3, padding=1)

        self.conv_out = nn.Conv2d(channel, 1, 1)
        self.f1_dr = DimensionalReduction(128, 64)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
        

    def forward(self, zt4_e, zt6_e, zt8_e, zt4_u, zt6_u, zt8_u, f1, iternum=4):
        out=[]
        #size = zt4_e.size()[2:]
        out1_8 = self.fb_8(zt8_e, zt8_u) #outx_y: x denotes interation number, y denotes layer number 
        out1_6 = self.fb_6(zt6_e, zt6_u)
        out1_4 = self.fb_4(zt4_e, zt4_u)

        x8_cbr = self.x8_cbr(out1_8)
        x8_cbr = self.upsample_8_6(x8_cbr)

        x6_cbr = torch.cat((self.x6_cbr(out1_6), x8_cbr), dim=1)
        x6_cbr = self.conv_86(x6_cbr)
        x6_cbr = self.upsample_6_4(x6_cbr)

        x4_cbr = torch.cat((self.x4_cbr(out1_4), x6_cbr), dim=1)

        iter = self.iter_864(x4_cbr)

        y_out = self.upsample_f1(iter)
        y_out = torch.cat((f1, y_out), dim=1)
        y_out = self.out_cbr(y_out)
        y_out = self.conv_out(y_out)
        
        out.append(self.upsample(y_out))
        
        for _ in range(1, iternum):
            outx_8 = self.upsample_ifb(out1_8) #outx_y: x denotes interation number, y denotes layer number 
            #outx_6 = self.fb_6(zt6_e, zt6_u)
            #outx_4 = self.fb_4(zt4_e, zt4_u)

            x8_cbr = torch.cat((outx_8, iter), dim=1)
            x8_cbr = self.x8_cbr(self.conv_8_ifb(x8_cbr))
            x8_cbr = self.upsample_8_6(x8_cbr)

            x6_cbr = torch.cat((self.x6_cbr(out1_6),  x8_cbr), dim=1)
            x6_cbr = self.conv_86(x6_cbr)
            x6_cbr = self.upsample_6_4(x6_cbr)

            x4_cbr = torch.cat((out1_4, iter), dim=1)
            x4_cbr = self.conv_4_ifb(x4_cbr)
            x4_cbr = torch.cat((self.x4_cbr(out1_4), x6_cbr), dim=1)

            iter = self.iter_864(x4_cbr)
            y_out = self.upsample_f1(iter)
            y_out = torch.cat((f1, y_out), dim=1)
            y_out = self.out_cbr(y_out)
            y_out = self.conv_out(y_out)
            
            
            out.append(self.upsample(y_out))
        '''
        out = torch.cat((self.upsample_8_6(out1_8), out2), dim=1)
        out = self.conv_zt34(out)
        out = torch.cat((self.upsample_6_4(out), out3), dim=1)
        out = self.conv_zt345(out)
        '''
        return out





class PromptGenerator(nn.Module):
    def __init__(self, ckpt_pth='./checkpoints/sam2.1_hiera_base_plus.pt'):
        super(PromptGenerator, self).__init__()
        

        # self.trunk = Hiera(embed_dim=112, num_heads=2, drop_path_rate=0.1)

        # self.neck_position_encoding = PositionEmbeddingSine(num_pos_feats=256)
        # self.neck = FpnNeck(position_encoding=self.neck_position_encoding, d_model=256, 
        #                     backbone_channel_list=[896, 448, 224, 112], 
        #                     fpn_top_down_levels=[2, 3], fpn_interp_model='nearest')

        # self.image_encoder = ImageEncoder(trunk=self.trunk, neck=self.neck, scalp=1)
        # self.ckpt_pth = ckpt_pth
        
    
        checkpoint = torch.load(self.ckpt_pth, map_location="cuda")
        # Extract the state_dict of the image_encoder part
        # image_encoder_state_dict = {k[len("image_encoder."):]: v for k, v in checkpoint['model'].items() if k.startswith("image_encoder.")}
        # # Load the state_dict into the image encoder
        # self.image_encoder.load_state_dict(image_encoder_state_dict)
        # print("Image encoder weights loaded successfully.")
        
        # for param in self.image_encoder.parameters():
        #     param.requires_grad = False

        self.prompt_gen = KANBBoxPredictorVisionFeat(input_dim=256, hidden_dim=128, num_components=4)
        # self.prompt_gen = KANBBoxPredictorFPN(input_dim=256, hidden_dim=128, num_components=4)

    def forward(self, x):
        # context path (encoder)
        # visual_feats = self.image_encoder(x)
        # #endpoints = self.context_encoder.extract_endpoints(x)
        # vision_features = visual_feats["vision_features"]
        # vision_pos_enc = visual_feats["vision_pos_enc"]
        # backbone_fpn = visual_feats["backbone_fpn"]

        prompt_out = self.prompt_gen(x)
    
        return prompt_out


if __name__ == '__main__':
    net = PromptGenerator().eval()
    inputs = torch.randn(2, 3, 1024, 1024)
    outs = net(inputs)
    print(outs["vision_features"].size())
    print(outs['vision_pos_enc'][0].size())
    for i in range(len(outs['backbone_fpn'])):
        print(outs['backbone_fpn'][i].size())