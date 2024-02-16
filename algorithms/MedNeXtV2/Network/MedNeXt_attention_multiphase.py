import torch.nn as nn
import torch
from networks.network_init import init_weights
from networks.grid_attention_layer import GridAttentionBlock3D


class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor= sub_sample_factor)
        self.gate_block_2 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv3d(in_size*2, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm3d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock3D') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)
        gate_2, attention_2 = self.gate_block_2(input, gating_signal)

        return self.combine_gates(torch.cat([gate_1, gate_2], 1)), torch.cat([attention_1, attention_2], 1)


class conv_block(nn.Module):
    def __init__(self, c_in, scale, k_size, c_output):
        super().__init__()
        self.dw_conv = nn.Conv3d(c_in, c_in, kernel_size=k_size, groups=c_in, stride=1,
                                 padding=(k_size - 1) // 2, bias=False)
        self.expansion = nn.Conv3d(c_in, c_in * scale, kernel_size=(1, 1, 1), stride=1)
        self.act = nn.GELU()
        self.compress = nn.Conv3d(c_in * scale, c_output, kernel_size=(1, 1, 1), stride=1)
        self.norm = nn.GroupNorm(32, c_in)

    def forward(self, x):
        identity = x
        out = self.norm(self.dw_conv(x))
        out = self.act(self.expansion(out))
        out = self.compress(out)
        out = torch.add(out, identity)
        return out


class point_conv_block(nn.Module):
    def __init__(self, c_in, scale, k_size, c_output):
        super().__init__()
        self.dw_conv = nn.Conv3d(c_in, c_in, kernel_size=k_size, groups=c_in, stride=1,
                                 padding=(k_size - 1) // 2, bias=False)
        self.expansion = nn.Conv3d(c_in, c_in * scale, kernel_size=(1, 1, 1), stride=1)
        self.act = nn.GELU()
        self.compress = nn.Conv3d(c_in * scale, c_output, kernel_size=(1, 1, 1), stride=1)
        self.norm = nn.GroupNorm(32, c_in)
        self.shortcut = nn.Conv3d(in_channels=c_in, out_channels=c_output, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.norm(self.dw_conv(x))
        out = self.act(self.expansion(out))
        out = self.compress(out)
        out = torch.add(out, shortcut)
        return out



class down_block(nn.Module):
    def __init__(self, c_in, scale, k_size):
        super().__init__()
        self.dw_conv = nn.Conv3d(c_in, c_in, kernel_size=k_size, groups=c_in, stride=2, padding=(k_size - 1) // 2,
                                 bias=False)
        self.norm = nn.GroupNorm(32, c_in)
        self.expansion = nn.Conv3d(c_in, scale * c_in, kernel_size=(1, 1, 1), stride=1)
        self.act = nn.GELU()
        self.compress = nn.Conv3d(scale * c_in, 2 * c_in, kernel_size=(1, 1, 1), stride=1)
        self.shortcut = nn.Conv3d(c_in, 2 * c_in, kernel_size=(1, 1, 1), stride=2)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.norm(self.dw_conv(x))
        out = self.act(self.expansion(out))
        out = self.compress(out)
        out = torch.add(out, shortcut)
        return out



class up_block(nn.Module):
    def __init__(self, c_in, scale, k_size):
        super().__init__()
        self.dw_conv = nn.ConvTranspose3d(c_in, c_in, kernel_size=k_size, stride=2, padding=(k_size - 1) // 2,
                                          output_padding=1, groups=c_in, bias=False)
        self.norm = nn.GroupNorm(32, c_in)
        self.expansion = nn.Conv3d(c_in, c_in * scale, kernel_size=(1, 1, 1), stride=1)
        self.act = nn.GELU()
        self.compress = nn.Conv3d(c_in * scale, c_in // 2, kernel_size=(1, 1, 1), stride=1)
        self.shortcut = nn.ConvTranspose3d(c_in, c_in // 2, kernel_size=(1, 1, 1), output_padding=1, stride=2)

    def forward(self, x1, x2):
        short = self.shortcut(x1)
        x1 = self.norm(self.dw_conv(x1))
        x1 = self.act(self.expansion(x1))
        x1 = self.compress(x1)
        x1 = torch.add(x1, short)
        out = torch.cat([x1, x2], dim=1)
        return out


class MedNeXt_multiphase(nn.Module):
    def __init__(self, in_channel, base_c, deep_supervision, k_size, num_block, scale, num_class):
        super().__init__()
        self.stem_p1 = nn.Conv3d(in_channel, base_c, kernel_size=(1, 1, 1), stride=1)
        self.stem_p2 = nn.Conv3d(in_channel, base_c, kernel_size=(1, 1, 1), stride=1)
        self.stem_p3 = nn.Conv3d(in_channel, base_c, kernel_size=(1, 1, 1), stride=1)
        self.stem_p4 = nn.Conv3d(in_channel, base_c, kernel_size=(1, 1, 1), stride=1)

        self.layer_p1_1 = self._make_layer(base_c, num_block[0], scale[0], k_size, base_c)
        self.layer_p2_1 = self._make_layer(base_c, num_block[0], scale[0], k_size, base_c)
        self.layer_p3_1 = self._make_layer(base_c, num_block[0], scale[0], k_size, base_c)
        self.layer_p4_1 = self._make_layer(base_c, num_block[0], scale[0], k_size, base_c)
        self.down_p1_1 = down_block(base_c, scale[1], k_size)
        self.down_p2_1 = down_block(base_c, scale[1], k_size)
        self.down_p3_1 = down_block(base_c, scale[1], k_size)
        self.down_p4_1 = down_block(base_c, scale[1], k_size)

        self.layer_p1_2 = self._make_layer(base_c * 2, num_block[1], scale[1], k_size, base_c * 2)
        self.layer_p2_2 = self._make_layer(base_c * 2, num_block[1], scale[1], k_size, base_c * 2)
        self.layer_p3_2 = self._make_layer(base_c * 2, num_block[1], scale[1], k_size, base_c * 2)
        self.layer_p4_2 = self._make_layer(base_c * 2, num_block[1], scale[1], k_size, base_c * 2)
        self.down_p1_2 = down_block(base_c * 2, scale[2], k_size)
        self.down_p2_2 = down_block(base_c * 2, scale[2], k_size)
        self.down_p3_2 = down_block(base_c * 2, scale[2], k_size)
        self.down_p4_2 = down_block(base_c * 2, scale[2], k_size)

        self.layer_p1_3 = self._make_layer(base_c * 4, num_block[2], scale[2], k_size, base_c * 4)
        self.layer_p2_3 = self._make_layer(base_c * 4, num_block[2], scale[2], k_size, base_c * 4)
        self.layer_p3_3 = self._make_layer(base_c * 4, num_block[2], scale[2], k_size, base_c * 4)
        self.layer_p4_3 = self._make_layer(base_c * 4, num_block[2], scale[2], k_size, base_c * 4)
        self.down_p1_3 = down_block(base_c * 4, scale[3], k_size)
        self.down_p2_3 = down_block(base_c * 4, scale[3], k_size)
        self.down_p3_3 = down_block(base_c * 4, scale[3], k_size)
        self.down_p4_3 = down_block(base_c * 4, scale[3], k_size)

        self.layer_p1_4 = self._make_layer(base_c * 8, num_block[3], scale[3], k_size, base_c * 8)
        self.layer_p2_4 = self._make_layer(base_c * 8, num_block[3], scale[3], k_size, base_c * 8)
        self.layer_p3_4 = self._make_layer(base_c * 8, num_block[3], scale[3], k_size, base_c * 8)
        self.layer_p4_4 = self._make_layer(base_c * 8, num_block[3], scale[3], k_size, base_c * 8)
        self.down_p1_4 = down_block(base_c * 8, scale[4], k_size)
        self.down_p2_4 = down_block(base_c * 8, scale[4], k_size)
        self.down_p3_4 = down_block(base_c * 8, scale[4], k_size)
        self.down_p4_4 = down_block(base_c * 8, scale[4], k_size)

        self.bottleneck = self._make_layer(base_c * 16, num_block[5], scale[5], k_size, base_c * 16)

        self.attentionblock1_1 = MultiAttentionBlock(in_size=base_c, gate_size=base_c * 2, inter_size=base_c,
                                                   nonlocal_mode='concatenation', sub_sample_factor=(2, 2, 2))
        self.attentionblock1_2 = MultiAttentionBlock(in_size=base_c, gate_size=base_c * 2, inter_size=base_c,
                                                   nonlocal_mode='concatenation', sub_sample_factor=(2, 2, 2))
        self.attentionblock1_4 = MultiAttentionBlock(in_size=base_c, gate_size=base_c * 2, inter_size=base_c,
                                                   nonlocal_mode='concatenation', sub_sample_factor=(2, 2, 2))

        self.attentionblock2_1 = MultiAttentionBlock(in_size=base_c * 2, gate_size=base_c * 4, inter_size=base_c * 2,
                                                   nonlocal_mode='concatenation', sub_sample_factor=(2,2,2))
        self.attentionblock2_2 = MultiAttentionBlock(in_size=base_c * 2, gate_size=base_c * 4, inter_size=base_c * 2,
                                                   nonlocal_mode='concatenation', sub_sample_factor=(2, 2, 2))
        self.attentionblock2_4 = MultiAttentionBlock(in_size=base_c * 2, gate_size=base_c * 4, inter_size=base_c * 2,
                                                   nonlocal_mode='concatenation', sub_sample_factor=(2, 2, 2))

        self.attentionblock3_1 = MultiAttentionBlock(in_size=base_c * 4, gate_size=base_c * 8, inter_size=base_c * 4,
                                                   nonlocal_mode='concatenation', sub_sample_factor=(2,2,2))
        self.attentionblock3_2 = MultiAttentionBlock(in_size=base_c * 4, gate_size=base_c * 8, inter_size=base_c * 4,
                                                   nonlocal_mode='concatenation', sub_sample_factor=(2, 2, 2))
        self.attentionblock3_4 = MultiAttentionBlock(in_size=base_c * 4, gate_size=base_c * 8, inter_size=base_c * 4,
                                                   nonlocal_mode='concatenation', sub_sample_factor=(2, 2, 2))

        self.attentionblock4_1 = MultiAttentionBlock(in_size=base_c * 8, gate_size=base_c * 16, inter_size=base_c * 8,
                                                   nonlocal_mode='concatenation', sub_sample_factor=(2,2,2))
        self.attentionblock4_2 = MultiAttentionBlock(in_size=base_c * 8, gate_size=base_c * 16, inter_size=base_c * 8,
                                                   nonlocal_mode='concatenation', sub_sample_factor=(2, 2, 2))
        self.attentionblock4_4 = MultiAttentionBlock(in_size=base_c * 8, gate_size=base_c * 16, inter_size=base_c * 8,
                                                   nonlocal_mode='concatenation', sub_sample_factor=(2, 2, 2))

        self.up1 = up_block(base_c * 16, scale[6], k_size)
        self.layer6 = self._make_layer(base_c * 16, num_block[6], scale[6], k_size, c_output=base_c * 8)

        self.up2 = up_block(base_c * 8, scale[7], k_size)
        self.layer7 = self._make_layer(base_c * 8, num_block[7], scale[7], k_size, c_output=base_c * 4)

        self.up3 = up_block(base_c * 4, scale[8], k_size)
        self.layer8 = self._make_layer(base_c * 4, num_block[8], scale[8], k_size, c_output=base_c * 2)

        self.up4 = up_block(base_c * 2, scale[9], k_size)
        self.layer9 = self._make_layer(base_c * 2, num_block[9], scale[9], k_size, c_output=base_c)

        #deep supervision
        self.do_ds = deep_supervision
        if self.do_ds:
            self.ds4 = nn.Conv3d(base_c * 16, num_class, kernel_size=(1, 1, 1), stride=1)
            self.ds3 = nn.Conv3d(base_c * 8, num_class, kernel_size=(1, 1, 1), stride=1)
            self.ds2 = nn.Conv3d(base_c * 4, num_class, kernel_size=(1, 1, 1), stride=1)
            self.ds1 = nn.Conv3d(base_c * 2, num_class, kernel_size=(1, 1, 1), stride=1)

        self.out = nn.Conv3d(base_c, num_class, kernel_size=(1, 1, 1), stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.GroupNorm):
                init_weights(m, init_type='kaiming')

    def _make_layer(self, c_in, n_conv, ratio, k_size, c_output):
        layers = []
        for i in range(n_conv):
            if c_in == c_output:
                layers.append(conv_block(c_in, ratio, k_size, c_output))
            elif i == n_conv - 1:
                layers.append(point_conv_block(c_in, ratio, k_size, c_output))
            elif i != n_conv - 1:
                layers.append(conv_block(c_in, ratio, k_size, c_in))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_p1 = self.stem_p1(x[:, 0:1, :, :, :])
        x_p2 = self.stem_p2(x[:, 1:2, :, :, :])
        x_p3 = self.stem_p3(x[:, 2:3, :, :, :])
        x_p4 = self.stem_p4(x[:, 3:4, :, :, :])

        out_p1_1 = self.layer_p1_1(x_p1)
        d_p1_1 = self.down_p1_1(out_p1_1)
        out_p2_1 = self.layer_p2_1(x_p2)
        d_p2_1 = self.down_p2_1(out_p2_1)
        out_p3_1 = self.layer_p3_1(x_p3)
        d_p3_1 = self.down_p3_1(out_p3_1)
        out_p4_1 = self.layer_p4_1(x_p4)
        d_p4_1 = self.down_p4_1(out_p4_1)

        out_p1_2 = self.layer_p1_2(d_p1_1)
        d_p1_2 = self.down_p1_2(out_p1_2)
        out_p2_2 = self.layer_p2_2(d_p2_1)
        d_p2_2 = self.down_p2_2(out_p2_2)
        out_p3_2 = self.layer_p3_2(d_p3_1)
        d_p3_2 = self.down_p3_2(out_p3_2)
        out_p4_2 = self.layer_p4_2(d_p4_1)
        d_p4_2 = self.down_p4_2(out_p4_2)

        out_p1_3 = self.layer_p1_3(d_p1_2)
        d_p1_3 = self.down_p1_3(out_p1_3)
        out_p2_3 = self.layer_p2_3(d_p2_2)
        d_p2_3 = self.down_p2_3(out_p2_3)
        out_p3_3 = self.layer_p3_3(d_p3_2)
        d_p3_3 = self.down_p3_3(out_p3_3)
        out_p4_3 = self.layer_p4_3(d_p4_2)
        d_p4_3 = self.down_p4_3(out_p4_3)

        out_p1_4 = self.layer_p1_4(d_p1_3)
        d_p1_4 = self.down_p1_4(out_p1_4)
        out_p2_4 = self.layer_p2_4(d_p2_3)
        d_p2_4 = self.down_p2_4(out_p2_4)
        out_p3_4 = self.layer_p3_4(d_p3_3)
        d_p3_4 = self.down_p3_4(out_p3_4)
        out_p4_4 = self.layer_p4_4(d_p4_3)
        d_p4_4 = self.down_p4_4(out_p4_4)

        bottle_out5 = self.bottleneck(d_p3_4)
        if self.do_ds:
            out_ds4 = self.ds4(bottle_out5)

        g_conv1_4, _ = self.attentionblock4_1(out_p1_4, bottle_out5)
        g_conv2_4, _ = self.attentionblock4_2(out_p2_4, bottle_out5)
        g_conv4_4, _ = self.attentionblock4_4(out_p4_4, bottle_out5)

        up1 = self.up1(bottle_out5, torch.add(torch.add(g_conv1_4, g_conv2_4), g_conv4_4) / 3)
        out6 = self.layer6(up1)
        if self.do_ds:
            out_ds3 = self.ds3(out6)

        g_conv1_3, _ = self.attentionblock3_1(out_p1_3, out6)
        g_conv2_3, _ = self.attentionblock3_2(out_p2_3, out6)
        g_conv4_3, _ = self.attentionblock3_4(out_p4_3, out6)

        up2 = self.up2(out6, torch.add(torch.add(g_conv1_3, g_conv2_3), g_conv4_3) / 3)
        out7 = self.layer7(up2)
        if self.do_ds:
            out_ds2 = self.ds2(out7)

        g_conv1_2, _ = self.attentionblock2_1(out_p1_2, out7)
        g_conv2_2, _ = self.attentionblock2_2(out_p2_2, out7)
        g_conv4_2, _ = self.attentionblock2_4(out_p4_2, out7)

        up3 = self.up3(out7, torch.add(torch.add(g_conv1_2, g_conv2_2), g_conv4_2) / 3)
        out8 = self.layer8(up3)
        if self.do_ds:
            out_ds1 = self.ds1(out8)

        g_conv1_1, _ = self.attentionblock1_1(out_p1_1, out8)
        g_conv2_1, _ = self.attentionblock1_2(out_p2_1, out8)
        g_conv4_1, _ = self.attentionblock1_4(out_p4_1, out8)

        up4 = self.up4(out8, torch.add(torch.add(g_conv1_1, g_conv2_1), g_conv4_1) / 3)
        out9 = self.layer9(up4)

        out = self.out(out9)

        if self.do_ds:
            return [out_ds4, out_ds3, out_ds2, out_ds1, out]
        else:
            return out


def get_mednet():
    num_block = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    scale = [2, 3, 4, 4, 4, 4, 4, 3, 2, 2]  # MedNeXt-B
    kernel_size = 3
    net = MedNeXt_multiphase(1, 16, False, kernel_size, num_block, scale, 4)
    return net


# debug
if __name__ == '__main__':
    net = get_mednet()
    x = torch.rand(4, 4, 96, 96, 32)
    out = net(x)
    import pdb
    pdb.set_trace()
    print(out.shape)
