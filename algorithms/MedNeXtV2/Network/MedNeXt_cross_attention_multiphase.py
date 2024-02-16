import torch.nn as nn
import torch


class MultiPhaseFusion(nn.Module):
    def __init__(self, in_channel=768, heads=8):
        super().__init__()
        self.attn0 = CrossAttention(in_channel, heads)
        self.attn1 = CrossAttention(in_channel, heads)
        self.attn2 = CrossAttention(in_channel, heads)

    def forward(self, feat_p, feat_a, feat_v, feat_d):
        feat_vp = self.attn0(feat_v, feat_p)
        feat_va = self.attn1(feat_v, feat_a)
        feat_vd = self.attn2(feat_v, feat_d)
        out = (feat_v + feat_vp + feat_va + feat_vd) / 4
        return out


class CrossAttention(nn.Module):
    def __init__(self, in_channel=768, heads=8):
        super().__init__()
        self.heads = heads
        self.query = nn.Conv3d(in_channel, in_channel // 2, kernel_size=(1, 1, 1))
        self.key = nn.Conv3d(in_channel, in_channel // 2, kernel_size=(1, 1, 1))
        self.value = nn.Conv3d(in_channel, in_channel, kernel_size=(1, 1, 1))

    def forward(self, query, key):
        value = key
        b, c, h, w, d = query.shape
        heads = self.heads
        c_attn = (c // 2) // heads
        query = self.query(query).reshape(b, heads, -1, h * w * d)  # batch, channel, h, w, d
        key = self.key(key).reshape(b, heads, -1, h * w * d)
        value = self.value(value).reshape(b, heads, -1, h * w * d)

        attn = torch.einsum('bhci,bhcj->bhij', query, key)  # batch, head, hwd, hwd
        attn = attn / (c_attn ** 0.5)
        attn = torch.softmax(attn, -1)

        out = torch.einsum('bhij,bhcj->bhci', attn, value)  # batch, head, channel, hwd
        out = out.reshape(b, c, h, w, d)
        return out


class conv_block(nn.Module):
    def __init__(self, c_in, scale, k_size):
        super().__init__()
        self.dw_conv = nn.Conv3d(c_in, c_in, kernel_size=k_size, groups=c_in, stride=1,
                                 padding=(k_size - 1) // 2, bias=False)
        self.expansion = nn.Conv3d(c_in, c_in * scale, kernel_size=(1, 1, 1), stride=1)
        self.act = nn.GELU()
        self.compress = nn.Conv3d(c_in * scale, c_in, kernel_size=(1, 1, 1), stride=1)
        self.norm = nn.GroupNorm(32, c_in)

    def forward(self, x):
        identity = x
        out = self.norm(self.dw_conv(x))
        out = self.act(self.expansion(out))
        out = self.compress(out)
        out = torch.add(out, identity)
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
        out = torch.add(x1, x2)
        return out


class MedNeXt_multiphase(nn.Module):
    def __init__(self, in_channel, base_c, deep_supervision, k_size, num_block, scale, num_class):
        super().__init__()
        self.stem_p1 = nn.Conv3d(in_channel, base_c, kernel_size=(1, 1, 1), stride=1)
        self.stem_p2 = nn.Conv3d(in_channel, base_c, kernel_size=(1, 1, 1), stride=1)
        self.stem_p3 = nn.Conv3d(in_channel, base_c, kernel_size=(1, 1, 1), stride=1)
        self.stem_p4 = nn.Conv3d(in_channel, base_c, kernel_size=(1, 1, 1), stride=1)

        self.layer_p1_1 = self._make_layer(base_c, num_block[0], scale[0], k_size)
        self.layer_p2_1 = self._make_layer(base_c, num_block[0], scale[0], k_size)
        self.layer_p3_1 = self._make_layer(base_c, num_block[0], scale[0], k_size)
        self.layer_p4_1 = self._make_layer(base_c, num_block[0], scale[0], k_size)
        self.down_p1_1 = down_block(base_c, scale[1], k_size)
        self.down_p2_1 = down_block(base_c, scale[1], k_size)
        self.down_p3_1 = down_block(base_c, scale[1], k_size)
        self.down_p4_1 = down_block(base_c, scale[1], k_size)

        # self.fusion_1 = MultiPhaseFusion(in_channel=base_c, heads=2)

        self.layer_p1_2 = self._make_layer(base_c * 2, num_block[1], scale[1], k_size)
        self.layer_p2_2 = self._make_layer(base_c * 2, num_block[1], scale[1], k_size)
        self.layer_p3_2 = self._make_layer(base_c * 2, num_block[1], scale[1], k_size)
        self.layer_p4_2 = self._make_layer(base_c * 2, num_block[1], scale[1], k_size)
        self.down_p1_2 = down_block(base_c * 2, scale[2], k_size)
        self.down_p2_2 = down_block(base_c * 2, scale[2], k_size)
        self.down_p3_2 = down_block(base_c * 2, scale[2], k_size)
        self.down_p4_2 = down_block(base_c * 2, scale[2], k_size)

        # self.fusion_2 = MultiPhaseFusion(in_channel=base_c * 2, heads=2)

        self.layer_p1_3 = self._make_layer(base_c * 4, num_block[2], scale[2], k_size)
        self.layer_p2_3 = self._make_layer(base_c * 4, num_block[2], scale[2], k_size)
        self.layer_p3_3 = self._make_layer(base_c * 4, num_block[2], scale[2], k_size)
        self.layer_p4_3 = self._make_layer(base_c * 4, num_block[2], scale[2], k_size)
        self.down_p1_3 = down_block(base_c * 4, scale[3], k_size)
        self.down_p2_3 = down_block(base_c * 4, scale[3], k_size)
        self.down_p3_3 = down_block(base_c * 4, scale[3], k_size)
        self.down_p4_3 = down_block(base_c * 4, scale[3], k_size)

        # self.fusion_3 = MultiPhaseFusion(in_channel=base_c * 4, heads=2)

        self.layer_p1_4 = self._make_layer(base_c * 8, num_block[3], scale[3], k_size)
        self.layer_p2_4 = self._make_layer(base_c * 8, num_block[3], scale[3], k_size)
        self.layer_p3_4 = self._make_layer(base_c * 8, num_block[3], scale[3], k_size)
        self.layer_p4_4 = self._make_layer(base_c * 8, num_block[3], scale[3], k_size)
        self.down_p1_4 = down_block(base_c * 8, scale[4], k_size)
        self.down_p2_4 = down_block(base_c * 8, scale[4], k_size)
        self.down_p3_4 = down_block(base_c * 8, scale[4], k_size)
        self.down_p4_4 = down_block(base_c * 8, scale[4], k_size)

        # self.fusion_4 = MultiPhaseFusion(in_channel=base_c * 8, heads=2)

        self.layer_p1_5 = self._make_layer(base_c * 16, num_block[4], scale[4], k_size)
        self.layer_p2_5 = self._make_layer(base_c * 16, num_block[4], scale[4], k_size)
        self.layer_p3_5 = self._make_layer(base_c * 16, num_block[4], scale[4], k_size)
        self.layer_p4_5 = self._make_layer(base_c * 16, num_block[4], scale[4], k_size)

        self.fusion_5 = MultiPhaseFusion(in_channel=base_c * 16, heads=2)
        # self.bottleneck = self._make_layer(base_c * 16 * 4, num_block[5], scale[5], k_size)

        self.up1 = up_block(base_c * 16, scale[6], k_size)
        self.layer6 = self._make_layer(base_c * 8, num_block[6], scale[6], k_size)

        self.up2 = up_block(base_c * 8, scale[7], k_size)
        self.layer7 = self._make_layer(base_c * 4, num_block[7], scale[7], k_size)

        self.up3 = up_block(base_c * 4, scale[8], k_size)
        self.layer8 = self._make_layer(base_c * 2, num_block[8], scale[8], k_size)

        self.up4 = up_block(base_c * 2, scale[9], k_size)
        self.layer9 = self._make_layer(base_c, num_block[9], scale[9], k_size)

        #deep supervision
        self.do_ds = deep_supervision
        if self.do_ds:
            self.ds4 = nn.Conv3d(base_c * 16, num_class, kernel_size=(1, 1, 1), stride=1)
            self.ds3 = nn.Conv3d(base_c * 8, num_class, kernel_size=(1, 1, 1), stride=1)
            self.ds2 = nn.Conv3d(base_c * 4, num_class, kernel_size=(1, 1, 1), stride=1)
            self.ds1 = nn.Conv3d(base_c * 2, num_class, kernel_size=(1, 1, 1), stride=1)

        self.out = nn.Conv3d(base_c, num_class, kernel_size=(1, 1, 1), stride=1)

    def _make_layer(self, c_in, n_conv, ratio, k_size):
        layers = []
        for _ in range(n_conv):
            layers.append(conv_block(c_in, ratio, k_size))
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

        # fusion_1 = self.fusion_1(out_p1_1, out_p2_1, out_p3_1, out_p4_1)

        out_p1_2 = self.layer_p1_2(d_p1_1)
        d_p1_2 = self.down_p1_2(out_p1_2)
        out_p2_2 = self.layer_p2_2(d_p2_1)
        d_p2_2 = self.down_p2_2(out_p2_2)
        out_p3_2 = self.layer_p3_2(d_p3_1)
        d_p3_2 = self.down_p3_2(out_p3_2)
        out_p4_2 = self.layer_p4_2(d_p4_1)
        d_p4_2 = self.down_p4_2(out_p4_2)

        # fusion_2 = self.fusion_2(out_p1_2, out_p2_2, out_p3_2, out_p4_2)

        out_p1_3 = self.layer_p1_3(d_p1_2)
        d_p1_3 = self.down_p1_3(out_p1_3)
        out_p2_3 = self.layer_p2_3(d_p2_2)
        d_p2_3 = self.down_p2_3(out_p2_3)
        out_p3_3 = self.layer_p3_3(d_p3_2)
        d_p3_3 = self.down_p3_3(out_p3_3)
        out_p4_3 = self.layer_p4_3(d_p4_2)
        d_p4_3 = self.down_p4_3(out_p4_3)

        # fusion_3 = self.fusion_3(out_p1_3, out_p2_3, out_p3_3, out_p4_3)

        out_p1_4 = self.layer_p1_4(d_p1_3)
        d_p1_4 = self.down_p1_4(out_p1_4)
        out_p2_4 = self.layer_p2_4(d_p2_3)
        d_p2_4 = self.down_p2_4(out_p2_4)
        out_p3_4 = self.layer_p3_4(d_p3_3)
        d_p3_4 = self.down_p3_4(out_p3_4)
        out_p4_4 = self.layer_p4_4(d_p4_3)
        d_p4_4 = self.down_p4_4(out_p4_4)

        # fusion_4 = self.fusion_4(out_p1_4, out_p2_4, out_p3_4, out_p4_4)

        out_p1_5 = self.layer_p1_5(d_p1_4)
        out_p2_5 = self.layer_p2_5(d_p2_4)
        out_p3_5 = self.layer_p3_5(d_p3_4)
        out_p4_5 = self.layer_p4_5(d_p4_4)

        fusion_5 = self.fusion_5(out_p1_5, out_p2_5, out_p3_5, out_p4_5)
        # bottle_out5 = self.bottleneck(torch.cat((out_p1_5, out_p2_5, out_p3_5, out_p4_5), dim=1))
        if self.do_ds:
            out_ds4 = self.ds4(fusion_5)

        up1 = self.up1(fusion_5, out_p3_4)
        out6 = self.layer6(up1)
        if self.do_ds:
            out_ds3 = self.ds3(out6)

        up2 = self.up2(out6, out_p3_3)
        out7 = self.layer7(up2)
        if self.do_ds:
            out_ds2 = self.ds2(out7)

        up3 = self.up3(out7, out_p3_2)
        out8 = self.layer8(up3)
        if self.do_ds:
            out_ds1 = self.ds1(out8)

        up4 = self.up4(out8, out_p3_1)
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
    x = torch.rand(4, 4, 32, 32, 32)
    out = net(x)
    import pdb
    pdb.set_trace()
    print(out.shape)
