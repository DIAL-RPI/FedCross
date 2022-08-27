import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = self.conv(x)
        return y

class enc_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(enc_block, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
        self.down = nn.MaxPool3d(2)

    def forward(self, x):
        y_conv = self.conv(x)
        y = self.down(y_conv)
        return y, y_conv

class dec_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(dec_block, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
        self.up = nn.ConvTranspose3d(out_ch, out_ch, 2, stride=2)

    def forward(self, x):
        y_conv = self.conv(x)
        y = self.up(y_conv)
        return y, y_conv

def concatenate(x1, x2):
    diffZ = x2.size()[2] - x1.size()[2]
    diffY = x2.size()[3] - x1.size()[3]
    diffX = x2.size()[4] - x1.size()[4]
    x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                    diffY // 2, diffY - diffY//2,
                    diffZ // 2, diffZ - diffZ//2))        
    y = torch.cat([x2, x1], dim=1)
    return y

class UNet(nn.Module):
    def __init__(self, in_ch, base_ch, cls_num):
        super(UNet, self).__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.cls_num = cls_num

        self.enc1 = enc_block(in_ch, base_ch)
        self.enc2 = enc_block(base_ch, base_ch*2)
        self.enc3 = enc_block(base_ch*2, base_ch*4)
        self.enc4 = enc_block(base_ch*4, base_ch*8)

        self.dec1 = dec_block(base_ch*8, base_ch*8)
        self.dec2 = dec_block(base_ch*8+base_ch*8, base_ch*4)
        self.dec3 = dec_block(base_ch*4+base_ch*4, base_ch*2)
        self.dec4 = dec_block(base_ch*2+base_ch*2, base_ch)
        self.lastconv = double_conv(base_ch+base_ch, base_ch)

        self.outconv = nn.Conv3d(base_ch, cls_num+1, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, enc1_conv = self.enc1(x)
        x, enc2_conv = self.enc2(x)
        x, enc3_conv = self.enc3(x)
        x, enc4_conv = self.enc4(x)
        x, _ = self.dec1(x)
        x, _ = self.dec2(concatenate(x, enc4_conv))
        x, _ = self.dec3(concatenate(x, enc3_conv))
        x, _ = self.dec4(concatenate(x, enc2_conv))
        x = self.lastconv(concatenate(x, enc1_conv))
        x = self.outconv(x)
        y = self.softmax(x)

        if self.training:
            return y, x
        else:
            return y

    def description(self):
        return 'U-Net (input channel = {0:d}) for {1:d}-class segmentation (base channel = {2:d})'.format(self.in_ch, self.cls_num+1, self.base_ch)