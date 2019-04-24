import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

from u import Path, from_torch, to_torch, Flatten
from util import *

class ResNetGomoku(ResNet):
    def __init__(self, config):
        nn.Module.__init__(self)

        block = BasicBlock if config.get('res_basic_block', True) else Bottleneck 
        inplanes = config.get('res_inplanes', [64, 128, 256, 512])
        num_blocks = config.get('res_num_blocks', [3, 4, 6, 3])
        self.inplanes = inplanes[0]
        self.groups = config.get('res_groups', 1)
        self.base_width = config.get('res_base_width', 64)
        
        layers = [self._make_layer(block, inplane, num_block) for inplane, num_block in zip(inplanes, num_blocks)]
        self.shared = nn.Sequential(
            nn.Conv2d(2, inplanes[0], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(inplanes[0]),
            nn.ReLU(inplace=True),
            *layers
        )

        out_plane = inplanes[-1]

        self.value_head = nn.Sequential(
            nn.Conv2d(out_plane, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(config.board_dim ** 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh()
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(out_plane, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(2 * config.board_dim ** 2, config.board_dim ** 2)
        )

        self.loss_policy = nn.BCELoss()
        self.loss_value = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=config.lr)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if config.get('zero_init_residual', False):
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward(self, x, label_value=None, label_policy=None):
        s = self.shared(x)
        v, p = self.value_head(s), self.policy_head(s)
        mask = x.sum(dim=1).reshape(p.shape).byte()
        p.data.masked_fill_(mask, -np.inf)
        p = p.softmax(dim=-1)
        if label_value is None:
            return 0, dict(value=v, policy=p)
        loss_value = self.loss_value(v, label_value)
        loss_policy = self.loss_policy(p, label_policy)

        loss = loss_value + loss_policy
        return loss, dict(loss=loss, loss_value=loss_value, loss_policy=loss_policy)
