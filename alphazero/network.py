import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
            nn.Conv2d(config.state_size, inplanes[0], kernel_size=5, stride=1, padding=2),
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
            nn.Linear(256, 1)
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(out_plane, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(2 * config.board_dim ** 2, config.board_dim ** 2)
        )

        self.loss_value = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=config.lr, weight_decay=config.l2_reg)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
    
    def forward(self, x, label_value=None, label_policy=None):
        s = self.shared(x)
        v_, p_ = self.value_head(s), self.policy_head(s)
        mask = x[:, :2].sum(dim=1).byte()
        # p_.data.masked_fill_(mask.reshape(p_.shape), -np.inf)
        
        v = v_.tanh()
        p = p_.log_softmax(dim=-1).reshape(mask.shape)
        # p.data.masked_fill_(mask, 0)

        if label_value is None:
            return 0, dict(value=v, policy=p.exp())
        loss_value = self.loss_value(v.squeeze(dim=1), label_value)
        log_label = label_policy.log()
        log_label[torch.isinf(log_label)] = 0

        loss_policy = (label_policy * (log_label - p)).sum(dim=(1, 2)).mean()
        loss = loss_value + loss_policy
        entropy = -(p * p.exp()).sum(dim=(1, 2)).mean()
        return loss, dict(loss=loss, loss_value=loss_value, loss_policy=loss_policy, entropy=entropy)

class ConvNetGomoku(ResNetGomoku):
    def __init__(self, config):
        nn.Module.__init__(self)

        self.shared = nn.Sequential(
            nn.Conv2d(config.state_size, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(2 * config.board_dim ** 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1),
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(4 * config.board_dim ** 2, config.board_dim ** 2)
        )

        self.loss_value = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=config.lr, weight_decay=config.l2_reg)

class ConvNetLargeGomoku(ResNetGomoku):
    def __init__(self, config):
        nn.Module.__init__(self)

        self.shared = nn.Sequential(
            nn.Conv2d(config.state_size, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(2 * config.board_dim ** 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1),
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(4 * config.board_dim ** 2, config.board_dim ** 2)
        )

        self.loss_value = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=config.lr, weight_decay=config.l2_reg)

class FullyConvNetGomoku(nn.Module):
    def __init__(self, config):
        super(FullyConvNetGomoku, self).__init__()

        self.shared = nn.Sequential(
            nn.Conv2d(config.state_size, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Tanh()
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

        self.loss_value = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=config.lr, weight_decay=config.l2_reg)

    def forward(self, x, label_value=None, label_policy=None):
        s = self.shared(x)
        v_, p_ = self.value_head(s), self.policy_head(s)
        v = F.max_pool2d(v_, kernel_size=v_.size()[2:]).squeeze(-1).squeeze(-1)
        value = self.value_fc(v)

        batch, _, board_dim1, board_dim2 = p_.shape
        policy = p_.reshape(batch, -1).log_softmax(dim=-1).reshape(batch, board_dim1, board_dim2)
        
        if label_value is None:
            return 0, dict(value=value, policy=policy.exp())

        loss_value = self.loss_value(value.squeeze(dim=1), label_value)
        log_label = label_policy.log()
        log_label[torch.isinf(log_label)] = 0

        loss_policy = (label_policy * (log_label - policy)).sum(dim=(1, 2)).mean()
        loss = loss_value + loss_policy
        entropy = -(policy * policy.exp()).sum(dim=(1, 2)).mean()
        return loss, dict(loss=loss, loss_value=loss_value, loss_policy=loss_policy, entropy=entropy)