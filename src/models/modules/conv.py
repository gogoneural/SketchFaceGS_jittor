import jittor as jt
from jittor import nn

class ZeroConv(nn.Module):
    def __init__(self, in_ch, out_ch, hidden_ch=None, n_layers=2, kernel_size=3, padding=1):
        super().__init__()

        # 默认隐藏通道 = out_ch（常见做法）
        if hidden_ch is None:
            hidden_ch = out_ch

        layers = []

        # 第 1 层：in_ch → hidden_ch
        layers.append(nn.Conv2d(in_ch, hidden_ch, kernel_size, padding=padding))
        layers.append(nn.SiLU())

        # 中间层：(hidden_ch → hidden_ch) * (n_layers - 2)
        for _ in range(n_layers - 2):
            layers.append(nn.Conv2d(hidden_ch, hidden_ch, kernel_size, padding=padding))
            layers.append(nn.SiLU())

        # 最后一层：hidden_ch → out_ch
        layers.append(nn.Conv2d(hidden_ch, out_ch, kernel_size, padding=padding))

        # 组合为 sequential
        self.net = nn.Sequential(*layers)

        # —— 全零初始化 —— #
        for m in self.net:
            if isinstance(m, nn.Conv2d):
                jt.init.constant_(m.weight, 0.0)
                if m.bias is not None:
                    jt.init.constant_(m.bias, 0.0)

    def execute(self, x):
        return self.net(x)
