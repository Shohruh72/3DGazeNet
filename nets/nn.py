import torch
import timm


class GazeNet(torch.nn.Module):
    def __init__(self, backbone_id):
        super(GazeNet, self).__init__()
        self.backbone = timm.create_model(backbone_id, num_classes=481 * 2 * 3)
        self.loss = torch.nn.L1Loss(reduction='mean')
        self.hard_mining = False
        self.num_face = 1103
        self.num_eye = 481 * 2

    def forward(self, x):
        return self.backbone(x)

    def loss(self, y_hat, y, hm=False):
        bs = y.size(0)
        y_hat = y_hat.view((bs, -1, 3))
        loss = torch.abs(y_hat - y)  # (B,K,3)
        loss[:, :, 2] *= 0.5
        if hm:
            loss = torch.mean(loss, dim=(1, 2))  # (B,)
            loss, _ = torch.topk(loss, k=int(bs * 0.25), largest=True)
        loss = torch.mean(loss) * 20.0
        return loss

