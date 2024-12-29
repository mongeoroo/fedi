import torch
import torch.nn as nn
import copy


class FeDi_builder(nn.Module):
    """
    Build a FeDi_builder model.
    """
    def __init__(self, base_encoder, pred_dim=8192):
        """
        pred_dim: dimension of the predictor and projector (default: 8192)
        """
        super(FeDi_builder, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=2048, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(pred_dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(pred_dim, pred_dim, bias=False)) # output layer
        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(pred_dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, pred_dim)) # output layer


        self.teacher = copy.deepcopy(self.encoder)
        for p in self.teacher.parameters():
            p.requires_grad = False


    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """

        z1 = self.predictor(self.encoder(x1)) # NxC
        z2 = self.predictor(self.encoder(x2)) # NxC

        with torch.no_grad():
            p1 = self.teacher(x1)
            p2 = self.teacher(x2)

        return z1, z2, p1.detach(), p2.detach()
