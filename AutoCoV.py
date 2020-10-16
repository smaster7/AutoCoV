import torch
import torch.nn as nn
import torch.nn.functional as F


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        # for encoder
        self.enc_fc1 = nn.Linear(FEATURE_LEN, 200)
        self.enc_fc2 = nn.Linear(200, 100)
        self.enc_fc3 = nn.Linear(100, 20)
        self.enc_fc4 = nn.Linear(20, 2)

        self.enc_bn1 = nn.BatchNorm1d(200)
        self.enc_bn2 = nn.BatchNorm1d(100)
        self.enc_bn3 = nn.BatchNorm1d(20)
        self.enc_bn4 = nn.BatchNorm1d(2)

        # for decoder
        self.dec_fc1 = nn.Linear(2, 20)
        self.dec_fc2 = nn.Linear(20, 100)
        self.dec_fc3 = nn.Linear(100, 200)
        self.dec_fc4 = nn.Linear(200,FEATURE_LEN)

        self.dec_bn1 = nn.BatchNorm1d(20)
        self.dec_bn2 = nn.BatchNorm1d(100)
        self.dec_bn3 = nn.BatchNorm1d(200)

        # === for classifier === #
        self.cf_fc1 = nn.Linear(2, 16)
        self.cf_fc2 = nn.Linear(16, NUM_CLASSES)

        self.cf_bn1 = nn.BatchNorm1d(16)

        self.dropout = nn.Dropout(DROP_OUT_RATE)

    def encoder(self, x):
        x = self.dropout(torch.tanh(self.enc_bn1(self.enc_fc1(x))))
        x = self.dropout(torch.tanh(self.enc_bn2(self.enc_fc2(x))))
        x = self.dropout(torch.tanh(self.enc_bn3(self.enc_fc3(x))))
        x = torch.tanh(self.enc_bn4(self.enc_fc4(x)))
        return x

    def decoder(self, x):
        x = self.dropout(torch.tanh(self.dec_bn1(self.dec_fc1(x))))
        x = self.dropout(torch.tanh(self.dec_bn2(self.dec_fc2(x))))
        x = self.dropout(torch.tanh(self.dec_bn3(self.dec_fc3(x))))
        x = torch.tanh(self.dec_fc4(x))
        return x

    def classifier(self, z):
        out = self.dropout(torch.tanh(self.cf_bn1(self.cf_fc1(z))))
        out = self.cf_fc2(out)
        return out

    def forward(self, x):
        z = self.encoder(x)
        x_out = self.decoder(z)
        pred = self.classifier(z)
        return x_out, pred, z