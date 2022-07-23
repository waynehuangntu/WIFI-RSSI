import torch.nn as nn

class DenoisingAutoEncoder(nn.Module):
    def __init__(self, input_size):
        super(DenoisingAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.Sigmoid(),
            nn.Linear(256, 512),
            nn.Sigmoid(),
            nn.Linear(512, 1024),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Linear(256, input_size),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('sigmoid')) 

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AutoRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoRegression, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, hidden_size),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Sigmoid(),
            nn.Linear(64, 128),
            nn.Sigmoid(),
            nn.Linear(128, input_size),
            nn.Sigmoid()
        )

        self.regression = nn.Sequential(            
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )

        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('sigmoid'))
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('sigmoid'))
        
        for m in self.regression.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = self.encoder(x)
        o1 = self.decoder(x)
        o2 = self.regression(x)
        return (o1, o2)



