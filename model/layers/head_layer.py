from torch import nn
    
class Head(nn.Module): 

    # 输出层，在本层上进行输出信息调节
    def __init__(self, smiles_embed_dim, dropout=0.0,desc_skip_connection = True,act='GELU'):
        super().__init__()
        print(f"Initializing an instance of {self.__class__.__name__}")
        self.desc_skip_connection = desc_skip_connection 

        fc1_out_emb = smiles_embed_dim if self.desc_skip_connection else smiles_embed_dim//2
        self.fc1 = nn.Linear(smiles_embed_dim, fc1_out_emb)
        self.dropout1 = nn.Dropout(dropout)
        self.activate1 = nn.GELU() if act=='GELU' else nn.ReLU()

        fc2_out_emb = fc1_out_emb if self.desc_skip_connection else fc1_out_emb//2
        self.fc2 = nn.Linear(fc1_out_emb,  fc2_out_emb)
        self.activate2 = nn.GELU() if act=='GELU' else nn.ReLU()

        fc3_out_emb =  fc2_out_emb//2
        self.fc3 = nn.Linear(fc2_out_emb, fc3_out_emb)
        self.activate3 = nn.GELU() if act=='GELU' else nn.ReLU()

        self.final = nn.Linear(fc3_out_emb, 1)
        self.activate4 = nn.GELU() if act=='GELU' else nn.ReLU()
    
    
    def forward(self, x):

        x_out = self.fc1(x)
        x_out = self.activate1(x_out)
        x_out = self.dropout1(x_out)

        if self.desc_skip_connection is True:
            x_out = x_out + x

        z = self.fc2(x_out)
        z = self.activate2(z)

        if self.desc_skip_connection is True:
            z = self.fc3(z + x_out)
        else:
            z = self.fc3(z)

        z = self.activate3(z)
        z = self.final(z)
        z = self.activate4(z)

        return z