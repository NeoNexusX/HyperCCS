import torch
from torch import nn
from fast_transformers.attention import AttentionLayer
from fast_transformers.attention.full_attention import FullAttention
from fast_transformers.masking import FullMask
import torch.nn.functional as F


class Aggre_Attention(nn.Module):
    def __init__(self, 
                 smiles_embed_dim, 
                 ecfp_length=1024,
                 num_adducts=10,  # 根据实际离子类型数量调整
                 mz_dim=1,
                 dropout=0.0):
        
        super().__init__()
        print(f"Initializing an instance of {self.__class__.__name__}")

        mz_dim = 0 if mz_dim ==0 else smiles_embed_dim 
        adduct_dim = smiles_embed_dim 
        ecfp_dim = 0 if ecfp_length == 0  else smiles_embed_dim

        # SMILES特征增强
        self.smiles_attn = nn.Sequential(
            nn.Linear(smiles_embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # 数值特征处理
        if mz_dim == 0:
            self.mz_net = None
            print('self.mz_net is None')
        else:
            self.mz_net = nn.Sequential(
                nn.Linear(1, mz_dim),
                nn.LayerNorm(mz_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        # 离子类型嵌入
        self.adduct_emb = nn.Embedding(num_adducts, adduct_dim)
        self.adduct_net = nn.Sequential(
            nn.Linear(adduct_dim, adduct_dim),
            nn.LayerNorm(adduct_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ECFP特征处理
        if ecfp_dim == 0 :
            self.ecfp_net = None
            print('self.ecfp_net is None')
        else:
            self.ecfp_net = nn.Sequential(
                nn.Linear(ecfp_length, ecfp_dim),
                nn.LayerNorm(ecfp_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        # 跨模态融合
        self.heads_num = 12
        self.cross_attn = AttentionLayer(
            attention=FullAttention(),  # 使用线性注意力
            d_model=smiles_embed_dim,      # 输入维度
            n_heads=self.heads_num,                     # 头数保持与原来一致
            d_keys=smiles_embed_dim//self.heads_num,    # 键维度
            d_values=smiles_embed_dim//self.heads_num,  # 值维度
            event_dispatcher=None
        )
        
        # 最终融合层
        if ecfp_length == 0 and mz_dim!= 0:
            total_dim = smiles_embed_dim * 3
        elif mz_dim == 0 and ecfp_length!= 0:
            total_dim = smiles_embed_dim * 3
        elif mz_dim == 0 and ecfp_length == 0 :
            total_dim = smiles_embed_dim * 2
        else:
            total_dim = smiles_embed_dim * 4

        self.final_fusion = nn.Sequential(
            nn.Linear(total_dim, total_dim//2),
            nn.LayerNorm(total_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(total_dim//2, smiles_embed_dim),
            nn.LayerNorm(smiles_embed_dim)
        )
 
    def forward(self, smiles_emb, m_z, adduct, ecfp, mask):

        # smiles_emb   [batch, length, embsize]
        # mask   [batch, length]
        # m/z    [batch]
        # adduct [batch]
        # ecfp   [batch, ecfp_length]
        combined_list = []
        attn_weights = self.smiles_attn(smiles_emb).squeeze(-1)  # [batch, seq_len]
        attn_weights = attn_weights.masked_fill(~mask.bool(), -1e6)
        attn_weights = F.softmax(attn_weights, dim=1)
        smiles_feat = torch.einsum('bsd,bs->bd', smiles_emb, attn_weights)  # [batch, embed_dim]

        # 数值特征处理
        if self.mz_net:
            mz_feat = self.mz_net(m_z.unsqueeze(-1))  # [batch, mz_dim]
            combined_list.append(mz_feat)
        
        # 离子类型特征
        adduct_feat = self.adduct_net(self.adduct_emb(adduct))  # [batch, adduct_dim]
        combined_list.append(adduct_feat)

        # 跨模态注意力（使用数值特征作为query）
        if self.ecfp_net :
             # # ECFP特征
            ecfp_feat = self.ecfp_net(ecfp)  # [batch, ecfp_dim]\
            combined_list.append(ecfp_feat)

        cross_query = torch.cat([data.unsqueeze(1) for data in combined_list], dim=1)  # [batch, 3, embed_dim]
 
        query_mask = torch.ones((cross_query.shape[0], cross_query.shape[1]), device=cross_query.device, dtype=torch.bool)# [batch, 3]
    
        # Convert mask to bool once and reuse
        bool_mask = mask.bool()
        key_mask = FullMask(bool_mask)

        # Optimized attention mask computation
        attention_mask = FullMask(
            torch.einsum('bi,bj->bij', query_mask, bool_mask)  # More efficient than bmm
            .unsqueeze(1)
            .expand(-1, self.heads_num, -1, -1)  # Use expand instead of repeat when possible
        )
        query_mask = FullMask(query_mask)  # Already bool, no need for .bool()

        # 注意力机制增强
        attn_out = self.cross_attn(
            queries=cross_query,
            keys= smiles_emb,
            values= smiles_emb,
            attn_mask = attention_mask,
            query_lengths = query_mask,
            key_lengths = key_mask
        )
        attn_out = attn_out.mean(dim=1)  # [batch, embed_dim]

        enhanced_smiles = smiles_feat + attn_out
        combined_list.append(enhanced_smiles)

        combined = torch.cat(combined_list, dim=1)
        
        return combined ,self.final_fusion(combined)
    
class Aggre_Linear(nn.Module): 

    def __init__(self, smiles_reflect_dim, ecfp_length=1024, num_adducts=2, dropout=0.1):

        super().__init__()    
        print(f"Initializing an instance of {self.__class__.__name__}")

        #1. to `adduct`  'ecfp'  emb refletion
        self.adduct_emb = nn.Embedding(num_adducts, smiles_reflect_dim)


        self.ecfp_emb = nn.Linear(ecfp_length, smiles_reflect_dim)

        #2. Processing continuous variables
        self.mz_fc = nn.Linear(1, smiles_reflect_dim)
        self.dropout = nn.Dropout(dropout)

        #3 concat together
        self.aggregator = nn.Sequential(
            nn.Linear(smiles_reflect_dim*4 , smiles_reflect_dim*2),
            nn.GELU(),
            nn.Linear(smiles_reflect_dim*2 , smiles_reflect_dim),
            nn.GELU(),
            )
        # smiles_reflect_dim 

    def forward(self, smiles_emb, m_z, adduct, ecfp, mask):

        # smiles_emb   [batch, length, embsize]
        # mask   [batch, length]
        # m/z    [batch]
        # adduct [batch]
        # ecfp   [batch, ecfp_length]

        adduct = self.adduct_emb(adduct)
        adduct_emb = adduct.unsqueeze(1).repeat(1, smiles_emb.size(1), 1)

        # Step 2: Process `ecfp`
        ecfp_emb = self.ecfp_emb(ecfp)  # Shape: [batch, smiles_emb_dim]
        ecfp_emb = ecfp_emb.unsqueeze(1).repeat(1, smiles_emb.size(1), 1)  

        # Step 3: Process `m/z`
        mz_emb = self.mz_fc(m_z.unsqueeze(-1))  # Shape: [batch, smiles_emb_dim]
        mz_emb = mz_emb.unsqueeze(1).repeat(1, smiles_emb.size(1), 1)  # Broadcast to [batch, seq_len, emb_dim]

        # 汇聚堆叠在一起
        combined_features = torch.cat((smiles_emb,ecfp_emb,mz_emb,adduct_emb),dim=-1)

        # ecfp_adduct   [batch, ecfp_length + 1+]
        aggregated_output = self.aggregator(combined_features)


        return combined_features,aggregated_output
    
class Aggre_Abandoned(nn.Module):
        
    def __init__(self, smiles_embed_dim, dropout=0.0):
        super().__init__()
        self.desc_skip_connection = True 
        self.fcs = []  # nn.ModuleList()
        print('dropout is {}'.format(dropout))

        self.fc1 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.relu2 = nn.GELU()
        self.final = nn.Linear(smiles_embed_dim, 1)

    def forward(self, smiles_emb):
        x_out = self.fc1(smiles_emb)
        x_out = self.dropout1(x_out)
        x_out = self.relu1(x_out)

        if self.desc_skip_connection is True:
            x_out = x_out + smiles_emb

        z = self.fc2(x_out)
        z = self.dropout2(z)
        z = self.relu2(z)
        if self.desc_skip_connection is True:
            z = self.final(z + x_out)
        else:
            z = self.final(z)

        return z