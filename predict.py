import torch
import yaml
from argparse import Namespace
import numpy as np
from data_prepare.data_loader import PropertyPredictionDataModule
from data_prepare.tokenizer import MolTranBertTokenizer
from model.layers.main_layer import LightningModule
from fast_transformers.masking import LengthMask as LM
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
from data_prepare.data_loader import ADDUCT2IDX
import argparse

IDX2ADDUCT = {v: k for k, v in ADDUCT2IDX.items()}

def prepare_data(model_name):
     
     with open(f'Pretrained/hparams/{model_name}.yaml', 'r') as f:
        config = Namespace(**yaml.safe_load(f))
        
        # prepare data:
        data_module = PropertyPredictionDataModule(config)
        data_module.test_setup()
        data_module.prepare_data()

        # data loader
        test_loader = data_module.val_dataloader()

        return test_loader[0]

def prepare_model(model_name):
        
        with open(f'Pretrained/hparams/{model_name}.yaml', 'r') as f:
        
            config = Namespace(**yaml.safe_load(f))

            tokenizer = MolTranBertTokenizer('bert_vocab.txt')
            ckpt = f'Pretrained/checkpoints/{model_name}.ckpt'
            if model_name == 'in-house':
                ckpt = f'Pretrained/checkpoints/AllCCS2.ckpt'

            model = LightningModule(config, tokenizer).load_from_checkpoint(ckpt, strict=False,config=config, tokenizer=tokenizer,vocab=len(tokenizer.vocab))

            # Check for GPU availability
            device = torch.device('cuda')
            model = model.to(device)  # Move model to GPU if available
            model.eval()
            
            return model

def predict(batch,model):
    with torch.no_grad():

        idx = batch[0]# idx
        mask = batch[1]# mask
        m_z = batch[2] # m/z
        adduct = batch[3] # adduct
        ecfp = batch[4] # ecfp
        targets = batch[-1] # ccs
        device = "cuda"
        idx, mask, m_z, adduct, ecfp,targets = [x.to(device) for x in batch]
        # First, move idx back to CPU if it's on GPU

        idx_cpu = idx.cpu()
        # Decode each sequence in the batch
        smiles_list = []
        for seq in idx_cpu:
            # Convert indices to tokens (automatically handles special tokens)
            tokens = model.tokenizer.convert_ids_to_tokens(seq)
            # Convert tokens to SMILES string
            smiles = model.tokenizer.convert_tokens_to_string(tokens)
            smiles_list.append(smiles)

        token_embeddings = model.tok_emb(idx) # each index maps to a (learnable) vector
        x = model.drop(token_embeddings)
        x = model.blocks(x, length_mask=LM(mask.sum(-1)))

        if 'early' in model.hparams.type :

            if 'hyperccs' in model.hparams.type:
                _ ,x = model.aggre(x, m_z, adduct, ecfp, mask)

            input_mask_expanded = mask.unsqueeze(-1).expand(x.size()).float()
            # input_mask_expanded : [batch, seq_len, emb_dim]
            masked_embedding = x * input_mask_expanded
            sum_embeddings = torch.sum(masked_embedding, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-7)
            loss_input = sum_embeddings / sum_mask

        elif model.hparams.type == 'later':
            _ ,loss_input = model.aggre(x, m_z, adduct, ecfp, mask)

        pred, actual = model.get_pred(loss_input, targets)
        
    return pred.cpu().detach().numpy().flatten(),actual.cpu().detach().numpy(),smiles_list,adduct.cpu().detach().numpy().flatten()

def main(model_name):
    model = prepare_model(model_name)
    model.eval()
    test_dataloader = prepare_data(model_name)

    pre_output = []
    truth = []
    smiles = []
    adducts = []

    for batch in test_dataloader:
        pred, actual, smile, adduct = predict(batch,model)
        smiles.extend(smile)
        adducts.extend(adduct)
        pre_output.extend(pred)
        truth.extend(actual)

    y_hat = np.stack(pre_output)
    y = np.stack(truth)
    adducts = [IDX2ADDUCT.get(adduct) for adduct in adducts]
    
    # Print shapes to verify
    print("Shape of y:", y.shape)  # Should be (number of elements)
    print("Shape of y_hat:", y_hat.shape)  
    print("Shape of adducts:", len(adducts))  
    print("Shape of smiles:", len(smiles))  

    # Calculate R² score
    r2 = r2_score(y, y_hat)
    print(f"R² score: {r2:.4f}")
    
    # Save results to CSV
    results = pd.DataFrame({
        'true_ccs': y,
        'predicted_ccs': y_hat,
        'smiles': smiles,
        'adducts': adducts,
    })
    results.to_csv(f'predictions_{model_name}.csv', index=False)
    print(f"Predictions saved to predictions_{model_name}.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CCS prediction with specified model.')
    parser.add_argument('--dataset_name', type=str, default='AllCCS2', 
                       help='Name of the model to use (default: AllCCS2)')
    
    args = parser.parse_args()

    main(args.dataset_name)