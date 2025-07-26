import os
import numpy as np
import torch
import time
from tqdm import tqdm
import argparse
from Feature import getData
from RNArankmodel import RNAranknet1, RNAranknet2

def RNArankpredict(args):
    device = torch.device('cpu')
    torch.set_num_threads(4)
    os.makedirs(args.output_dir, exist_ok=True)

    model_classes = [RNAranknet1, RNAranknet2]
    models = []
    for i, model_cls in enumerate(model_classes, start=1):
        model = model_cls()
        model_path = os.path.join(args.params_file, f'RNArank_model{i}.pth')
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        models.append(model)

    if os.path.isdir(args.pdb):
        pdb_list = [os.path.join(args.pdb, f) for f in os.listdir(args.pdb) if f.endswith('.pdb')]
    else:
        pdb_list = [args.pdb]
    for pdb_path in tqdm(pdb_list, desc="Processing PDB files"):
        pdb_name = os.path.basename(pdb_path).split('.')[0]
        pixels, fea_1d_model1, fea_2d_model1, fea_1d_model2, fea_2d_model2 = getData(pdb_path)
        plDDT_list = []
        with torch.no_grad():
            for i, model in enumerate(models, start=1):
                pixels = torch.Tensor(pixels).to(device)
                fea_1d = torch.Tensor(eval(f'fea_1d_model{i}')).to(device)
                fea_2d = torch.Tensor(eval(f'fea_2d_model{i}')).to(device)
                lddt_pred, deviation_pred, contact_pred, (dev, cont) = model(pixels, fea_1d, fea_2d)
                plDDT_list.append(lddt_pred.cpu().detach().numpy().astype(np.float16))
        plDDT_final = np.average(plDDT_list, axis=0)*100
        txt_file = os.path.join(args.output_dir, f'{pdb_name}_plDDT.txt')
        with open(txt_file, 'w') as f:
            f.write(f"{np.average(plDDT_final):.3f}\n")
            for i in range(plDDT_final.shape[0]):
                f.write(f"{i+1}\t {plDDT_final[i]:.3f}\n")


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser("Predict.py")
    parser.add_argument('--pdb', required=True, help="Path to a PDB file or to a text file containing multiple PDB paths")
    parser.add_argument('--output_dir', default='./example/output/', help="Path to save result, default='./output/'")
    parser.add_argument('--params_file', default='./params/', help="Model path for RNArank")

    args = parser.parse_args()
    RNArankpredict(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time: {elapsed_time:.5f} seconds\n\n")


