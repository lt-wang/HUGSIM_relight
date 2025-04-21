import sys
import os
sys.path.append(os.getcwd())

from argparse import ArgumentParser
import torch
import shutil
from glob import glob
from export_scene import load_and_remove_redundant
import time
from tqdm import tqdm

def export_single_vehicle(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        
    time.sleep(0.1)
    
    # export vehicle ckpt files
    input_ckpt = os.path.join(input_path, f"gs.pth")
    output_ckpt = os.path.join(output_path, f"gs.pth")
    print(f"Exporting {input_ckpt} to {output_ckpt} ...")
    model_param, iteration = load_and_remove_redundant(input_ckpt, load_with_affine=True, out_with_affine=False, scene=False)
    torch.save((model_param, iteration), output_ckpt)
    
    # copy wlh.json
    shutil.copy(os.path.join(input_path, "wlh.json"), os.path.join(output_path, "wlh.json"))
    


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    
    for vehicle_path in tqdm(glob(os.path.join(args.input_path, "2024_*"))):
        vehicle_id = os.path.basename(vehicle_path)
        output_path = os.path.join(args.output_path, vehicle_id)
        export_single_vehicle(vehicle_path, output_path)