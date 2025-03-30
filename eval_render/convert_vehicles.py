import sys
import os
sys.path.append(os.getcwd())

from gaussian_renderer import GaussianModel
from argparse import ArgumentParser
import torch
from glob import glob

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--vehicle_path", type=str)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.vehicle_path, "converted"), exist_ok=True)

    vehicle_files = glob(os.path.join(args.vehicle_path, "*", "gs.pth"))
    for vehicle_file in vehicle_files:
        vehicle_name = os.path.basename(os.path.dirname(vehicle_file))
        print(f"Loading {vehicle_file} ...")
        gaussians = GaussianModel(3, feat_mutable=False)
        (model_params, first_iter) = torch.load(vehicle_file, weights_only=False)
        model_params = list(model_params)
        model_params.append(None)
        gaussians.restore(model_params, None)
        print(f"Saving {vehicle_name} as inria ply and splat format ...")
        gaussians.save_splat(os.path.join(args.vehicle_path, "converted", f"{vehicle_name}.ply"), os.path.join(args.vehicle_path, "converted", f"{vehicle_name}.splat"))