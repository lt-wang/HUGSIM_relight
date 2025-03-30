import sys
import os
sys.path.append(os.getcwd())

from gaussian_renderer import GaussianModel
from argparse import ArgumentParser
import torch

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--iteration", type=int, default=30_000)
    args = parser.parse_args()

    print(f"Loading {args.model_path} {args.iteration} checkpoints ...")

    gaussians = GaussianModel(3, affine=True)
    (model_params, first_iter) = torch.load(os.path.join(args.model_path, "ckpts", f"chkpnt{args.iteration}.pth"), weights_only=False)
    gaussians.restore(model_params, None)

    print(f"Saving semantic pcd to {args.model_path}/vis ...")
    os.makedirs(os.path.join(args.model_path, "vis"), exist_ok=True)
    gaussians.save_semantic_pcd(os.path.join(args.model_path, "vis", "semantic.ply"))
    
    print(f"Saving inria ply and splat to {args.model_path}/vis ...")
    gaussians.save_splat(os.path.join(args.model_path, "vis", "points.ply"), os.path.join(args.model_path, "vis", "scene.splat"))