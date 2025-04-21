import sys
import os
sys.path.append(os.getcwd())

from gaussian_renderer import GaussianModel
from argparse import ArgumentParser
import torch
from omegaconf import OmegaConf

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()

    cfg = OmegaConf.load(os.path.join(args.model_path, "cfg.yaml"))
    print(f"Loading {args.model_path} checkpoints ...")

    gaussians = GaussianModel(cfg.model.sh_degree, affine=True)
    (model_params, first_iter) = torch.load(os.path.join(args.model_path, "scene.pth"), weights_only=False)
    gaussians.restore(model_params, None)

    print(f"Saving semantic pcd to {args.model_path}/vis ...")
    os.makedirs(os.path.join(args.model_path, "vis"), exist_ok=True)
    gaussians.save_semantic_pcd(os.path.join(args.model_path, "vis", "semantic.ply"))
    
    print(f"Saving inria ply and splat to {args.model_path}/vis ...")
    gaussians.save_splat(os.path.join(args.model_path, "vis", "points.ply"), os.path.join(args.model_path, "vis", "scene.splat"))