import sys
import os

from omegaconf import OmegaConf
sys.path.append(os.getcwd())

from argparse import ArgumentParser, Namespace
import torch
import shutil
from glob import glob
import time

def load_and_remove_redundant(ckpt_path, load_with_affine=True, out_with_affine=True, scene=True):
    """
    Default params for ver0 scenes
    """
    
    # load
    (model_params, iteration) = torch.load(ckpt_path)
    
    (
        active_sh_degree, 
        xyz, 
        features_dc, 
        features_rest,
        feats3D,
        scaling, 
        rotation, 
        opacity,
        max_radii, grad_accum, denom, optim, # unused
        spatial_lr_scale,
    ) = model_params[:13]
    
    if load_with_affine:
        appearance_model = model_params[13]
        
    if scene:
        ground_model_params = model_params[14]

    # simplify and save
    simplified_params = [
        active_sh_degree, 
        xyz, 
        features_dc, 
        features_rest,
        feats3D,
        scaling, 
        rotation, 
        opacity,
        spatial_lr_scale,
    ]

    if out_with_affine:
        simplified_params.append(appearance_model.state_dict())
    
    if scene:
        simplified_params.append(ground_model_params)
    
    return simplified_params, iteration


def export_single_scene(input_path, output_path, iteration, from_ver0):
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    time.sleep(0.1)
    
    # copy scene ckpt files
    scene_ckpt_file = os.path.join(input_path, "ckpts", f"chkpnt{iteration}.pth")
    output_scene_ckpt_file = os.path.join(output_path, "scene.pth")
    print(f"Exporting {scene_ckpt_file} to {output_scene_ckpt_file} ...")
    if from_ver0:
        model_param, iteration = load_and_remove_redundant(scene_ckpt_file)
        torch.save((model_param, iteration), output_scene_ckpt_file)
    else:
        try:
            shutil.copy(scene_ckpt_file, output_scene_ckpt_file)
        except shutil.SameFileError:
            pass
        
    # copy dynamic files
    for dynamic_ckpt_file in glob(os.path.join(input_path, "ckpts", f"dynamic_*_chkpnt{iteration}.pth")):
        dynamic_id = os.path.basename(dynamic_ckpt_file).split("_")[1]
        output_dynamic_ckpt_file = os.path.join(output_path, f"dynamic_{dynamic_id}.pth")
        print(f"Exporting {dynamic_ckpt_file} to {output_dynamic_ckpt_file} ...")
        if from_ver0:
            model_param, iteration = load_and_remove_redundant(dynamic_ckpt_file, load_with_affine=True, out_with_affine=False, scene=False)
            torch.save((model_param, iteration), output_dynamic_ckpt_file)
        else:
            try:
                shutil.copy(dynamic_ckpt_file, output_dynamic_ckpt_file)
            except shutil.SameFileError:
                pass
    
    # copy unicycle model
    for unicycle_ckpt_file in glob(os.path.join(input_path, "ckpts", f"unicycle_*.pth")):
        output_unicycle_ckpt_file = os.path.join(output_path, os.path.basename(unicycle_ckpt_file))
        print(f"Exporting {unicycle_ckpt_file} to {output_unicycle_ckpt_file} ...")
        try:
            shutil.copy(unicycle_ckpt_file, output_unicycle_ckpt_file)
        except shutil.SameFileError:
            pass
        
    # copy config yaml
    model_config = OmegaConf.load(os.path.join(input_path, 'cfg.yaml'))
    model_config.model_path = output_path
    OmegaConf.save(model_config, os.path.join(output_path, "cfg.yaml"))
    
    # copy ground info
    try:
        shutil.copy(os.path.join(input_path, "ground_param.pkl"), os.path.join(output_path, "ground_param.pkl"))
    except shutil.SameFileError:
        pass
    
    # copy metadata
    try:
        shutil.copy(os.path.join(input_path, "meta_data.json"), os.path.join(output_path, "meta_data.json"))
    except shutil.SameFileError:
        pass
    
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--iteration", type=int, default=30_000)
    parser.add_argument("--ver0", action="store_true")
    args = parser.parse_args()

    export_single_scene(args.model_path, args.output_path, args.iteration, args.ver0)