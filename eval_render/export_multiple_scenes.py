import sys
import os

from omegaconf import OmegaConf
sys.path.append(os.getcwd())

from argparse import ArgumentParser, Namespace
import torch
import shutil
from glob import glob
from export_scene import export_single_scene
import json
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--scenes_info", type=str, default="/nas/users/hyzhou/PAMI2024/release/ss/index.json")
    parser.add_argument("--output_path", type=str, default="/nas/users/hyzhou/PAMI2024/release/ss")
    parser.add_argument("--iteration", type=int, default=30_000)
    parser.add_argument("--ver0", action="store_true")
    args = parser.parse_args()

    with open(args.scenes_info, "r") as f:
        scenes_info = json.load(f)
    for _, scene in scenes_info.items():
        scene_path = scene["scene_path"]
        scenarios = scene["scenario_path"]
        
        # export scene
        datasets_name = ["nuscenes", "waymo", "kitti360", "pandaset"]
        dataset = next(item for item in datasets_name if item in scene_path)
        scene_name = os.path.basename(scene_path)
        if dataset == "nuscenes":
            scene_name = scene_name.split("_")[0]
        output_scene_path = os.path.join(args.output_path, 'scenes', dataset, scene_name)
        
        if args.ver0:
            cfg_args_path = os.path.join(scene_path, "cfg_args")
            with open(cfg_args_path, "r") as f:
                cfg = eval(f.read())
            cfg_dict = vars(cfg).copy()
            save_cfg = {
                "model_path": output_scene_path,
                "source_path": cfg_dict["source_path"],
                "affine": True,
                "data_type": dataset,
                "model": {
                    "sh_degree": cfg_dict["sh_degree"],
                    "data_device": cfg_dict["data_device"],
                    "white_background": cfg_dict["white_background"],
                }
            }
            save_cfg = OmegaConf.create(save_cfg)
            OmegaConf.save(save_cfg, os.path.join(scene_path, "cfg.yaml"))
        
        export_single_scene(scene_path, output_scene_path, args.iteration, args.ver0)
        
        # copy scenarios
        for scenario_path in scenarios:
            os.makedirs(os.path.join(args.output_path, 'scenarios', dataset), exist_ok=True)
            scenario_cfg = OmegaConf.load(scenario_path)
            for actors in scenario_cfg.plan_list:
                actors[-3] = actors[-3].replace("/postprocess/shadow.pth", "")
            OmegaConf.save(scenario_cfg, os.path.join(args.output_path, 'scenarios', dataset, os.path.basename(scenario_path)))