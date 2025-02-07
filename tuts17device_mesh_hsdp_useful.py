# Meta.ai (URL_for_Citation: https://www.meta.ai/c/9dd9413b-4b53-4a11-9617-ae8dc86c050b)
# Code demonstrating the HSDP approach to device_mesh and provides more informative
# terminal output.
# Set environmental variables first:
# export TORCHELASTIC_ERROR_HANDLING=FULL
# torchrun --nproc_per_node=8 --log_dir ./output tuts17device_mesh_hsdp_useful.py
#
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel


def main():
    # Initialize the process group.
    dist.init_process_group(backend="nccl", init_method="env://")

    # Get the global rank and world size.
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create a 3D device mesh.
    mesh_3d = DeviceMesh(device_type="cuda", mesh=(2, 2, 2), mesh_dim_names=["replicate", "shard", "tp"])

    # Slice the 3D mesh into sub-meshes.
    # hsdp_mesh = mesh_3d.get_submesh(["replicate", "shard"])
    hsdp_mesh = DeviceMesh(
        device_type="cuda",
        mesh=[2, 2],
        mesh_dim_names=["replicate", "shard", "tp"],
    )
    
    # Get the process groups for each sub-mesh.
    replicate_dim = mesh_3d.get_dim(mesh_dim_names="replicate")
    shard_dim = mesh_3d.get_dim(mesh_dim_names="shard")
    tp_dim = mesh_3d.get_group()

    # Print information about the process groups.
    with open(f"output_{dist.get_rank()}.txt", "w") as f:
        f.write(f"Global Rank: {global_rank}, World Size: {world_size}\n")
        f.write(f"Replicate dimension: {replicate_dim}\n")
        f.write(f"Shard dimension: {shard_dim}\n")
        f.write(f"TP dimension: {tp_dim}]\n")

    # Clean up resources.
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
