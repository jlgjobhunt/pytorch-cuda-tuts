# Meta.ai (URL_for_Citation: https://www.meta.ai/c/9dd9413b-4b53-4a11-9617-ae8dc86c050b)
# Code demonstrating the HSDP approach to device_mesh and provides more informative
# terminal output.
# Enable environmental variables before running to debug:
# export TORCHELASTIC_ERROR_HANDLING=FULL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# torchrun --nproc_per_node=8 --log_dir ./output Last.tuts17device_mesh.py



import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_process_group(backend="nccl", init_method="env://"):
    try:
        dist.init_process_group(backend=backend, init_method=init_method)
        logger.info("Process group initialized")
    except Exception as e:
        logger.error(f"Error initializing process group: {e}")
        exit(1)


def create_device_mesh(device_type="cuda", mesh=[1, 1], mesh_dim_names=["replicate", "shard"]):
    try:
        if device_type == "cuda" and not torch.cuda.is_available():
            logger.error("CUDA devices are not available")
            exit(1)

        if len(mesh) != 2:
            logger.error("Invalid mesh shape. Expected a 2D mesh")
            exit(1)

        device_mesh = DeviceMesh(
            device_type=device_type,
            mesh=mesh,
            mesh_dim_names=mesh_dim_names,
        )
        logger.info("Device mesh created")
        return device_mesh
    except Exception as e:
        logger.error(f"Error creating device mesh: {e}")
        dist.destroy_process_group()
        torch.cuda.empty_cache()
        exit(1)


def main():
    init_process_group()

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    logger.info(f"Global Rank: {global_rank}, World Size: {world_size}")

    try:
        device_mesh = create_device_mesh(mesh=[2, 2], mesh_dim_names=["row", "col"])
    except Exception as e:
        logger.error(f"Error creating device mesh: {e}")
        dist.destroy_process_group()
        torch.cuda.empty_cache()
        exit(1)

    mesh_shape = device_mesh.mesh
    mesh_dim_names = device_mesh.mesh_dim_names

    logger.info(f"Mesh Shape: {mesh_shape}")
    logger.info(f"Mesh Dimension Names: {mesh_dim_names}")

    try:
        # Add your code here
        pass
    except Exception as e:
        logger.error(f"Error: {e}")
        dist.destroy_process_group()
        torch.cuda.empty_cache()
        exit(1)
    else:
        dist.destroy_process_group()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()