# PyTorch DeviceMesh
# Making this work required PyTorch >= 6.0 =
# URL_for_Citation: https://hub.docker.com/layers/pytorch/pytorch/2.6.0-cuda12.6-cudnn9-devel/images/sha256-faa67ebc9c9733bf35b7dae3f8640f5b4560fd7f2e43c72984658d63625e4487
# 
# Installing PyTorch 2.6.0 Docker Image with CUDA 12.6 and cuDNN 9.
# This was necessary because device_mesh was not yet available even
# in the 2.6. minor version nightly builds on Manjaro in conda or pip.
# 
# docker pull pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
# 
# Run the above docker image:
# docker run --gpus all -it pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel /bin/bash
#
# From the docker container's shell, use python to verify the
# presence of device_mesh, the elusive feature.
# docker run --gpus all -it pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel /bin/bash
#  
# Note that even after doing this, I cannot get DeviceMesh. It is only bleeding edge.
#
# URL_for_Citation: https://pytorch.org/tutorials/recipes/distributed_device_mesh.html
# URL_for_Citation: https://pytorch.org/docs/stable/elastic/quickstart.html
#
# Setting up distributed communicators, i.e. NVIDIA Collective Communication Library (NCCL) communicators
# for distributed training can pose a significant challenge.
# For workloads where users need to compose different parallelisms,
# users would need to manually set up and manage NCCL communicators (for example, ProcessGroup)
# for each parallelism solution. This process coudl be complicated and susceptible
# to errors. DeviceMesh can simplify this process, making it more manageable and less prone to errors.
#
#
# What is DeviceMesh?
# DeviceMesh is a higher level abstraction that manages ProcessGroup. It allows users
# to effortlessly create inter-node and intra-node process groups without worrying
# about how to set up ranks correctly for different sub process groups. Users can also
# easily manage the underlying process_groups/devices for multi-dimensional parallelism via DeviceMesh.
#
#
# Why DeviceMesh is Useful
# DeviceMesh is useful when working wiht multi-dimensional parallelism (i.e. 3-D parallel) where parallelism composability is required.
# For example, when your parallelism solutions require both communication across hosts and within each host.
# The image above shows that we can create 2D mesh that connects the devices within each host, and connects
# each device with its counterpart on the other hosts in a homogenous setup.
#
# Without DeviceMesh, users would need to manually set up NCCL communicators, CUDA devices
# on each process before applying any parallelism, which could be quite complicated.
# The following code snippet illustrates a hybrid sharding 2-D Parallel pattern setup
# without DeviceMesh. First, we need to manually calculate the shard group and replicate group.
# Then, we need to assign the correct shard and replicate group to each rank.

import os
import torch
import torch.distributed.algorithms._device_mesh as device_mesh
device_mesh.DeviceMesh()

# Understand World Topology
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
print(f"Running example on {rank=} in a world with {world_size=}")

# Create process groups to manage 2-0 like parallel pattern.
dist.init_process_group("nccl", init_method='env://')
torch.cuda.set_device(rank)


# Create a DeviceMesh object.
device_mesh = DeviceMesh('cuda', mesh_shape=[2, 2])


# Create shard groups (e.g. (0, 1, 2, 3), (4, 5, 6, 7))
# and assign the correct shard group to each rank.
num_node_devices = torch.cuda.device_count()
shard_rank_lists = list(range(0, num_node_devices // 2)), list(range(num_node_devices // 2,
                                                                     num_node_devices))
shard_groups = (
    dist.new_group(shard_rank_lists[0]),
    dist.new_group(shard_rank_lists[1]),
)
current_shard_group = (
    shard_groups[0] if rank in shard_rank_lists[0] else shard_groups[1]
)

# Create replicate groups (for example, (0, 4), (1, 5), (2, 6), (3, 7))
# and assign the correct replicate group to each rank.
current_replicate_group = None
shard_factor = len(shard_rank_lists[0])
for i in range(num_node_devices // 2):
    replicate_group_ranks = list(range(i, num_node_devices, shard_factor))
    replicate_group = dist.new_group(replicate_group_ranks)
    if rank in replicate_group_ranks:
        current_replicate_group = replicate_group


# Clean up after performing computations on hte device mesh.
dist.destroy_process_group()



# Running the above code requires PyTorch Elastic.
# Let's create a file named 2d_setup.py and then run the torchrun command:
# torchrun --nproc_per_node=8 --rdzv_id=100 -rdzv_endpoint=localhost:29400 2d_setup.py
# torchrun --nproc_per_node=8 --rdzv_id=100 -rdzv_endpoint=localhost:29400,0,1,2,3,4,5,6,7 tuts14device_mesh.py
# torchrun --nproc_per_node=8 tuts14device_mesh.py


# How to use DeviceMesh with HSDP.
# Hybrid Sharding Data Parallel (HSDP) is a 2D strategy to perform
# FSDP within a host and DDP across hosts.
#
# Let's see an example of how DeviceMesh can assist with applying
# HSDP to your model with a simple setup.
# With DeviceMesh, users would not need to manually create and
# manage shard group and replicate group.
#
#
import torch
import torch.nn as nn

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))
    

# HSDP: MeshShape(2, 4)
mesh_2d = init_device_mesh("cuda", (2, 4))
model = FSDP(
    ToyModel(), device_mesh=mesh_2d, sharding_strategy=ShardingStrategy.HYBRID_SHARD
)

