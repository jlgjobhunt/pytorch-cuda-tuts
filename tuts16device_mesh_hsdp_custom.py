# How to use DeviceMesh for your custom parallel solutions.
#
# URL_for_Citation: https://pytorch.org/tutorials/recipes/distributed_device_mesh.html#how-to-use-devicemesh-for-your-custom-parallel-solutions
#
# When working with large scale training, you might have more complex custom parallel
# training composition. For example, you may need to slice out submeshes for different
# parallelism solutions. DeviceMesh allows users to slice child mesh from the parent
# mesh and re-use the NCCL communicators already created when the parent mesh is
# initialized.
#
#
from torch.distributed.device_mesh import init_device_mesh
mesh_3d = init_device_mesh("cuda", (2, 2, 2), mesh_dim_names=("replicate", "shard", "tp"))


# Users can slice child meshes from the parent mesh.
hsdp_mesh = mesh_3d["replicate", "shard"]
tp_mesh = mesh_3d["tp"]


# Users can access the underlying process group thru 'get_group' API.
replicate_group = hsdp_mesh.get_group(mesh_dim="replicate")
tp_group = tp_mesh.get_group()


