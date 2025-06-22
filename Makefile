UV_VERSION ?= 0.7.13

define GPUENVVARS
# Hydra debug
export HYDRA_FULL_ERROR=1

# Set environment variables for GPU
export CUDA_HOME="/usr/local/cuda"
export CUDA_VERSION="12.5.1"
export CUDA_MAJOR_VERSION="12"
export CUDA_MINOR_VERSION="5"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64/stubs"
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs"

export NVARCH="x86_64"

export NVIDIA_VISIBLE_DEVICES="all"
export NVIDIA_DRIVER_CAPABILITIES="compute,utility"

export NV_CUDA_CUDART_VERSION="12.5.82-1"
export NV_CUDA_CUDART_DEV_VERSION="12.5.82-1"
export NV_CUDA_LIB_VERSION="12.5.1-1"
export NV_CUDA_NSIGHT_COMPUTE_VERSION="12.5.1-1"
export NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE="cuda-nsight-compute-12-5=12.5.1-1"
export NV_NVTX_VERSION="12.5.82-1"
export NV_NVPROF_VERSION="12.5.82-1"
export NV_NVPROF_DEV_PACKAGE="cuda-nvprof-12-5=12.5.82-1"

# cuDNN
export NV_CUDNN_VERSION="9.2.1.18-1"
export NV_CUDNN_PACKAGE="libcudnn9-cuda-12=9.2.1.18-1"
export NV_CUDNN_PACKAGE_DEV="libcudnn9-dev-cuda-12=9.2.1.18-1"

# cuBLAS
export NV_LIBCUBLAS_VERSION="12.5.3.2-1"
export NV_LIBCUBLAS_PACKAGE="libcublas-12-5=12.5.3.2-1"
export NV_LIBCUBLAS_PACKAGE_NAME="libcublas-12-5"
export NV_LIBCUBLAS_DEV_VERSION="12.5.3.2-1"
export NV_LIBCUBLAS_DEV_PACKAGE="libcublas-dev-12-5=12.5.3.2-1"
export NV_LIBCUBLAS_DEV_PACKAGE_NAME="libcublas-dev-12-5"

# NCCL (for multi-GPU)
export NCCL_VERSION="2.22.3-1"
export NV_LIBNCCL_PACKAGE="libnccl2=2.22.3-1+cuda12.5"
export NV_LIBNCCL_PACKAGE_NAME="libnccl2"
export NV_LIBNCCL_PACKAGE_VERSION="2.22.3-1"
export NV_LIBNCCL_DEV_PACKAGE="libnccl-dev=2.22.3-1+cuda12.5"
export NV_LIBNCCL_DEV_PACKAGE_NAME="libnccl-dev"
export NV_LIBNCCL_DEV_PACKAGE_VERSION="2.22.3-1"

endef

define TPUENVVARS
# Unset environment variables that are not needed
for var in MASTER_ADDR MASTER_PORT TPU_PROCESS_ADDRESSES XRT_TPU_CONFIG; do
    unset $var
done

# Hydra debug
export HYDRA_FULL_ERROR=1

# Set environment variables for TPU
export ISTPUVM=1
export PJRT_DEVICE=TPU
export PT_XLA_DEBUG_LEVEL=1
export TF_CPP_MIN_LOG_LEVEL=2
export TPU_ACCELERATOR_TYPE=v3-8
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_HOST_BOUNDS=1,1,1
export TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434
export TPU_SKIP_MDS_QUERY=1
export TPU_WORKER_HOSTNAMES=localhost
export TPU_WORKER_ID=0
export XLA_TENSOR_ALLOCATOR_MAXSIZE=100000000

endef

export GPUENVVARS
export TPUENVVARS

.PHONY: tpusetup gpusetup uv

tpusetup: tpuenvs remove-tf uv
gpusetup: gpuenvs remove-tf uv

uv:
	@echo '# uv setup\\n' >> ~/.bashrc
	@curl -LsSf https://astral.sh/uv/$(UV_VERSION)/install.sh | sh
	@echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
	@echo 'eval "$(uvx --generate-shell-completion bash)"' >> ~/.bashrc

remove-tf:
	@pip uninstall tensorflow tensorflow-tpu tensorboard -y

tpuenvs:
	@echo "$$TPUENVVARS" >> ~/.bashrc

gpuenvs:
	@echo "$$GPUENVVARS" >> ~/.bashrc
