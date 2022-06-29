#!/usr/bin/bash
docker run -it --gpus=all --ipc=host -v $(realpath in):/input -v $(realpath out):/output -v $(realpath model):/model mathiser/inference_server_models:CNS_JESKAL_504_DCPTATLAS_merged
