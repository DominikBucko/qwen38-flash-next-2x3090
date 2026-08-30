# Third-party notices

This source repository does not contain model weights.

- The runtime overlay is derived from vLLM and is distributed under the Apache
  License 2.0; see [`LICENSE`](LICENSE).
- The assembled model references Qwen/Qwen3.8-Flash-Next and is governed by the
  Qwen Community License supplied with the upstream checkpoint.
- Intel's AutoRound checkpoint and RadixArk's NVFP4 checkpoint remain governed
  by their respective model cards, notices, and inherited Qwen terms.
- Humming kernels, PyTorch, CUDA, Docker, and Hugging Face tooling are external
  dependencies and are not redistributed by this source repository.

The checkpoint assembly script copies the upstream model license into the
Hugging Face upload tree. Users are responsible for reviewing the current
upstream terms before publishing or using an assembled checkpoint.
