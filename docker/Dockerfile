FROM vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8

RUN python3 -m pip install --no-cache-dir humming-kernels==0.1.12

COPY runtime/install_overlay.py /opt/qwen38/runtime/install_overlay.py
COPY runtime/vllm-overlay /opt/qwen38/runtime/vllm-overlay
RUN python3 /opt/qwen38/runtime/install_overlay.py /opt/qwen38/runtime/vllm-overlay

COPY configs/2x3090-128gb.env /opt/qwen38/configs/2x3090-128gb.env
COPY configs/static_hot_cache_rankings.json /workspace/static_hot_cache_rankings.json
COPY scripts/serve-container.sh /usr/local/bin/qwen38-serve
COPY tests/test_qsa_exact_topk_cpu.py /opt/qwen38/tests/test_qsa_exact_topk_cpu.py
RUN chmod 0755 /usr/local/bin/qwen38-serve

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/qwen38-serve"]
