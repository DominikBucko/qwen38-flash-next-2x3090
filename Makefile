IMAGE ?= qwen38-flash-next-2x3090:locked
PYTHON ?= python3

ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: validate release-check build-image preflight serve test-exact-qsa

validate:
	$(PYTHON) scripts/validate_repo.py

release-check: validate
	$(PYTHON) scripts/check_release_ready.py

build-image: validate
	docker build --pull -f docker/Dockerfile -t $(IMAGE) .

preflight:
	./scripts/preflight.sh

serve: build-image preflight
	IMAGE=$(IMAGE) ./scripts/docker_serve.sh

test-exact-qsa: build-image
	docker run --rm --entrypoint python $(IMAGE) /opt/qwen38/tests/test_qsa_exact_topk_cpu.py
