"""GPU-free lifecycle regression for the pinned graph-capture implementation.

Execute the real ModelCudaGraphManager.capture body with small stand-ins for
CUDA and model state. This checks hook placement, not CUDA semaphore semantics;
the latter also needs the two-shape GPU/full-model integration test.
"""
from __future__ import annotations

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
GPU = ROOT / "runtime/vllm-overlay/v1/worker/gpu"


class Buffer:
    def __getitem__(self, key):
        return self

    def __setitem__(self, key, value):
        pass

    def fill_(self, value):
        pass


class CaptureHookTests(unittest.TestCase):
    def exercise(self, hooked: bool):
        events = []
        ready = False
        mode = SimpleNamespace(NONE="none", FULL="full", PIECEWISE="piecewise")

        class Base:
            def capture(self, create_forward_fn, progress_bar_desc):
                for tokens in (8, 4):
                    desc = SimpleNamespace(num_tokens=tokens, num_reqs=tokens // 4,
                                           num_active_loras=0, cg_mode=mode.FULL,
                                           max_query_len=4)
                    # Match the base manager: fresh inputs for warmup and capture.
                    for warmup in (True, False):
                        forward = create_forward_fn(desc, warmup)
                        forward(mode.NONE)

        def prepare(*args, **kwargs):
            events.append(("prepare", args[1]))
            return {}, {}

        def signal(tokens):
            nonlocal ready
            ready = True
            events.append(("signal", tokens))

        def model(**kwargs):
            nonlocal ready
            if hooked:
                self.assertTrue(ready, "forward consumed a PLE flag that was not re-armed")
                ready = False
            events.append(("forward", None))
            return Buffer()

        tree = ast.parse((GPU / "cudagraph_utils.py").read_text())
        manager_class = next(node for node in tree.body
                             if isinstance(node, ast.ClassDef)
                             and node.name == "ModelCudaGraphManager")
        manager_class.body = [node for node in manager_class.body
                              if isinstance(node, ast.FunctionDef) and node.name == "capture"]
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
            ast.alias(name="annotations")], level=0), manager_class], type_ignores=[])
        namespace = {
            "CudaGraphManager": Base,
            "CUDAGraphMode": mode,
            "prepare_inputs_to_capture": prepare,
            "set_forward_context": lambda *a, **kw: nullcontext(),
            "torch": SimpleNamespace(empty_like=lambda x: Buffer()),
        }
        exec(compile(ast.fix_missing_locations(module), str(GPU / "cudagraph_utils.py"), "exec"), namespace)
        manager = namespace["ModelCudaGraphManager"]()
        manager.__dict__.update(use_breakable_cg=False, max_num_reqs=2, dp_size=1,
                                is_first_pp_rank=True, is_last_pp_rank=True,
                                vllm_config=None, hidden_states=None, aux_hidden_states=[])
        buffers = SimpleNamespace(input_ids=Buffer(), positions=Buffer(), is_padding=Buffer())
        state = SimpleNamespace(prepare_dummy_inputs=lambda *a: {})
        manager.capture(model, state, buffers, None, None, [], None,
                        pre_forward_hook=signal if hooked else None)
        return events

    def test_each_shape_warmup_and_capture_rearms_ple(self):
        events = self.exercise(hooked=True)
        self.assertEqual([value for name, value in events if name == "signal"], [8, 8, 4, 4])
        for index, event in enumerate(events):
            if event[0] == "forward":
                self.assertEqual(events[index - 1][0], "signal")

    def test_models_without_ple_do_not_require_a_hook(self):
        events = self.exercise(hooked=False)
        self.assertEqual(sum(name == "forward" for name, _ in events), 4)
        self.assertFalse(any(name == "signal" for name, _ in events))

    def test_runner_connects_ple_hook_without_affecting_non_ple_models(self):
        tree = ast.parse((GPU / "model_runner.py").read_text())
        hooks = [keyword.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "capture"
                 for keyword in node.keywords if keyword.arg == "pre_forward_hook"]
        self.assertEqual(len(hooks), 1)
        expression = compile(ast.Expression(hooks[0]), "<pre_forward_hook>", "eval")
        signal = object()
        connector = SimpleNamespace(signal_dummy_outputs=signal)
        self.assertIs(eval(expression, {"self": SimpleNamespace(_ple_offload_connector=connector)}), signal)
        self.assertIsNone(eval(expression, {"self": SimpleNamespace(_ple_offload_connector=None)}))


if __name__ == "__main__":
    unittest.main()
