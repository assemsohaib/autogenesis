"""Bound legacy vf-eval worker executors inside the audit container.

verifiers 0.3.1 scales each server and worker executor to 512 threads, independently
of CLI concurrency. That exceeds the whole container's 512 PID limit. Only executor
capacity changes; environment prompts, rewards and rollout behavior stay intact.
"""

try:
    from verifiers.legacy.utils import thread_utils
except ImportError:
    pass  # older releases do not use this worker implementation
else:
    _scale_executors = thread_utils.scale_executors

    def _bounded_scale_executors(concurrency: int) -> int:
        return _scale_executors(min(concurrency, 8))

    thread_utils.scale_executors = _bounded_scale_executors
