"""Build a tensor dependency graph from the trace and generate a save/delete plan."""

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TensorNode:
    __slots__ = ("tid", "ptr", "shape", "dtype", "produced_at", "consumers", "free")

    def __init__(self, tid, ptr, shape, dtype, produced_at):
        self.tid = tid
        self.ptr = ptr
        self.shape = shape
        self.dtype = dtype
        self.produced_at = produced_at
        self.consumers = []
        self.free = []

    def to_dict(self):
        immediate = False
        persist = False
        if self.consumers:
            first = min(self.consumers)
            immediate = first == self.produced_at + 1
            # persist if consumers aren't all consecutive (there's a gap)
            span = max(self.consumers) - first
            persist = span > len(self.consumers) - 1

        return {
            "shape": self.shape,
            "dtype": self.dtype,
            "produced_at": self.produced_at,
            "consumers": self.consumers,
            "immediate": immediate,
            "persist": persist,
            "free": self.free,
        }


def _build_tensor_graph(trace):
    """Walk the trace in execution order and create TensorNodes for every unique tensor."""
    active = {}       # ptr (Python id) -> most recent TensorNode
    nodes = []        # all TensorNodes, indexed by tid
    next_tid = 0

    for order_str, entry in sorted(trace.items(), key=lambda x: int(x[0])):
        order = int(order_str)

        for t in entry["inputs"]:
            ptr = t["id"]
            if ptr not in active:
                node = TensorNode(next_tid, ptr, t["shape"], t["dtype"], 0)
                nodes.append(node)
                active[ptr] = node
                next_tid += 1
                logger.debug("External tensor %d (ptr=%d) %s %s", node.tid, ptr, t["shape"], t["dtype"])
            active[ptr].consumers.append(order)
            t["id"] = active[ptr].tid

        for t in entry["outputs"]:
            ptr = t["id"]
            node = TensorNode(next_tid, ptr, t["shape"], t["dtype"], order)
            nodes.append(node)
            active[ptr] = node
            t["id"] = node.tid
            next_tid += 1

    return nodes


def _compute_free_lists(nodes, trace):
    """For each tensor, which input tensors can be freed once it's created."""
    # Map each step to its input tensor IDs
    step_inputs = {}
    for order_str, entry in trace.items():
        step_inputs[int(order_str)] = [t["id"] for t in entry["inputs"]]

    # Map each tensor to the last step that consumes it
    last_consumer = {}
    for node in nodes:
        if node.consumers:
            last_consumer[node.tid] = max(node.consumers)

    # All outputs of the same step share the same free list
    step_free = {}
    for step_str in trace:
        step = int(step_str)
        inputs = step_inputs.get(step, [])
        step_free[step] = [tid for tid in inputs if last_consumer.get(tid) == step]

    for node in nodes:
        node.free = step_free.get(node.produced_at, [])


def _build_schedule(nodes, trace):
    """For each execution step, list which tensor IDs can be freed after it."""
    schedule = {order_str: {"free_after": []} for order_str in trace}

    for node in nodes:
        if node.consumers:
            key = str(max(node.consumers))
            if key in schedule:
                schedule[key]["free_after"].append(node.tid)

    return schedule


def run(model_name):
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    trace_path = logs_dir / f"{model_name}_trace.json"

    with open(trace_path) as f:
        trace = json.load(f)

    nodes = _build_tensor_graph(trace)
    _compute_free_lists(nodes, trace)
    logger.info("Built %d tensor nodes from %d trace entries", len(nodes), len(trace))

    schedule = _build_schedule(nodes, trace)

    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2)
    logger.info("Updated trace with static tensor IDs -> %s", trace_path)

    plan = {
        "tensors": {str(n.tid): n.to_dict() for n in nodes},
        "schedule": schedule,
    }
    save_path = logs_dir / f"{model_name}.saveTensors"
    with open(save_path, "w") as f:
        json.dump(plan, f, indent=2)
    logger.info("Saved tensor plan (%d nodes) -> %s", len(nodes), save_path)


def main():
    parser = argparse.ArgumentParser(description="Build tensor graph and save/delete plan from trace")
    parser.add_argument("model_name", help="Model folder name under models/")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    run(args.model_name)


if __name__ == "__main__":
    main()
