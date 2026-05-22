"""Inference engine that executes the pre-computed module plan."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


class InferenceEngine:
    def __init__(self, model_dir, prompt, height, width, use_chat_template=True):
        self.model_dir = model_dir
        self.prompt = prompt
        self.height = height
        self.width = width
        self.use_chat_template = use_chat_template
        self.trace = self.load_trace(model_dir.name)
        self.vmem_table = None
        self.curr_memory_usage = None
        self.current_batch_info = None
        self.curr_module_wrapper = None
        self.next_module_wrapper = None
        self.next_module_loader = None

    def load_trace(self, model_name):
        trace_path = LOGS_DIR / f"{model_name}_trace.json"
        with open(trace_path) as f:
            trace = json.load(f)

        self.total_modules = len(trace)

        self.embedding_step = None
        for order_str, entry in trace.items():
            if entry["module_name"] == "model.language_model.embed_token":
                self.embedding_step = int(order_str)
                break

        logger.info("Trace loaded: %d modules, embedding at step %s", self.total_modules, self.embedding_step)
        return trace

    def inference(self):
        input_dict = self.preprocess()
        if input_dict is None:
            return None

        embed_entry = self.trace[str(self.embedding_step)]
        module_name = embed_entry["module_name"]
        from inference.embedding import embed
        inputs_embeds = embed(self.model_dir, module_name, input_dict["input_ids"])

        input_dict["inputs_embeds"] = inputs_embeds
        del input_dict["input_ids"]
        return input_dict
    
    def preprocess(self):
        from inference.preprocess import preprocess
        inputs = preprocess(self.model_dir, self.prompt, self.height, self.width, self.use_chat_template)
        return inputs
    

