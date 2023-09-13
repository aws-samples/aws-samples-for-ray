from fastapi import FastAPI  # noqa
from ray import serve  # noqa

import torch # noqa
import os
from transformers import AutoTokenizer # noqa

app = FastAPI()

hf_model = "openlm-research/open_llama_3b"
local_model_path = "open_llama_3b_split"


@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, llama_model_handle) -> None:
        self.handle = llama_model_handle

    @app.get("/infer")
    async def infer(self, sentence: str):
        ref = await self.handle.infer.remote(sentence)
        result = await ref
        return result


@serve.deployment(
    ray_actor_options={"resources": {"neuron_cores": 32},
                       "runtime_env": {"env_vars": {"NEURON_CC_FLAGS": "--model-type=transformer-inference"}}},
    autoscaling_config={"min_replicas": 1, "max_replicas": 1},
)
class LlamaModel:
    def __init__(self):
        import torch # noqa
        from transformers import AutoTokenizer # noqa
        from transformers_neuronx.llama.model import LlamaForSampling # noqa
        from transformers import LlamaForCausalLM # noqa
        from transformers_neuronx.module import save_pretrained_split # noqa

        if not os.path.exists(local_model_path):
            print(f"Saving model split for {hf_model} to local path {local_model_path}")
            self.model = LlamaForCausalLM.from_pretrained(hf_model)
            save_pretrained_split(self.model, local_model_path)
        else:
            print(f"Using existing model split {local_model_path}")

        print(f"Loading and compiling model {local_model_path} for Neuron")
        self.neuron_model = LlamaForSampling.from_pretrained(local_model_path, batch_size=1, tp_degree=32, amp='f16')
        self.neuron_model.to_neuron()
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)

    def infer(self, sentence: str):
        input_ids = self.tokenizer.encode(sentence, return_tensors="pt")
        with torch.inference_mode():
            generated_sequences = self.neuron_model.sample(input_ids, sequence_length=2048, top_k=50)
        return [self.tokenizer.decode(seq) for seq in generated_sequences]


entrypoint = APIIngress.bind(LlamaModel.bind())