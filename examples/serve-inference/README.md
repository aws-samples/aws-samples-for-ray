This example compiles Open LLAMA-3B model and deploys the model on Trainium (Trn1)  instance
using Ray Serve and FastAPI. It uses transformers-neuronx to shard the model across devices/neuron cores
via Tensor parallelism. 


## Setup 
* Clone this repo to run the example on your local laptop:
```
git clone https://github.com/aws-samples/aws-samples-for-ray

cd aws-samples-for-ray/examples/serve-inference
```

* Replace subnet and security-group where you intend to launch the cluster in `cluster-inference-serve.yaml`
```
sed -i 's/subnet-replace-me/subnet-ID/g' cluster-inference-serve.yaml
sed -i 's/sg-replace-me/sg-ID/g' cluster-inference-serve.yaml
```

* Start your Ray cluster from your local laptop (pre-requisite of Ray installation):
```
ray up cluster-inference-serve.yaml
```

* Deploy the model using serve
```
ray exec cluster-inference-serve.yaml \
'source aws_neuron_venv_pytorch/bin/activate && cd neuron_demo && serve run aws_neuron_core_inference_serve:entrypoint --runtime-env-json="{\"env_vars\":{\"NEURON_CC_FLAGS\": \"--model-type=transformer-inference\",\"FI_EFA_FORK_SAFE\":\"1\"}}"' \
--tmux
```

* Wait for the serve deployment to complete (typically takes ~5minutes)
```
ray exec cluster-inference-serve.yaml 'source aws_neuron_venv_pytorch/bin/activate && serve status'
```
Sample expected output of serve status
```
proxies:
  proxy-uuid: HEALTHY
applications:
  default:
    status: RUNNING
    message: ''
    last_deployed_time_s: 1694558868.7500472
    deployments:
      LlamaModel:
        status: HEALTHY
        replica_states:
          RUNNING: 1
        message: ''
      APIIngress:
        status: HEALTHY
        replica_states:
          RUNNING: 1
        message: ''
```


## Usage
Attach to the head node of the Ray cluster
```
ray attach cluster-inference-serve.yaml
```

Navigate to the python interpreter on head node
```
source aws_neuron_venv_pytorch/bin/activate && python
```

Start using the model
```
import requests

response = requests.get(f"http://127.0.0.1:8000/infer?sentence=AWS is super cool")
print(response.status_code, response.json())
```

## Demo with gradio
The demo file gradio_ray_serve.py integrates Llama2 with Gradio app on Ray Serve. Llama 2 inference is deployed through Gradio app on Ray Serve so it can process and respond to HTTP requests.
```
source aws_neuron_venv_pytorch/bin/activate
pip install gradio
serve run gradio_ray_serve:app --runtime-env-json='{"env_vars":{"NEURON_CC_FLAGS": "--model-type=transformer-inference", "FI_EFA_FORK_SAFE":"1"}}'
``` 

## Teardown
To teardown the cluster/resources
```
ray down cluster-inference-serve.yaml -y
```

