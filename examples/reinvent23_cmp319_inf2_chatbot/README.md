# Deploying a basic Llama2 chatbot with AWS Inferentia and Ray Serve

This example shows how to prepare and deploy a Llama2-7B model using [AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/) accelerators and [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) model hosting. The [transformers-neuronx library](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/transformers-neuronx/index.html) is used to provide tensor parallelism, which allows large language models (LLMs) like Llama2 to be sharded across multiple Inferentia NeuronCores to provide accelerated inference. The example also shows how to enable autoscaling of the Ray Serve application so that the deployments scale up and down based on user demand. The chatbot's web interface is created using the popular [Gradio](https://www.gradio.app/) package. 

## Prerequisites

Please ensure that you have a recent version of Python installed, along with the latest version of [Ray](https://docs.ray.io/en/latest/ray-overview/installation.html) and the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html). In most cases you can simply run `pip3 install -U ray[default]` to install Ray.

## Step 1: Infrastructure Setup 

* Clone this repo to run the example on your local environment:

```bash
git clone https://github.com/aws-samples/aws-samples-for-ray
cd aws-samples-for-ray/examples/serve-inference
```

* Modify `1_cluster-inference-serve.yaml` and replace ${AMI_ID}, ${SUBNET_ID}, and ${SECURITYGROUP_ID} with the desired AMI ID, subnet ID, and security group ID that will be used to deploy your Ray cluster. If you are unsure of the AMI ID, you can run the following command to determine the latest Ubuntu-based Neuron deep learning AMI (DLAMI): 

```
aws ec2 describe-images --region us-west-2 --owners amazon \
--filters 'Name=name,Values=Deep Learning AMI Neuron PyTorch 1.13 (Ubuntu 20.04) ????????' 'Name=state,Values=available' \
--query 'reverse(sort_by(Images, &CreationDate))[:1].ImageId' --output text
```

1. Start your Ray cluster from your local laptop (pre-requisite of Ray installation):

```bash
ray up 1_cluster-inference-serve.yaml
```

You will be prompted to confirm the cluster creation as follows. Input `y` and proceed.

```console
Cluster: inference-serve

2023-11-27 15:37:17,639 INFO util.py:375 -- setting max workers for head node type to 0
Loaded cached provider configuration
If you experience issues with the cloud provider, try re-running the command with --no-config-cache.
AWS config
IAM Profile: ray-autoscaler-v1 [default]
EC2 Key pair (all available node types): ray-autoscaler_us-west-2 [default]
VPC Subnets (all available node types): subnet-0a910e572266c13bd [default]
EC2 Security groups (ray.head.default): sg-0581f8300b3b2455e [default]
EC2 Security groups (ray.worker.default): sg-0294e801896f7b828
EC2 AMI (all available node types): ami-0396c2a8448f872d2

No head node found. Launching a new cluster. Confirm [y/N]: 
```




2. Log in to the head node.
Once cluster is launched, you can login to the head node with the following command.

```bash
ray attach -p 8000 1_cluster-inference-serve.yaml
```

You will see terminal of the head node as follows.

```console
2023-11-27 15:24:55,387 INFO util.py:375 -- setting max workers for head node type to 0
Loaded cached provider configuration
If you experience issues with the cloud provider, try re-running the command with --no-config-cache.
Fetched IP: 44.242.165.202
Warning: Permanently added '44.242.165.202' (ED25519) to the list of known hosts.
ubuntu@ip-10-0-95-229:~$ 
```

Once log into the node mode the current working directory to `neuron_demo`.

```bash
cd ~/neuron_demo
source /opt/aws_neuron_venv_pytorch/bin/activate
```

The rest of the contents assume that you are working on the node and the directory. Also make sure that the virtual environment is used.

## Step 2: Deploy Llama2 with Ray Serve

Now that we have a Ray cluster with a head node and Inf2 instances, let's deploy the Llama2 model on the infrastructure. The example script `2_aws_neuron_core_inference_serve.py` creates a basic Ray Serve deployment to host the Llama2 model and respond to user queries using plain HTTP requests. In a later section we will build on this example to provide a web interface.

We can deploy `app` defined in the script as follows.

```bash
serve run 2_aws_neuron_core_inference_serve:app
```

It will show output similar to the following:

```console
2023-11-28 00:00:17,426 INFO scripts.py:471 -- Running import path: '2_aws_neuron_core_inference_serve:app'.
...
2023-11-28 00:31:31,561 SUCC scripts.py:519 -- Deployed Serve app successfully
```

You can then submit requests to the model via HTTP requests using tools such as curl. To submit an HTTP request to your model, first open a new terminal and connect to your head node:
```
ray attach 1_cluster-inference-serve.yaml 
```

Then run the following curl command to send an HTTP request to the model:

```
curl http://127.0.0.1:8000?sentence=write%20a%20poem%20about%20singing%20cats
```

Alternatively, you can submit requests through Python on head node

```
python
```

Start using the model

```
import requests

response = requests.get(f"http://127.0.0.1:8000/infer?sentence=AWS is super cool")
print(response.status_code, response.json())
```

When you are finished testing your model, please press `CONTROL-C` in your first terminal to stop the Ray Serve application.


## Step 3: Auto-scale your deployment

In the previous step, you have deployed Llama2 model with basic configuration. You only have one inference server, which might be insufficient depends on your service requirement. In this step, you will deploy `3_aws_neuron_core_inference_serve_autoscaling.py` which comes with inference service auto scaling based on the incoming requests.

```bash
serve run 3_aws_neuron_core_inference_serve_autoscaling:app
```

Wait for the serve deployment to complete. You can check the progress with `serve` command on the head-node. 

```bash
watch serve status
```

Sample expected output of serve status after the deployment completion is shown below.

```console
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

Similarly, on the head-node, you can also check the node deployment status with the following command. 

```bash
watch ray status
```

You will see an output similar to below, showing that we have 1 head-node and 1 worker-node.

```console
Every 2.0s: ray status         ip-10-0-80-251: Tue Nov 28 06:16:08 2023

======== Autoscaler status: 2023-11-28 06:16:07.392656 ========
Node status
---------------------------------------------------------------
Healthy:
 1 ray.worker.default
 1 ray.head.default
Pending:
 (no pending nodes)
Recent failures:
 (no failures)

Resources
---------------------------------------------------------------
Usage:
 2.0/96.0 CPU
 0B/248.72GiB memory
 12.0/12.0 neuron_cores
 44B/110.59GiB object_store_memory

Demands:
 (no resource demands)
```

Now let's try to put load onto this service. You will use a cli tool called [hey](https://github.com/rakyll/hey) for the purpose.
Run the following command on the head node while keep other three windows keep opening.

```bash
hey -c 1 -q 0.1 -n 3 http://127.0.0.1:8000/infer?sentence=AWS
```

This command will send three requests with no concurrency.  After a while, hey will print stats like below.

```console
Summary:
  Total:        32.3237 secs
  Slowest:      2.3361 secs
  Fastest:      2.2983 secs
  Average:      2.3189 secs
  Requests/sec: 0.0928
  

Response time histogram:
  2.298 [1]     |■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
  2.302 [0]     |
  2.306 [0]     |
  2.310 [0]     |
  2.313 [0]     |
  2.317 [0]     |
  2.321 [0]     |
  2.325 [1]     |■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
  2.329 [0]     |
  2.332 [0]     |
  2.336 [1]     |■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■


Latency distribution:
  10% in 2.3222 secs
  25% in 2.3361 secs
  0% in 0.0000 secs
  0% in 0.0000 secs
  0% in 0.0000 secs
  0% in 0.0000 secs
  0% in 0.0000 secs

Details (average, fastest, slowest):
  DNS+dialup:   0.0002 secs, 2.2983 secs, 2.3361 secs
  DNS-lookup:   0.0000 secs, 0.0000 secs, 0.0000 secs
  req write:    0.0001 secs, 0.0000 secs, 0.0002 secs
  resp wait:    2.3184 secs, 2.2979 secs, 2.3355 secs
  resp read:    0.0001 secs, 0.0001 secs, 0.0001 secs

Status code distribution:
  [200] 3 responses
```

Notice that number of replica for our deployment has not changed in this case. Let's try to put more pressure on it.

```bash
hey -c 10 -q 1 -n 1000 http://127.0.0.1:8000/infer?sentence=AWS
```

Notice that after a while `serve` starts creating more replicas.

```console
proxies:
  4fd8f56025ddf134866ff112508c01438deac85df8a3cf9b8dab265d: HEALTHY
applications:
  default:
    status: RUNNING
    message: ''
    last_deployed_time_s: 1701151334.9220335
    deployments:
      LlamaModel:
        status: HEALTHY
        replica_states:
          STARTING: 3
          RUNNING: 1
        message: ''
      APIIngress:
        status: HEALTHY
        replica_states:
          RUNNING: 4
        message: ''
```

In parallel you can see that additional worker nodes are coming up.

```bash
Healthy:
 1 ray.worker.default
 1 ray.head.default
Pending:
 10.0.85.241: ray.worker.default, waiting-for-ssh
 10.0.74.58: ray.worker.default, waiting-for-ssh
Recent failures:
 (no failures)
```

Those `Pending` instances are eventually transition to `Healthy` state.

```console
======== Autoscaler status: 2023-11-28 07:07:19.867270 ========
Node status
---------------------------------------------------------------
Healthy:
 3 ray.worker.default
 1 ray.head.default
Pending:
 (no pending nodes)
Recent failures:
 (no failures)
```


## Step 4: Launch the chatbot using Gradio
The demo file `4_aws_neuron_core_inference_serve__gradio.py` integrates the Llama2-7B-chat model with a Gradio application hosted via Ray Serve. The Gradio application allows the user to submit prompts to the model, and displays the text that is generated in response to the prompts.

To launch the demo, run the following commands on the head-node:
 
```
cd ~/neuron_demo
source /opt/aws_neuron_venv_pytorch/bin/activate
serve run 4_aws_neuron_core_inference_serve__gradio:app 
``` 

When the Ray Serve application launches, you can then access the Gradio web interface by browsing to [http://127.0.0.1:8000](http://127.0.0.1:8000) on your local machine. If you are unable to access this URL on your local machine, please make sure that you have used the `-p 8000` option when attaching to your head-node, ex: `ray attach -p 8000 1_cluster-inference-serve.yaml`.

## Teardown

To teardown the cluster and associated resources, please run the `ray down` command and reference the cluster configuration file as follows:

```
ray down -y 1_cluster-inference-serve.yaml
```
