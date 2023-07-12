# End-to-End Machine Learning using Ray on SageMaker

This workshop goes through a common customer example when beginning and maturing their MLOps journey from Initial to Scalable. 

![Ray on AWS](images/open-source-Ray-AWS-integrations.png)

This workshop will introduce the use of [Ray](https://docs.ray.io/en/latest/ray-overview/getting-started.html) on Sagemaker. Ray is an open-source distributed computing framework designed to accelerate and simplify the development of scalable and efficient machine learning applications. It provides a simple and flexible API for defining and executing [tasks](https://docs.ray.io/en/latest/ray-core/tasks.html) and [actors](https://docs.ray.io/en/latest/ray-core/actors.html) on a cluster of machines, allowing you to easily scale your machine learning workloads from a single machine to thousands of nodes.

Ray is designed to [support a wide range of machine learning libraries and frameworks](https://docs.ray.io/en/latest/train/getting-started.html), including popular tools like TensorFlow, PyTorch, and scikit-learn. It also includes built-in support for distributed reinforcement learning, hyperparameter tuning, and model serving, making it easy to build end-to-end machine learning pipelines.

One of the key features of Ray is its ability to handle both task parallelism and actor-based concurrency, allowing you to easily express complex computation patterns and data dependencies. Ray also provides efficient scheduling and data sharing mechanisms, enabling fast and scalable execution of machine learning workloads.

Ray is widely used in both industry and academia, and has a large and active community of contributors and users. Whether you’re building a large-scale machine learning system or just getting started with distributed computing, Ray provides a powerful and easy-to-use platform for building and running your machine learning applications.

## Lab Introduction

We suggest use `Data Science 3.0` and `Python 3` kernel image, instance type `ml.t3.medium` for SageMaker Studio notebook while running this workshop.

[Lab 1](1-data-prep-feature-store-ray.ipynb): Prepare data using Ray in SageMaker Processing Job and ingest data into SageMaker Feature Store

[Lab 2](2-training-registry-ray.ipynb): Use Ray to train and tune a XgBoost model on SageMaker and regiter the best model to SageMaker Model Registry

[Lab 3](3-deployment.ipynb): Deploy trained XgBoost model as a SageMaker real-time endpoint

[Lab 4](4-sagemaker-pipeline-ray.ipynb): Build a MLOps pipeline using SageMaker Pipelines


## Additional Resources

* Check this [Blog Post](https://aws.amazon.com/blogs/machine-learning/mlops-foundation-roadmap-for-enterprises-with-amazon-sagemaker/)
for further details on the MLOps Foundations Roadmap for Enterprises on Amazon SageMaker.

* [SageMaker MLOps Custom Project Templates GitHub Repository](https://github.com/aws-samples/sagemaker-custom-project-templates): This repository contains example projects templates based on common customer patterns. 
