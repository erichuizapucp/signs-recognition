#!/bin/bash

algorithm_name=self-supervised-psl-training-nsdmv3
echo "using algorithm: ${algorithm_name}"

export AWS_PROFILE=authx_dev
chmod +x scripts/sagemaker-train-nsdmv3.sh

account=$(aws sts get-caller-identity --query Account --output text)
echo "using account: ${account}"

region=$(aws configure get region)
region=${region:-us-east-1}
echo "using region: ${region}"

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"
echo "image fullname: ${fullname}"

aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

aws ecr get-login-password --region "${region}" | docker login --username AWS --password-stdin \
  "${account}.dkr.ecr.${region}.amazonaws.com"

docker build  -t ${algorithm_name} --file docker/Dockerfile-sagemaker-train-nsdmv3 .
docker tag ${algorithm_name} ${fullname}

docker push ${fullname}
