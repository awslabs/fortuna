#!/bin/bash

helpFunction()
{
   echo ""
   echo "Build and push a Docker image. Please pass the following arguments: $0 -a ACCOUNT_ID -r REGION -t TAG"
   echo -e "\t-a: AWS account ID where to create a Docker image."
   echo -e "\t-r: Region where to create a Docker image."
   echo -e "\t-t: Image name tag."
   exit 1 # Exit script after printing help
}

while getopts "a:r:t:" opt
do
   case "$opt" in
      a ) ACCOUNT_ID="$OPTARG" ;;
      r ) REGION="$OPTARG" ;;
      t ) TAG="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$ACCOUNT_ID" ] || [ -z "$REGION" ]
then
   echo "Some or all of the parameters are empty.";
   helpFunction
fi

set -e
set -x

# Define image account ID per region - see https://github.com/aws/deep-learning-containers/blob/master/available_images.md#available-deep-learning-containers-images
if [[ (${REGION:0:2} == "us") || ($REGION == "ap-south-1") || ($REGION == "ap-northeast-1") || ($REGION == "ap-northeast-2") || (${REGION:0:12} == "ap-southeast-1") || ($REGION == "ap-southeast-2") || ($REGION == "ca-central-1") || ($REGION == "eu-central-1") || (${REGION:0:7} == "eu-west") || ($REGION == "eu-north-1") || ($REGION == "sa-east-1") ]]
then
  IMAGE_ACCOUNT_ID=763104351884

elif [[ $REGION == "af-south-1" ]]
then
  IMAGE_ACCOUNT_ID=626614931356

elif [[ $REGION == "ap-east-1" ]]
then
  IMAGE_ACCOUNT_ID=871362719292

elif [[ $REGION == "ap-south-2" ]]
then
  IMAGE_ACCOUNT_ID=772153158452

elif [[ $REGION == "ap-northeast-3" ]]
then
  IMAGE_ACCOUNT_ID=364406365360

elif [[ $REGION == "ap-southeast-3" ]]
then
  IMAGE_ACCOUNT_ID=907027046896

elif [[ $REGION == "ap-southeast-4" ]]
then
  IMAGE_ACCOUNT_ID=457447274322

elif [[ $REGION == "eu-central-2" ]]
then
  IMAGE_ACCOUNT_ID=380420809688

elif [[ $REGION == "eu-south-1" ]]
then
  IMAGE_ACCOUNT_ID=692866216735

elif [[ $REGION == "eu-south-2" ]]
then
  IMAGE_ACCOUNT_ID=503227376785

elif [[ $REGION == "me-south-1" ]]
then
  IMAGE_ACCOUNT_ID=217643126080

elif [[ $REGION == "me-central-1" ]]
then
  IMAGE_ACCOUNT_ID=914824155844

elif [[ ($REGION == "cn-north-1") || ($REGION == "cn-northwest-1") ]]
then
  IMAGE_ACCOUNT_ID=727897471807

else
  echo "The region $REGION is unknown."
  exit 1
fi

IMAGE_NAME="fortuna"

MESSAGE="Creation SUCCEDED."

echo "Creating a repository in ECR... If the message '$MESSAGE' does not appear on screen, your AWS credential were not up-to-date, and the script terminated abruptly. Please update your AWS credentials before running this script."

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "$IMAGE_NAME" > /dev/null 2>&1

echo $MESSAGE

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "$IMAGE_NAME" > /dev/null
fi

ECR_IMAGE="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$IMAGE_NAME"
if [ -n "$TAG" ]
then
    ECR_IMAGE="$ECR_IMAGE:$TAG"
    IMAGE_NAME="$IMAGE_NAME:$TAG"
fi

# circumvent possible error related to docker installation
FILE=$(realpath ~/.docker/config.json)
if [ -f $FILE ]; then
  sed -i "" "/credsStore/d" $FILE
fi

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Get the login command from ECR in order to pull down the SageMaker XGBoost image
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $IMAGE_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

SCRIPT_PATH=$(dirname "$(realpath "$0")")
cd $SCRIPT_PATH

# Build the docker image locally with the image name and then push it to ECR with the full name.
DOCKER_BUILDKIT=1 docker build -t "$IMAGE_NAME" --build-arg ACCOUNT_ID=$ACCOUNT_ID .

docker tag "$IMAGE_NAME" "$ECR_IMAGE"
docker push "$ECR_IMAGE"
