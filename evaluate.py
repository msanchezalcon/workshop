import pickle
import watson_machine_learning_client

# The Deployment ID of your model
deployment_uid =    "xxxxxxxxxxxxx"

# Credentials to your Watson Machine Learning service instance
wml_credentials = {
    "url":          "xxxxxxxxxxxxx",
    "username":     "xxxxxxxxxxxxx",
    "password":     "xxxxxxxxxxxxx",
    "instance_id":  "xxxxxxxxxxxxx"
}
client = watson_machine_learning_client.WatsonMachineLearningAPIClient(wml_credentials)

# Metadata describing your deployed model
deployment_details = client.repository.get_details(deployment_uid)

# Scoring endpoint URL that is used for making scoring requests
scoring_url = client.deployments.get_scoring_url(deployment_details)

# Convert to correct format
data = open("evaluation_data.pickle", "rb")
image = pickle.load(data)[0][3].reshape(1,32,32,1).tolist()
scoring_payload = {"values": image}

# Score the image
scoring_response = client.deployments.score(scoring_url, scoring_payload)
print(scoring_response)
