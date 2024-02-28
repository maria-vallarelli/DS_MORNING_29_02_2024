# Databricks notebook source
# Databricks notebook source

# COMMAND ----------

# MAGIC %pip install --upgrade mlflow emoji==0.6.0 nltk 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook logs a HuggingFace model with an input example and a model signature and registers it to the Databricks Model Registry.
# MAGIC
# MAGIC After you run this notebook in its entirety, you have a registered model for model serving with Databricks Model Serving 

# COMMAND ----------

# %sh 
# pwd


# COMMAND ----------

import transformers
import mlflow
import torch
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from torch.utils.data import DataLoader
import os
import pandas as pd
from emoji import demojize
from nltk.tokenize import TweetTokenizer

# COMMAND ----------

project_path = "/Users/MIO_NOME_UTENTE/Experiments/bertweet"
mlflow.set_experiment(project_path)
#TODO RICORDARSI DI DARE I PERMESSI

# COMMAND ----------

cache_dir = "/Users/maria.vallarelli@eniplenitude.com/Experiments/cachedir"
if not os.path.exists(cache_dir):
  os.makedirs(cache_dir)
tokenizer = transformers.BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True, cache_dir=cache_dir)

# COMMAND ----------

#VOGLIO REGISTRARE IL MODELLO IN UC
mlflow.set_registry_uri("databricks-uc")
uc_schema_name = 'p_sbx_data.data'
pytorch_model_name = f'{uc_schema_name}.bert_encoder_pytorch'
pyfunc_model_name = f'{uc_schema_name}.bert_encoder'

model_cache = "/Volumes/p_sbx_data/data/data/UC_assets/MIA_DIRECTORY_VOLUME/model_cache"


# COMMAND ----------

if not os.path.exists(model_cache):
  os.makedirs(model_cache)

# COMMAND ----------

class AugmentedBert(torch.nn.Module):
    def __init__(self, output_class_len, base_model, cache_dir, hidden_dim=64):
        super().__init__()
        self.bert_model = transformers.AutoModel.from_pretrained(base_model, cache_dir=cache_dir)
        self.emb_dim = 768
        self.fc1 = torch.nn.Linear(self.emb_dim, self.emb_dim)
        self.tanh = torch.nn.Tanh()
        self.gelu = torch.nn.GELU()

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
        )
        output = bert_output["last_hidden_state"][:, 0, :]
        output = self.fc1(output)
        output = self.tanh(output)
        output = self.gelu(output)
        return output


# COMMAND ----------

with mlflow.start_run():
    model = AugmentedBert(10, "vinai/bertweet-base", model_cache)
    X = tokenizer(
        ["This is a test"],
        max_length=64,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    y = model(X.input_ids, X.attention_mask)
    signature = mlflow.models.infer_signature(
        {
            "input_ids": X.input_ids.detach().numpy(),
            "attention_mask": X.attention_mask.detach().numpy(),
        },
        y.detach().numpy(),
    )

    mlflow.pytorch.log_model(
        model,
        "pytorch-model",
        signature=signature,
        registered_model_name=pytorch_model_name,
    )

# COMMAND ----------

class SampleDatasetWithEncodings(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]).clone().detach()
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def create_data_loader(tokenizer, X, y=None, batch_size=1, input_max_len=64):
    features = tokenizer(
        X,
        max_length=input_max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    if y is not None:
        dataset = SampleDatasetWithEncodings(features, y)
    else:
        dataset = SampleDatasetWithEncodings(features, [0] * features.get("input_ids").shape[0])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

# COMMAND ----------

model_name = pytorch_model_name
model_uri = f"models:/{model_name}/1"
model_artifact = '/Volumes/p_sbx_data/data/data/UC_assets/MIA_FOLDER/pytorch-model-artifacts'
if not os.path.exists(model_artifact):
  os.makedirs(model_artifact)
local_path = ModelsArtifactRepository(model_uri).download_artifacts("", dst_path="/Volumes/p_sbx_data/data/data/UC_assets/MIA_FOLDER/pytorch-model-artifacts") # download model from remote registry
print(local_path)

# COMMAND ----------

# %sh ls /Volumes/p_sbx_data/data/data/UC_assets/model_tp_extract/pytorch-model-artifacts

# COMMAND ----------

import os
import mlflow
import torch
import pandas as pd
import transformers


class ModelPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = torch.load(context.artifacts["torch-weights"])
        self.tokenizer = transformers.BertweetTokenizer.from_pretrained(
            "vinai/bertweet-base",
            normalization=True,
            local_files_only=True,
            cache_dir=context.artifacts["tokenizer_cache"],
        )

    def format_inputs(self, model_input):
        if isinstance(model_input, str):
            model_input = [model_input]
        if isinstance(model_input, pd.Series):
            model_input = model_input.tolist()
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.iloc[:, 0].tolist()
        return model_input

    def prepare_data(self, tokenizer, model_input):
        data_loader = create_data_loader(tokenizer, model_input)
        return data_loader.dataset.encodings

    def format_outputs(self, outputs):
        predictions = (torch.sigmoid(outputs)).data.numpy()
        classes = [
            "Technology",
            "Environment",
            "Politics",
            "Entertainment",
            "Environment",
            "Health & Wellness"
        ]
        return pd.DataFrame(
            [dict(zip(classes, prediction)) for prediction in predictions]
        )

    def predict(self, context, model_input):
        model_input = self.format_inputs(model_input)
        processed_input = self.prepare_data(self.tokenizer, model_input)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(
                input_ids=processed_input.get("input_ids"),
                attention_mask=processed_input.get("attention_mask"),
            )
        return self.format_outputs(outputs)

with mlflow.start_run() as run:
    model = ModelPyfunc()
    signature = mlflow.models.infer_signature(
        ["Is virtual reality the future of online gaming? Many companies seem to be investing heavily. #VR #Gaming", "The recent forest fires have highlighted the urgent need for climate action. #ClimateChange #Environment", "New policy changes promise to tackle income inequality. Will they be effective?","Music festivals are making a comeback post-pandemic. Can't wait to attend one! #MusicLife #Festivals","Mental health awareness is crucial, especially in these challenging times. #MentalHealthMatters #SelfCare"],
        pd.DataFrame(
            [
                {
                    "Technology": 0.78803562,
                    "Environment": 0.1839863,
                    "Politics": 0.180215,
                    "Entertainment": 0.18803562,
                    "Environment": 0.1839863,
                    "Health & Wellness": 0.180215
                },
                {
                    "Technology": 0.38803562,
                    "Environment": 0.6839863,
                    "Politics": 0.480215,
                    "Entertainment": 0.38803562,
                    "Environment": 0.6839863,
                    "Health & Wellness": 0.480215
                },
                {
                    "Technology": 0.18803562,
                    "Environment": 0.1839863,
                    "Politics": 0.780215,
                    "Entertainment": 0.18803562,
                    "Environment": 0.1839863,
                    "Health & Wellness": 0.180215
                },
                {
                    "Technology": 0.38803562,
                    "Environment": 0.6839863,
                    "Politics": 0.180215,
                    "Entertainment": 0.78803562,
                    "Environment": 0.3839863,
                    "Health & Wellness": 0.480215
                },
                {
                    "Technology": 0.18803562,
                    "Environment": 0.1839863,
                    "Politics": 0.180215,
                    "Entertainment": 0.18803562,
                    "Environment": 0.7839863,
                    "Health & Wellness": 0.180215
                },
                {
                    "Technology": 0.38803562,
                    "Environment": 0.1839863,
                    "Politics": 0.280215,
                    "Entertainment": 0.28803562,
                    "Environment": 0.3839863,
                    "Health & Wellness": 0.880215
                },
            ]
        ),
    )
    mlflow.pyfunc.log_model(
        "model",
        python_model=model,
        artifacts={
            "torch-weights": "/Volumes/p_sbx_data/data/data/UC_assets/MIA_FOLDER/pytorch-model-artifacts/data/model.pth",
            "tokenizer_cache": cache_dir,
        },
        input_example=["Exploring the capabilities of AI in healthcare has shown promising results. #ArtificialIntelligence #HealthTech", "The recent forest fires have highlighted the urgent need for climate action. #ClimateChange #Environment"],
        signature=signature,
        registered_model_name=pyfunc_model_name,
    )

# COMMAND ----------

tokenizer = TweetTokenizer()

def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalizeTweet(tweet):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())


# if __name__ == "__main__":
#     print(
#         normalizeTweet(
#             "SC has first two presumptive cases of coronavirus, DHEC confirms https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user-share… via @postandcourier"
#         )
#     )

# COMMAND ----------

# Array di stringhe (tweets) da normalizzare
tweets = [
    "LA QUESTIONE DEI COSTI POCO CHIARI 😕",
    "Incredibile come la tecnologia avanza rapidamente! #tech",
    "@user Non vedo l'ora che inizi il concerto! 🎶",
    "Il tempo oggi è meraviglioso ☀️",
    "DHEC confirms https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user-share… via @postandcourier 😢"
]

# Funzione per normalizzare un array di stringhe
def normalize_tweets(tweets):
    normalized_tweets = []
    for tweet in tweets:
        normalized_tweet = normalizeTweet(tweet)
        normalized_tweets.append(normalized_tweet)
    return normalized_tweets

# Applica la funzione di normalizzazione
normalized_tweets = normalize_tweets(tweets)

# Stampa i tweets normalizzati
for i, tweet in enumerate(normalized_tweets, 1):
    print(f"Tweet {i}: {tweet}")


# COMMAND ----------

import torch
from transformers import AutoTokenizer
import mlflow
#logged_model = 'runs:/f2160612160a42118dfdb48abcf856e0/model'

logged_model = 'runs:/MIA_RUN_ID/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
data = {
    'text': normalized_tweets
}
loaded_model.predict(pd.DataFrame(data))

# COMMAND ----------


