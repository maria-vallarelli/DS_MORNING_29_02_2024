# Databricks notebook source
# Databricks notebook source

# COMMAND ----------

from datasets import load_dataset
from transformers import pipeline

# COMMAND ----------

import os
cache_dir = "/Shared/MIA_FOLDER"
os.environ["HF_DATASETS_CACHE"] = cache_dir

# COMMAND ----------

xsum_dataset = load_dataset(
    "xsum", version="1.2.0", cache_dir=cache_dir
)  # Note: We specify cache_dir to use pre-cached data.
xsum_sample = xsum_dataset["train"].select(range(10))
display(xsum_sample.to_pandas())

# COMMAND ----------

# MAGIC %md ### Create a Hugging Face pipeline

# COMMAND ----------

from transformers import pipeline


# hf_model_name = "t5-small": Specifica il nome del modello pre-addestrato da utilizzare, in questo caso, "t5-small", che è una versione più piccola e meno costosa in termini computazionali del modello T5 (Text-to-Text Transfer Transformer) progettato per compiti di comprensione e generazione del testo.
# min_length = 20 e max_length = 40: Definiscono la lunghezza minima e massima del sommario generato.
# truncation = True: Indica che il testo di input dovrebbe essere troncato se supera la lunghezza massima gestita dal modello.
# do_sample = True: Abilita il campionamento durante la generazione del testo, consentendo di produrre output più vari e meno deterministici.


# Later, we plan to log all of these parameters to MLflow.
# Storing them as variables here will help with that.
hf_model_name = "t5-small"
min_length = 20
max_length = 40
truncation = True
do_sample = True

summarizer = pipeline(
    task="summarization",
    model=hf_model_name,
    min_length=min_length,
    max_length=max_length,
    truncation=truncation,
    do_sample=do_sample,
    model_kwargs={"cache_dir": cache_dir},
)  # Note: We specify cache_dir to use pre-cached models.

# COMMAND ----------

doc0 = xsum_sample["document"][0]
print(f"Summary: {summarizer(doc0)[0]['summary_text']}")
print("===============================================")
print(f"Original Document: {doc0}")

# COMMAND ----------

# MAGIC %md ### Track LLM development with MLflow
# MAGIC
# MAGIC [MLflow](https://mlflow.org/) has a Tracking component that helps you to track exactly how models or pipelines are produced during development.  Although we are not fitting (tuning or training) a model here, we can still make use of tracking to:
# MAGIC * Track example queries and responses to the LLM pipeline, for later review or analysis
# MAGIC * Store the model as an [MLflow Model flavor](https://mlflow.org/docs/latest/models.html#built-in-model-flavors), thus packaging it for simpler deployment

# COMMAND ----------

# Apply to a batch of articles
import pandas as pd

results = summarizer(xsum_sample["document"])
display(pd.DataFrame(results, columns=["summary_text"]))

# COMMAND ----------

# MAGIC %md [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html) is organized hierarchically as follows:
# MAGIC * **An [experiment](https://mlflow.org/docs/latest/tracking.html#organizing-runs-in-experiments)** generally corresponds to the creation of 1 primary model or pipeline.  In our case, this is our LLM pipeline.  It contains some number of *runs*.
# MAGIC    * **A [run](https://mlflow.org/docs/latest/tracking.html#organizing-runs-in-experiments)** generally corresponds to the creation of 1 sub-model, such as 1 trial during hyperparameter tuning in traditional ML.  In our case, executing this notebook once will only create 1 run, but a second execution of the notebook will create a second run.  This version tracking can be useful during iterative development.  Each run contains some number of logged parameters, metrics, tags, models, artifacts, and other metadata.
# MAGIC       * **A [parameter](https://mlflow.org/docs/latest/tracking.html#concepts)** is an input to the model or pipeline, such as a regularization parameter in traditional ML or `max_length` for our LLM pipeline.
# MAGIC       * **A [metric](https://mlflow.org/docs/latest/tracking.html#concepts)** is an output of evaluation, such as accuracy or loss.
# MAGIC       * **An [artifact](https://mlflow.org/docs/latest/tracking.html#concepts)** is an arbitrary file stored alongside a run's metadata, such as the serialized model itself.
# MAGIC       * **A [flavor](https://mlflow.org/docs/latest/models.html#storage-format)** is an MLflow format for serializing models.  This format uses the underlying ML library's format (such as PyTorch, TensorFlow, Hugging Face, or your custom format), plus metadata.
# MAGIC
# MAGIC MLflow has an API for tracking queries and predictions [`mlflow.llm.log_predictions()`](https://mlflow.org/docs/latest/python_api/mlflow.llm.html), which we will use below.  Note that, as of MLflow 2.3.1 (Apr 28, 2023), this API is Experimental, so it may change in later releases.  See the [LLM Tracking page](https://mlflow.org/docs/latest/llm-tracking.html) for more information.
# MAGIC
# MAGIC ***Tip***: We wrap our model development workflow with a call to `with mlflow.start_run():`.  This context manager syntax starts and ends the MLflow run explicitly, which is a best practice for code which may be moved to production.  See the [API doc](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run) for more information.

# COMMAND ----------

# Questo approccio alla registrazione di esperimenti e modelli offre diversi vantaggi:

# Tracciabilità: Permette di tracciare sistematicamente tutti gli aspetti dell'esperimento, facilitando l'analisi dei risultati e la riproducibilità degli esperimenti.
# Riusabilità: Salvando il modello con la sua configurazione, altri possono facilmente ricaricare e utilizzare il modello per inferenze future o per ulteriori sviluppi.
# Deployment: La registrazione del modello con firma e esempio di input rende più semplice il suo deployment, ad esempio, in servizi cloud o in applicazioni di produzione.

# COMMAND ----------

import mlflow

# Tell MLflow Tracking to use this explicit experiment path,
# which is located on the left hand sidebar under Machine Learning -> Experiments 
experiment_name = "/Users/maria.vallarelli@eniplenitude.com/Experiments/summarization"
mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    # LOG PARAMS
    mlflow.log_params(
        {
            "hf_model_name": hf_model_name,
            "min_length": min_length,
            "max_length": max_length,
            "truncation": truncation,
            "do_sample": do_sample,
        }
    )

    # --------------------------------
    # LOG INPUTS (QUERIES) AND OUTPUTS
    # Logged `inputs` are expected to be a list of str, or a list of str->str dicts.
    results_list = [r["summary_text"] for r in results]

    # Our LLM pipeline does not have prompts separate from inputs, so we do not log any prompts.
    mlflow.llm.log_predictions(
        inputs=xsum_sample["document"],
        outputs=results_list,
        prompts=["" for _ in results_list],
    )

    # ---------
    # LOG MODEL
    # We next log our LLM pipeline as an MLflow model.
    # This packages the model with useful metadata, such as the library versions used to create it.
    # This metadata makes it much easier to deploy the model downstream.
    # Under the hood, the model format is simply the ML library's native format (Hugging Face for us), plus metadata.

    # It is valuable to log a "signature" with the model telling MLflow the input and output schema for the model.
    signature = mlflow.models.infer_signature(
        xsum_sample["document"][0],
        mlflow.transformers.generate_signature_output(
            summarizer, xsum_sample["document"][0]
        ),
    )
    print(f"Signature:\n{signature}\n")

    # For mlflow.transformers, if there are inference-time configurations,
    # those need to be saved specially in the log_model call (below).
    # This ensures that the pipeline will use these same configurations when re-loaded.
    inference_config = {
        "min_length": min_length,
        "max_length": max_length,
        "truncation": truncation,
        "do_sample": do_sample,
    }

    # Logging a model returns a handle `model_info` to the model metadata in the tracking server.
    # This `model_info` will be useful later in the notebook to retrieve the logged model.
    model_info = mlflow.transformers.log_model(
        transformers_model=summarizer,
        artifact_path="summarizer",
        task="summarization",
        inference_config=inference_config,
        signature=signature,
        input_example="This is an example of a long news article which this pipeline can summarize for you.",
    )

# COMMAND ----------

loaded_summarizer = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
loaded_summarizer.predict(xsum_sample["document"][0])

# COMMAND ----------

results = loaded_summarizer.predict(xsum_sample.to_pandas()["document"])
display(pd.DataFrame(results, columns=["generated_summary"]))

# COMMAND ----------

# Define the name for the model in the Model Registry.
# We filter out some special characters which cannot be used in model names.
username = 'maria'
model_name = f"summarizer - {username}"
model_name = model_name.replace("/", "_").replace(".", "_").replace(":", "_")
print(model_name)

# COMMAND ----------

# Register a new model under the given name, or a new model version if the name exists already.
mlflow.set_registry_uri("databricks")
mlflow.register_model(model_uri=model_info.model_uri, name=model_name)

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

# COMMAND ----------

client.search_registered_models(filter_string=f"name = '{model_name}'")

# COMMAND ----------

# MAGIC %md mlflow.pyfunc è un modulo di MLflow che serve a definire un modello Python generico. Questo modulo è particolarmente importante perché consente di creare modelli che possono essere facilmente eseguiti in diversi ambienti, facilitando l'interoperabilità e la scalabilità delle soluzioni di machine learning. mlflow.pyfunc offre una convenzione per la crezione e l'invocazione di modelli, rendendolo un componente chiave nell'ecosistema MLflow per la gestione del ciclo di vita dei modelli di machine learning. Ecco alcuni dei suoi usi principali:
# MAGIC
# MAGIC Definizione di Modelli Personalizzati: Consente di incapsulare la logica di inferenza di qualsiasi modello Python in una classe PythonModel, rendendo il modello portabile e facile da eseguire in diversi contesti, come servizi web o piattaforme di data science.
# MAGIC
# MAGIC Interoperabilità: I modelli definiti come pyfunc possono essere eseguiti su una varietà di piattaforme di produzione e ambienti di machine learning senza la necessità di riscrivere il modello o utilizzare API specifiche della piattaforma. Questo facilita il passaggio dei modelli dalla fase di sviluppo alla produzione.
# MAGIC
# MAGIC Facilità di Uso: mlflow.pyfunc fornisce un'interfaccia semplice per l'invocazione di modelli che astrae i dettagli di implementazione, consentendo agli utenti di concentrarsi sulla logica del modello piuttosto che sulla gestione dell'infrastruttura.
# MAGIC
# MAGIC Registrazione e Versionamento dei Modelli: Attraverso l'integrazione con MLflow Model Registry, i modelli pyfunc possono essere facilmente registrati, versionati e tracciati, facilitando la collaborazione tra i membri del team e il controllo delle versioni del modello.
# MAGIC
# MAGIC Confezionamento e Distribuzione: MLflow consente di confezionare modelli pyfunc in formati standardizzati che possono essere distribuiti e eseguiti in diversi ambienti, come contenitori Docker, piattaforme cloud o direttamente su macchine locali.

# COMMAND ----------

model_version = 1
dev_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
dev_model

# COMMAND ----------

client.transition_model_version_stage(model_name, model_version, "staging")

# COMMAND ----------

staging_model = dev_model

# An actual CI/CD workflow might load the `staging_model` programmatically.  For example:
#   mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{Staging}")
# or
#   mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

# COMMAND ----------

results = staging_model.predict(xsum_sample.to_pandas()["document"])
display(pd.DataFrame(results, columns=["generated_summary"]))

# COMMAND ----------

client.transition_model_version_stage(model_name, model_version, "production")

# COMMAND ----------

# MAGIC %md ## Create a production workflow for batch inference
# MAGIC
# MAGIC Once the LLM pipeline is in Production, it may be used by one or more production jobs or serving endpoints.  Common deployment locations are:
# MAGIC * Batch or streaming inference jobs
# MAGIC * Model serving endpoints
# MAGIC * Edge devices
# MAGIC
# MAGIC Here, we will show batch inference using Apache Spark DataFrames, with Delta Lake format.  Spark allows simple scale-out inference for high-throughput, low-cost jobs, and Delta allows us to append to and modify inference result tables with ACID transactions.  See the [Apache Spark page](https://spark.apache.org/) and the [Delta Lake page](https://delta.io/) more more information on these technologies.

# COMMAND ----------

# Load our data as a Spark DataFrame.
# Recall that we saved this as Delta at the start of the notebook.
# Also note that it has a ground-truth summary column.
prod_data = spark.read.format("delta").load(prod_data_path).limit(10)
display(prod_data)

# COMMAND ----------

# MLflow lets you grab the latest model version in a given stage.  Here, we grab the latest Production version.
prod_model_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=f"models:/{model_name}/Production",
    env_manager="local",
    result_type="string",
)

# COMMAND ----------

# Run inference by appending a new column to the DataFrame

batch_inference_results = prod_data.withColumn(
    "generated_summary", prod_model_udf("document")
)
display(batch_inference_results)
