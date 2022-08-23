# %%
import logging
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


# Path to the trained model
model_type = "custom"
model_name = "outputs"

# Path to the Wikipedia passage collection
index_path = "indices/wiki_passages_to_build_index"

model_args = RetrievalArgs()
model_args.output_dir = model_name

# Loading the model automatically builds the index
model = RetrievalModel(
    model_type=model_type,
    model_name=model_name,
    args=model_args,
    prediction_passages=index_path,
)
