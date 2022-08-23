import streamlit as st
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs


model_type = "custom"
model_name = "outputs/"

index_path = "outputs/prediction_passage_dataset"

model_args = RetrievalArgs()
model_args.retrieve_n_docs = 10


# def simple_transformers_model(model):
#     return (type(model).__name__, model.args)


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model(model_type, model_name, index_path, model_args):
    model = RetrievalModel(
        model_type=model_type,
        model_name=model_name,
        prediction_passages=index_path,
        args=model_args,
    )

    return model


def get_model_details():
    model_name = st.text_input("Model name", "outputs")
    index_path = st.text_input("Path to the index", "outputs/prediction_passage_dataset")

    return model_name, index_path


def predict(model, query):
    retrieved_docs, *_ = model.predict([query])

    retrieved_docs = retrieved_docs[0]
    retrieved_docs = [doc.replace("\\", "") for doc in retrieved_docs]

    return retrieved_docs


def main():
    st.title("Dense retrieval with Simple Transformers")

    model_name, index_path = get_model_details()

    model_load_state = st.text("Loading model...")
    model = load_model(model_type, model_name, index_path, model_args)
    model_load_state.text("Model loaded succesfully.")

    query = st.text_input("Enter query:", "")

    if query:
        st.write(predict(model, query))


if __name__ == "__main__":
    main()
