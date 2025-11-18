from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from InstructorEmbedding import INSTRUCTOR

# Global variables to store the FAISS index and the embedding model
global_index = None
global_model = None
model_names = None

instruction = "Represent the sentence for similar task retrieval: "

def initialize_index(models, model_size='7b'):
    """
    Initialize the FAISS index for model retrieval. This function:
    - Loads a global embedding model.
    - Computes average embeddings for each model based on provided samples.
    - Creates and populates a FAISS index with the model embeddings.

    Parameters:
    - models: A list of dictionaries, each containing a 'model_name' and 'sample' entries.
              'sample' should be a list of dictionaries with 'inputs'.
    - model_size: A string indicating the model size (e.g., '7b', '13b') for naming conventions.
    """
    global global_index, global_model, model_names

    # Load the embedding model for retrieval
    global_model = INSTRUCTOR('Styxxxx/lora_retriever')

    all_model_embeddings = []
    model_names = []

    # Compute average embeddings for each model
    for model in models:
        if model_size == '7b':
            model_name = f"Styxxxx/llama2_7b_lora-{model['model_name']}"
        else:
            model_name = f"Styxxxx/llama2_13b_lora-{model['model_name']}"

        model_names.append(model_name)
        model_samples = []

        # Collect sample inputs for each model
        for sample in model['sample']:
            sample_context = sample['inputs']
            model_samples.append([instruction, sample_context])

        # Compute embeddings for the model's samples and take the mean
        embeddings = get_embeddings(model_samples)
        average_embedding = np.mean(embeddings, axis=0)
        all_model_embeddings.append(average_embedding)

    # Create a FAISS index with the collected embeddings
    all_model_embeddings = np.vstack(all_model_embeddings)
    d = all_model_embeddings.shape[1]
    global_index = faiss.IndexFlatIP(d)
    global_index.add(all_model_embeddings)

def get_embeddings(text_list):
    """
    Encode a list of text samples using the global embedding model.

    Parameters:
    - text_list: A list of texts to be encoded. Each element should be [instruction, text].
    """
    return global_model.encode(text_list)

def perform_search(query_list, k=20, exclude_list=None):
    """
    Perform a similarity search for each query in the provided query_list using the global FAISS index.
    Returns a list of retrieved model names and a binary mapping matrix indicating which models
    were retrieved for each query.

    Parameters:
    - query_list: List of query strings to retrieve similar tasks/models for.
    - k: The number of top results to retrieve for each query.
    - exclude_list: An optional list of models to exclude for each query.

    Returns:
    - all_results_list: A list of all retrieved unique model names across all queries.
    - mapping_matrix: A binary matrix (list of lists) indicating which models were retrieved for each query.
    """
    global global_index, model_names

    all_results_set = set()
    query_to_results_map = {}

    # Perform search for each query
    for j, query in enumerate(query_list):
        query_embedding = get_embeddings([[instruction, query]])[0]
        distances, indices = global_index.search(np.array([query_embedding]), k+1)

        contained = False
        results = []
        for i, idx in enumerate(indices[0]):
            if i == k and not contained:
                # If we reached the last allowed result but haven't accounted for exclusion yet, skip
                continue

            model_name = model_names[idx]

            # Exclude specific model for this query if applicable
            if exclude_list and model_name == exclude_list[j]:
                contained = True
                continue

            all_results_set.add(model_name)
            results.append(model_name)

        query_to_results_map[query] = results

    # Convert all results to a list and construct a mapping matrix
    all_results_list = list(all_results_set)
    mapping_matrix = []

    for query in query_list:
        mapping_vector = [1 if result in query_to_results_map[query] else 0 for result in all_results_list]
        mapping_matrix.append(mapping_vector)

    return all_results_list, mapping_matrix