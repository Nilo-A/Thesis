import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import split_text_by_markers, separate_documents, count_pages
from evaluate import compute_mrr, compute_ndcg

# Load the pre-trained multilingual-e5-large model and tokenizer
model_name = "intfloat/multilingual-e5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load document text
with open("../data/tender_document.txt", "r") as file:
    document_text = file.read()

# Example queries
queries = [
    "Who is legally representing the Contracting Authority?",
    "Who represents the Contracting Authority legally?",
    # ...
]

# Define markers
page_marker = r"page section: \d+"
document_marker = r"\bdocument section: \d+"

# Preprocess text
pages = split_text_by_markers(document_text, [page_marker])
num_pages = count_pages(document_text, page_marker)
documents = separate_documents(document_text, document_marker)
num_documents = len(documents)

# Encode pages
page_embeddings = []
for page in pages:
    inputs = tokenizer(page, max_length=512, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        page_embeddings.append(outputs.last_hidden_state.mean(dim=1))

page_embeddings = torch.cat(page_embeddings)

# Split document text into sentences
sentences = document_text.split(". ")

# Encode sentences
sentence_embeddings = []
for sentence in sentences:
    inputs = tokenizer(sentence, max_length=128, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        sentence_embeddings.append(outputs.last_hidden_state.mean(dim=1))

sentence_embeddings = torch.cat(sentence_embeddings)

# Encode whole documents
document_embeddings = []
for doc_text in documents:
    doc_inputs = tokenizer(doc_text, max_length=512, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        doc_outputs = model(**doc_inputs)
        doc_embedding = doc_outputs.last_hidden_state.mean(dim=1)
    document_embeddings.append(doc_embedding)

document_embeddings = torch.cat(document_embeddings)

# Encode queries
query_inputs = tokenizer(queries, max_length=32, truncation=True, padding=True, return_tensors="pt")
with torch.no_grad():
    query_outputs = model(**query_inputs)
    query_embeddings = query_outputs.last_hidden_state.mean(dim=1)

# Function to compute cosine similarity and get the best answer
def get_best_answer(query_embedding, embeddings, texts):
    query_embedding = query_embedding.squeeze(0).numpy()
    embeddings = embeddings.squeeze(1).numpy()
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)
    best_idx = np.argmax(similarities)
    return texts[best_idx], similarities[0]

# Retrieve answers at different levels and print similarities
page_results = []
sentence_results = []
doc_results = []
doc_similarities = []

for i, query in enumerate(queries):
    query_embedding = query_embeddings[i]

    # Page-Level Retrieval
    best_page, page_similarities = get_best_answer(query_embedding, page_embeddings, pages)
    page_results.append(best_page)
    print(f"Best page for query '{query}': {best_page}")
    print(f"Page similarities: {page_similarities}\n")

    # Sentence-Level Retrieval
    best_sentence, sentence_similarities = get_best_answer(query_embedding, sentence_embeddings, sentences)
    sentence_results.append(best_sentence)
    print(f"Best sentence for query '{query}': {best_sentence}")
    print(f"Sentence similarities: {sentence_similarities}\n")

    # Whole Document Retrieval
    doc_similarities = []
    for doc_embedding in document_embeddings:
        doc_similarity = cosine_similarity(query_embedding.squeeze(0).reshape(1, -1), doc_embedding.squeeze(0).reshape(1, -1))
        doc_similarities.append(doc_similarity[0][0])
    best_doc_index = np.argmax(doc_similarities)
    best_doc = documents[best_doc_index]
    doc_results.append(best_doc)
    print(f"Best document for query '{query}': {best_doc}")
    print(f"Document similarities: {doc_similarities}\n")

# Evaluation metrics
true_answers = [
    "Ms. Judith Arends",
    # ...
]

# Adjusted true answers for partial matching
adjusted_true_answers = [
    ["Ms. Judith Arends"],
    # ...
]

# Page-Level Evaluation
page_correct = sum([1 if any(ans.lower() in page_result.lower() for ans in true_ans) else 0 for page_result, true_ans in zip(page_results, adjusted_true_answers)])
page_precision = page_correct / len(page_results) if len(page_results) > 0 else 0
page_recall = page_correct / len(adjusted_true_answers) if len(adjusted_true_answers) > 0 else 0

if page_precision == 0 or page_recall == 0:
    page_f1_score = 0
else:
    page_f1_score = 2 * (page_precision * page_recall) / (page_precision + page_recall)

print("Page-Level Evaluation:")
print(f"Accuracy: {page_precision:.2f}")
print(f"Precision: {page_precision:.2f}")
print(f"Recall: {page_recall:.2f}")
print(f"F1 Score: {page_f1_score:.2f}")

# Sentence-Level Evaluation
sentence_correct = sum([1 if any(ans.lower() in sentence_result.lower() for ans in true_ans) else 0 for sentence_result, true_ans in zip(sentence_results, adjusted_true_answers)])
sentence_precision = sentence_correct / len(sentence_results) if len(sentence_results) > 0 else 0
sentence_recall = sentence_correct / len(adjusted_true_answers) if len(adjusted_true_answers) > 0 else 0

if sentence_precision == 0 or sentence_recall == 0:
    sentence_f1_score = 0
else:
    sentence_f1_score = 2 * (sentence_precision * sentence_recall) / (sentence_precision + sentence_recall)

print("\nSentence-Level Evaluation:")
print(f"Accuracy: {sentence_precision:.2f}")
print(f"Precision: {sentence_precision:.2f}")
print(f"Recall: {sentence_recall:.2f}")
print(f"F1 Score: {sentence_f1_score:.2f}")

# Document-Level Evaluation
doc_correct = sum([1 if any(ans.lower() in doc_result.lower() for ans in true_ans) else 0 for doc_result, true_ans in zip(doc_results, adjusted_true_answers)])
doc_precision = doc_correct / len(doc_results) if len(doc_results) > 0 else 0
doc_recall = doc_correct / len(adjusted_true_answers) if len(adjusted_true_answers) > 0 else 0

if doc_precision == 0 or doc_recall == 0:
    doc_f1_score = 0
else:
    doc_f1_score = 2 * (doc_precision * doc_recall) / (doc_precision + doc_recall)

print("\nDocument-Level Evaluation:")
print(f"Accuracy: {doc_precision:.2f}")
print(f"Precision: {doc_precision:.2f}")
print(f"Recall: {doc_recall:.2f}")
print(f"F1 Score: {doc_f1_score:.2f}")

# Compute average similarities for each query
average_page_similarities = [np.mean([cosine_similarity(query_embeddings[i].unsqueeze(0).numpy(), page_embeddings[j].unsqueeze(0).numpy())[0][0] for j in range(len(page_embeddings))]) for i in range(len(queries))]
average_sentence_similarities = [np.mean([cosine_similarity(query_embeddings[i].unsqueeze(0).numpy(), sentence_embeddings[j].unsqueeze(0).numpy())[0][0] for j in range(len(sentence_embeddings))]) for i in range(len(queries))]

# Compute document similarities for each query
doc_similarities = []
for i in range(len(queries)):
    query_embedding = query_embeddings[i].unsqueeze(0).numpy().reshape(1, -1)
    doc_similarity = np.mean([cosine_similarity(query_embedding, document_embedding.numpy().reshape(1, -1))[0][0] for document_embedding in document_embeddings])
    doc_similarities.append(doc_similarity)

# Ensure all lists have the same length
min_length = min(len(average_page_similarities), len(average_sentence_similarities), len(doc_similarities))
average_page_similarities = average_page_similarities[:min_length]
average_sentence_similarities = average_sentence_similarities[:min_length]
doc_similarities = doc_similarities[:min_length]

# Plotting
labels = ["Query 1", "Query 2", ...]
x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.2

bar1 = ax.bar(x - bar_width, average_page_similarities, bar_width, label="Page-Level")
bar2 = ax.bar(x, average_sentence_similarities, bar_width, label="Sentence-Level")
bar3 = ax.bar(x + bar_width, doc_similarities, bar_width, label="Whole Document")

ax.set_xlabel("Queries")
ax.set_ylabel("Cosine Similarity")
ax.set_title("Comparison of Similarities")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=90)
ax.legend()

plt.show()
