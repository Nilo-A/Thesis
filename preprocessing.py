Functions for text splitting and tokenizing.

```python
import re

# Function to split text into pages based on markers
def split_text_by_markers(text, markers):
    pattern = '|'.join(re.escape(marker) for marker in markers)
    parts = re.split(f'({pattern})', text)
    pages = []

    # Combine parts to form pages
    for i in range(1, len(parts), 2):
        page = parts[i-1].strip() + ' ' + parts[i].strip()
        pages.append(page.strip())

    # Add the last part if it doesn't end with a marker
    if len(parts) % 2 == 0:
        pages.append(parts[-1].strip())

    return pages

# Function to separate the combined text into individual documents
def separate_documents(document_text, document_marker):
    documents = re.split(document_marker, document_text)
    documents = [doc.strip() for doc in documents if doc.strip()]
    return documents

# Function to count pages
def count_pages(text, page_marker):
    return len(re.findall(page_marker, text))
