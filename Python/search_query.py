from langchain_community.tools import DuckDuckGoSearchResults
from langchain.docstore.document import Document

def search_duckduckgo(query):
    search = DuckDuckGoSearchResults(num_results=7)
    context = search.invoke(query)

    # Create a LangChain Document object
    doc = Document(
        page_content=context,
        # Add metadata for better organization (optional)
        metadata={
            "source": "DuckDuckGo Search Results",
            "query": query,
        }
    )

    print(doc)
    return doc

# Example usage:
query = "What is the capital of France?"
document = search_duckduckgo(query)

# Now you can use the 'document' object in LangChain tools and features
print(document.page_content)  # Access the search results content
print(document.metadata)  # Access the metadata