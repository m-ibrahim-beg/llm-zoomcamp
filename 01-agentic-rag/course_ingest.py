from gitsource import GithubRepositoryDataReader
from minsearch import Index


def load_course_content():
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    files = reader.read()
    # print(len(files))

    documents = []

    for file in files:
        doc = file.parse()
        documents.append(doc)

    # print(len(documents))
    # print(documents[0].keys())
    # dict, keys content, filename at each element in documents list
    return documents

def build_index(documents):
    index = Index(
        text_fields=['content'],
        keyword_fields=['filename']
    )
    index.fit(documents)
    return index


if __name__ == "__main__":
    documents = load_course_content()
    print(f"Q1 Answer: Number of documents is {len(documents)}")
    # Q1 answer: 72
    index = build_index(documents)

    def search(query, num_results=5):
        # boost_dict = {'question': 3.0, 'section': 0.5}
        # filter_dict = {'course': course}

        return index.search(
            query,
            num_results=num_results
            # boost_dict=boost_dict,
            # filter_dict=filter_dict
        )

    query = "How does the agentic loop keep calling the model until it stops?"
    search_results = search(query, num_results=3)
    print(f'top {len(search_results)} search results')
    for result in search_results:
        print(result['filename'])
    print(f"Q2 Answer: {search_results[0]['filename']}")
    # Q2 answer: 01-agentic-rag/lessons/14-agentic-loop.md