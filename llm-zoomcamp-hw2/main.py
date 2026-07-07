from embedder import Embedder

def main():
    print("Hello Hw2")


def question1():
    embed = Embedder()
    q1 = "How does approximate nearest neighbor search work?"
    v1 = embed.encode(q1)
    # print(v1[0])
    return v1
    # -0.02058203437252893
    # Q1 Answer: -0.02

def retrieve_content_given_filename(filename, documents):
    for doc in documents:
        if doc["filename"] == filename:
            return doc["content"]
    raise ValueError(f"No document found with filename: {filename}")

def question2():
    from gitsource import GithubRepositoryDataReader

    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    documents = [file.parse() for file in reader.read()]
    # documents: dict_keys[content, filename]
    # print(documents[0].keys())
    # print(documents[0]["content"])
    # print(documents[0]["filename"])

    content1 = retrieve_content_given_filename("02-vector-search/lessons/07-sqlitesearch-vector.md", documents)
    # # embed its content 
    embed = Embedder() 
    dv = embed.encode(content1)
    v1 = question1()
    # # compute cosine similarity with query from q1 
    similarity=v1.dot(dv)
    print(similarity)
    return similarity
    # 0.36107027225579694
    # Q2 Answer: 0.37

def question3():
    from gitsource import GithubRepositoryDataReader
    from gitsource import chunk_documents

    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    documents = [file.parse() for file in reader.read()]
    chunks = chunk_documents(documents, size=2000, step=1000)
    embed = Embedder()

    from tqdm.auto import tqdm
    import numpy as np

    X = []
    # print(chunks)
    for i in chunks:
        # print(i.keys())
        # print(i["content"])
        batch_vectors = embed.encode_batch([i["content"]])
        X.extend(batch_vectors)

    X = np.array(X)

    q1 = "How does approximate nearest neighbor search work?"
    v1 = embed.encode(q1)

    scores = X.dot(v1)
    # print(scores)
    idx = np.argmax(scores)

    chunks[idx]
    print(chunks[idx])
    print(chunks[idx]["filename"])
    return chunks[idx]["filename"]
    # Q3 Answer: 02-vector-search/lessons/07-sqlitesearch-vector.md

def question4():
    from gitsource import GithubRepositoryDataReader
    from gitsource import chunk_documents

    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    documents = [file.parse() for file in reader.read()]
    chunks = chunk_documents(documents, size=2000, step=1000)
    embed = Embedder()

    import numpy as np

    X = []
    # print(chunks)
    for i in chunks:
        # print(i.keys())
        # print(i["content"])
        batch_vectors = embed.encode_batch([i["content"]])
        X.extend(batch_vectors)

    X = np.array(X)

    from minsearch import VectorSearch
    vindex = VectorSearch(keyword_fields=["course"])
    vindex.fit(X, chunks)

    return vindex, embed
    query = "What metric do we use to evaluate a search engine?"
    query_vector = embed.encode(query)
    results = vindex.search(query_vector, num_results=3)
    # print(results)
    print(results[0])
    # filename: 04-evaluation/lessons/05-search-metrics.md
    # Q4 Answer 04-evaluation/lessons/05-search-metrics.md

def question5():
    vindex, embed = question4()
    from minsearch import Index
    from gitsource import GithubRepositoryDataReader
    from gitsource import chunk_documents

    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    documents = [file.parse() for file in reader.read()]
    chunks = chunk_documents(documents, size=2000, step=1000)

    index = Index(
        text_fields=['content'],
        # keyword_fields=['filename']
    )
    index.fit(chunks)

    query = "How do I store vectors in PostgreSQL?"
    query_vector = embed.encode(query)
    vector_search_results = vindex.search(query_vector, num_results=5)
    # print(vector_search_results)
    for result in vector_search_results:
        print(result['filename'])


    # index.search(
    #     query,
    #     num_results=5,
    #     filter_dict={"course": "llm-zoomcamp"}
    # )
    search_results = index.search(
        query,
        num_results=5
        # filter_dict={"course": "llm-zoomcamp"}
    )
    print(f'top {len(search_results)} search results')
    for result in search_results:
        print(result['filename'])

    # Q5 Answer: 02-vector-search/lessons/08-pgvector.md appears in vector search but not in text search

def question6():
    vindex, embed = question4()
    from minsearch import Index
    from gitsource import GithubRepositoryDataReader
    from gitsource import chunk_documents

    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    documents = [file.parse() for file in reader.read()]
    chunks = chunk_documents(documents, size=2000, step=1000)

    index = Index(
        text_fields=['content'],
        # keyword_fields=['filename']
    )
    index.fit(chunks)

    query = "How do I give the model access to tools?"
    query_vector = embed.encode(query)
    vector_search_results = vindex.search(query_vector, num_results=5)
    # print(vector_search_results)
    for result in vector_search_results:
        print(result['filename'])


    # index.search(
    #     query,
    #     num_results=5,
    #     filter_dict={"course": "llm-zoomcamp"}
    # )
    search_results = index.search(
        query,
        num_results=5
        # filter_dict={"course": "llm-zoomcamp"}
    )
    print(f'top {len(search_results)} search results')
    for result in search_results:
        print(result['filename'])

    def rrf(result_lists, k=60, num_results=5):
        scores = {}
        docs = {}

        for results in result_lists:
            for rank, doc in enumerate(results):
                key = (doc["filename"], doc["start"])
                scores[key] = scores.get(key, 0) + 1 / (k + rank)
                docs[key] = doc

        ranked = sorted(scores, key=scores.get, reverse=True)
        return [docs[key] for key in ranked[:num_results]]    

    final_results = rrf([vector_search_results, search_results])
    print(final_results)
    # Q6 Answer: 01-agentic-rag/lessons/16-other-frameworks.md


if __name__ == "__main__":
    # main()
    question6()
