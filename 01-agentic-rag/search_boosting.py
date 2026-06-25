import requests
from minsearch import Index

# get documents from API
docs_url = 'https://datatalks.club/faq/json/courses.json'
response = requests.get(docs_url)
courses_raw = response.json()

documents = []
url_prefix = 'https://datatalks.club/faq'

for course in courses_raw:
    course_url = f"{url_prefix}{course['path']}"
    course_response = requests.get(course_url)
    course_response.raise_for_status()
    course_data = course_response.json()
    documents.extend(course_data)

len(documents)

# builx search index

index = Index(
    text_fields=['question', 'section', 'answer'],
    keyword_fields=['course']
)
index.fit(documents)

question = 'I just discovered the course. Can I join now?'
search_results = index.search(
    question,
    boost_dict={'question': 2.0, 'section': 0.5},
    filter_dict={'course': 'llm-zoomcamp'},
    num_results=5
)


# search_results