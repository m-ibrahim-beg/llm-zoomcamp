from course_ingest import load_course_content, build_index
from course_rag_helper import RAGBase
from openai import OpenAI
from gitsource import chunk_documents
from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner, DisplayingRunnerCallback


documents = load_course_content()
chunks = chunk_documents(documents, size=2000, step=1000)
index = build_index(chunks)

openai_client = OpenAI()

assistant = RAGBase(
    index=index,
    llm_client=openai_client
)

# answer = assistant.rag("How does the agentic loop keep calling the model until it stops?")
# Q3 answer: input tokens 6912 (so 7000)
# print(answer)

# documents = load_course_content()
# chunks = chunk_documents(documents, size=2000, step=1000)
# print(len(chunks))
# Q4 answer:295

# Q5 answer: 2309 input tokens, which is 3x fewer

# Q6
# search function that uses chunk index 
def search(query: str) -> dict[str, str]:
    """
    Search the FAQ database for entries matching the given query.
    """
    return index.search(
        query,
        num_results=5,
        boost_dict={"question": 3.0, "section": 0.5},
        filter_dict={"course": "llm-zoomcamp"}
    )

agent_tools = Tools()
agent_tools.add_tool(search)
instructions = "You're a course teaching assistant. Answer the student's question using the search tool. Make multiple searches with different keywords before answering."

chat_interface = IPythonChatInterface()
callback = DisplayingRunnerCallback(chat_interface)

runner = OpenAIResponsesRunner(
    tools=agent_tools,
    developer_prompt=instructions,
    chat_interface=chat_interface,
    llm_client=OpenAIClient(model="gpt-5.4-mini")
)

result = runner.loop(
    prompt="How does the agentic loop work, and how is it different from plain RAG?",
    callback=callback,
)
print(result)
# result2 = runner.loop(
#     prompt="How do I run a different model?",
#     previous_messages=result.all_messages,
#     callback=callback,
# )
# Q6 answer: 3



# def build_context(search_results):
#     lines = []

#     for doc in search_results:
#         lines.append(doc['filename'])
#         lines.append('Lesson Content: ' + doc['content'])
#         lines.append('')

#     return '\n'.join(lines).strip()

# context = build_context(search_results)
# print(context)