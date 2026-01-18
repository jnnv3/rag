from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

DB_PATH = "vectorstores/db"

embedding = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={"device": "cpu"}
)

vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embedding
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

system_prompt = (
    "당신은 Q&A 작업의 보조자입니다.\n"
    "아래 제공된 문서 컨텍스트만 사용해 질문에 답하세요.\n"
    "질문이 '대상'이나 '조건'에 관한 경우 불필요한 설명 없이 조건만 정리하세요.\n"
    "모르면 '모르겠습니다'라고 말하세요.\n"
    "답변은 최대 3문장으로 간결하게 하세요.\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

def run_with_k(query, k):
    retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": k,
        "fetch_k": 30,
        "lambda_mult": 0.5
    }
)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    docs = retriever.get_relevant_documents(query)
    
    print(f"\n==============================")
    print(f"k = {k}")
    print(f"검색된 청크 수: {len(docs)}\n")

    for i, doc in enumerate(docs, 1):
        print(f"[{i}] source: {doc.metadata.get('source')}")
        print(doc.page_content[:300])
        print("-" * 60)

    result = rag_chain.invoke({"input": query, "chat_history": []})
    print("LLM 답변")
    print(result["answer"])


def ask_rag(query, chat_history=None):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 30,
            "lambda_mult": 0.5
        }
    )

    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    result = rag_chain.invoke({
        "input": query,
        "chat_history": chat_history or []
    })

    return result["answer"]


if __name__ == "__main__":
    query = "청년월세지원과 다른 청년 주거 정책의 차이점은?"

    for k in [1, 2, 3]:
        run_with_k(query, k)