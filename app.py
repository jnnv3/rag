import gradio as gr
from rag_chat import ask_rag
from langchain_core.messages import HumanMessage, AIMessage

def response(message, history):
    chat_history = []
    for human, ai in history:
        chat_history.append(HumanMessage(content=human))
        chat_history.append(AIMessage(content=ai))

    answer = ask_rag(message, chat_history)
    return answer


gr.ChatInterface(
    fn=response,
    title="청년정책 챗봇",
    textbox=gr.Textbox(
        placeholder="질문을 입력하세요",
        container=False
    ),
    chatbot=gr.Chatbot(height=500),
    examples=[
        ["청년월세지원과 다른 청년 주거 정책의 차이점은?"],
        ["청년정책의 주요 목표는 무엇인가요?"],
    ]
).launch()
