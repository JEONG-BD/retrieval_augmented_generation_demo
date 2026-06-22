import streamlit as st 

from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_classic.chains import RetrievalQA 
from langchain_classic import hub
from langchain_ollama import ChatOllama 
from dotenv import load_dotenv 
from langchain_ollama import OllamaEmbeddings 
from langchain_pinecone import PineconeVectorStore 



st.set_page_config(page_title='소득세 챗봇', page_icon='😄')

st.title('😄 소득세 챗봇') 
st.caption('소득세에 관련된 모든 것을 답해드립니다.')

if 'message_list' not in st.session_state:
    st.session_state.message_list = []
    
for message in st.session_state.message_list:
    with st.chat_message(message['role']):
        st.write(message['content'])


def get_ai_message(user_message):
    load_dotenv()
    embedding = OllamaEmbeddings(model="nomic-embed-text")    
    index_name = 'tax-index'
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    
    llm = ChatOllama(model='gemma4:latest')
    #prompt = hub.pull('rlm/rag-prompt',dangerously_pull_public_prompt=True) 
    retriever = database.as_retriever(search_kwargs={'k' : 4})
    
    rag_prompt = ChatPromptTemplate.from_template("""
        당신은 소득세 전문 AI입니다.
        아래 문서를 참고해서 사용자의 질문에 답변하세요.
        문서에 없는 내용은 모른다고 답변하세요.
        문서: {context}
        질문: {question}
        답변: """)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": rag_prompt}
    )

    dictionary = ['사람을 나타내는 표현 -> 거주지 ']
    
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전 : {dictionary} 
        질문 : {{question}}
        """)
    
    dictionary_chain = prompt | llm | StrOutputParser()
    tax_chain = {'query' : dictionary_chain} | qa_chain
    ai_message = tax_chain.invoke({'question': user_message})
    return ai_message 

if(user_question := st.chat_input(placeholder='소득세에 관련된 궁금한 내용들을 말씀해주세요')):
    with st.chat_message('user'):
        st.write(user_question)
    st.session_state.message_list.append({'role':'user', 'content':user_question})
    
    with st.spinner('답변을 생성하는 중입니다.'):
        ai_meessage = get_ai_message(user_question)
        with st.chat_message('ai'):
            st.write(ai_meessage)
        st.session_state.message_list.append({'role':'ai', 'content':ai_meessage})

