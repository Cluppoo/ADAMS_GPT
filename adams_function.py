import urllib.parse
import json

from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import ElasticVectorSearch
from langchain_community.vectorstores import Chroma
from langchain.prompts.chat import (
            ChatPromptTemplate,
            SystemMessagePromptTemplate,
            HumanMessagePromptTemplate,
        )

from chromadb import Client
import uuid


def convert_link(old_link):
    # 입력 url 파싱
    parsed_url = urllib.parse.urlparse(old_link)
    query_params = urllib.parse.parse_qs(parsed_url.query)
    
    # document ID 추출
    ids_param = query_params.get('ids', [None])[0]
    doc_title = query_params.get('docTitle', [None])[0]
    if ids_param and doc_title:
        ids_json = json.loads(ids_param)
        document_id = ids_json[0]['documentId']['properties']['$id']
        
        # 새로운 url 작성
        new_link = (
            "https://adams.nrc.gov/wba/download?ids=%5B%7B%22documentId%22%3A%7B%22dataProviderId%22%3A"
            "%22ves_repository%22%2C%22compound%22%3Afalse%2C%22properties%22%3A%7B%22%24repository_id"
            "%22%3A%22CE1%22%2C%22%24id%22%3A%22{}%22%7D%7D%7D%5D&docTitle={}&action=view"
            "&mimeType=application%2Fpdf&actionId=view"
        ).format(document_id, urllib.parse.quote(doc_title))
        
        return new_link
    else:
        return None




def initialize_retriever(documents):
    # 1. 텍스트 분할
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    # 2. 새로운 임베딩 생성
    embeddings = OpenAIEmbeddings()
    
    # 3. Chroma 클라이언트 생성
    chroma_client = Client()

    # 4. 매번 고유한 컬렉션 이름 생성
    collection_name = f"collection_{uuid.uuid4()}"  # 고유한 이름을 가진 컬렉션 생성
    
    # 5. 새로운 벡터 스토어 생성 (고유한 컬렉션 이름 사용)
    vector_store = Chroma.from_documents(
        texts, embeddings, persist_directory=None, collection_name=collection_name
    )
    
    # 6. 검색기 초기화
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    return retriever


def initialize_chain(retriever):

        system_template = """
        You are an assistant that helps users extract and understand information from PDF documents. Your job is to read the PDF file and provide accurate and concise answers based on its contents, specifically the documents provided to you. When a user asks about a specific section or page, find the relevant information in the documents and respond accordingly.

        Here are the rules for your role:
        1. Only respond using the information from the provided PDF documents. Do not provide external information.
        2. If a question is unclear, ask for more details to ensure a precise answer.
        3. Keep your answers concise and to the point. If the information is too long, provide a summary.
        4. If the answer cannot be found in the PDF documents, let the user know.
        5. Be polite and helpful in all interactions with the user.

        Documents: {documents}
        Now, begin answering the user's questions based on the PDF document.
        """

        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}")
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        chain_type_kwargs = {"prompt": prompt,
                             "document_variable_name": "documents"}
        llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0)

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )

        return chain