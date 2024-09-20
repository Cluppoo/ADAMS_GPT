# default packages
import os
import re
import io
import sys
__import__('pysqlite3')
import pysqlite3
sys.modules['sqlite3'] = sys.modules["pysqlite3"]
import requests
import wget

import urllib.parse
import json

import streamlit as st
import pandas as pd
import numpy as np

from st_pages import Page, show_pages, add_page_title

# chatgpt test
from langchain_community.llms import OpenAI

from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import ElasticVectorSearch
from langchain_community.vectorstores import Chroma

import os

from langchain.chains import RetrievalQAWithSourcesChain

from langchain.prompts.chat import (
            ChatPromptTemplate,
            SystemMessagePromptTemplate,
            HumanMessagePromptTemplate,
        )

# config files
import config


show_pages(
    [
        Page('app.py', 'Home', '🏠'),
    ]
)

add_page_title(layout='wide')


if "res_df" not in st.session_state:
    st.session_state.res_df = pd.DataFrame()

os.environ["OPENAI_API_KEY"] = ""

# 탭 리스트
tab_search_document, tab_search_result, tab_chatgpt = st.tabs(['ADAMS 문서 검색', '검색 결과 확인', 'ChatGPT_TEST'])

with tab_search_document:
    st.header('ADAMS 문서 검색')

    property_li = ['DocumentType']
    operator_li = list(config.operator_config.keys())
    value_li = config.document_types

    search_config = {
        'Property' : st.column_config.SelectboxColumn('검색 대상', options=property_li, default='DocumentType', required=True),
        'Operator' : st.column_config.SelectboxColumn('검색 기준', options=operator_li, default='contains', required=True),
        'Value' : st.column_config.SelectboxColumn('검색 값', options=value_li, required=True)
    }

    if "title_input" not in st.session_state:
        st.session_state["title_input"] = ""

    def title_property():
        st.session_state["title_input"] = f"!('$title',contains,'{st.session_state['text'].replace(' ', '+')}','')"

    search_cols1, search_cols2 = st.columns(2)
    with search_cols1:
        search_title = st.text_input('문서 제목 검색', key='text', on_change=title_property)
    with search_cols2:
        search_keyword = st.text_input('본문 내용 검색')

    search_title_property = st.session_state["title_input"]
    
    with st.form('검색 조건 설정'):

        sort_property = 'DocumentDate'
        sort_type = 'DESC' # ASC, DESC
        
        st.subheader('상세 검색 조건')
        search_df = pd.DataFrame(columns=['Property', 'Operator', 'Value']).reset_index(drop=True)
        search_df_editor = st.data_editor(data=search_df,
                                        column_config=search_config,
                                        num_rows='dynamic',
                                        hide_index=True,
                                        key='search_all',
                                        use_container_width=True)
        
        search_all_submit = st.form_submit_button('Submit')
        if search_all_submit:
            with st.spinner('API requsting...'):
                # 빈 테이블 행이 있을 경우 dropna를 통해 제거
                search_df_editor.dropna(axis=0, how='any', inplace=True)

                # url 정보 받아오기(all)
                if search_df_editor.shape[0] != 0:
                    options_set_li = []
                    for i in range(search_df_editor.shape[0]):
                        temp_options = search_df_editor.iloc[i].values
                        temp_option_set = f"!({temp_options[0]},{config.operator_config[temp_options[1]]},'{temp_options[2].replace(' ', '+')}','')"
                        options_set_li.append(temp_option_set)

                else:
                    options_set_li = []

                if st.session_state['text'] != '':
                    options_set_li.append(search_title_property)

                if len(options_set_li) > 0:
                    options_set = f"!({','.join(options_set_li)})"
                    url = f"https://adams.nrc.gov/wba/services/search/advanced/nrc?q=(mode:sections,sections:(filters:(public-library:!t),properties_search:{options_set},single_content_search:'{search_keyword.replace(' ', '+')}'))&qn=New&tab=content-search-pars&s={sort_property}&so={sort_type}"

                else:
                    url = f"https://adams.nrc.gov/wba/services/search/advanced/nrc?q=(mode:sections,sections:(filters:(public-library:!t),single_content_search:'{search_keyword.replace(' ', '+')}'))&qn=New&tab=content-search-pars&s={sort_property}&so={sort_type}"

                api_res = requests.get(url)

                doc_columns = ['DocumentTitle', 'DocumentType', 'DocumentDate', 'EstimatedPageCount', 'URI']

                try:
                    st.session_state.res_df = pd.read_xml(api_res.content, xpath='//search/resultset/result')
                    st.session_state.res_df = st.session_state.res_df[doc_columns]
                    st.session_state.res_df['URI'] = st.session_state.res_df['URI'].apply(lambda x: x.replace(' ', '+'))
                    st.text('API Request Done')
                except ValueError:
                    st.text('No Search Result')
                    st.session_state.res_df = pd.DataFrame(columns=doc_columns)



with tab_search_result:

    def pdf_download(url, title):
            os.makedirs('./downloads/', exist_ok=True)
            response = wget.download(url, f'./downloads/{title}.pdf')

            return None
    
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

    with st.form('search result'):

        res_column_config = {
            'URI' : st.column_config.LinkColumn('URI', display_text='LINK')
        }

        search_result_df = st.dataframe(st.session_state.res_df,
                                        column_config=res_column_config,
                                        use_container_width=True,
                                        hide_index=True,
                                        on_select='rerun',
                                        selection_mode='multi-row'
                                        )
        
        
        check_pdf_download = st.form_submit_button('Checked File Download')


        if check_pdf_download:
            with st.spinner('Downloading Files...'):
                select_index_li = search_result_df.selection['rows']
                if len(select_index_li) != 0:
                    for i in select_index_li:
                        select_row = st.session_state.res_df.iloc[i]
                        pdf_download(url=convert_link(select_row['URI']),
                                    title=select_row['DocumentTitle'])
                    st.text('File Downloads Done')
                else:
                    st.text('No Checked Rows')
        
with tab_chatgpt:
    if "api_key_check" not in st.session_state:
        st.session_state["api_key_check"] = False

    if "documents" not in st.session_state:
        st.session_state["documents"] = []

    with st.form('search result dataframe'):
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        os.environ["OPENAI_API_KEY"] = openai_api_key

        res_column_config_gpt = {
            'URI' : st.column_config.LinkColumn('URI', display_text='LINK')
        }

        search_res_df_gpt = st.dataframe(st.session_state.res_df,
                                        column_config=res_column_config_gpt,
                                        use_container_width=True,
                                        hide_index=True,
                                        on_select='rerun',
                                        selection_mode='single-row')
        
        check_gpt_button = st.form_submit_button('Checked File to GPT')

        if check_gpt_button:
            st.session_state["api_key_check"] = True

            with st.spinner('GPT requsting...'):
                if not openai_api_key:
                    st.info("Please add your OpenAI API key to continue.")
                    st.stop()

                selected_index_gpt = search_res_df_gpt.selection['rows'][0]
                selected_url_gpt = convert_link(st.session_state.res_df.iloc[selected_index_gpt]['URI'])

                loader = PyPDFLoader(selected_url_gpt)
                documents = loader.load()

                st.session_state["documents"] = documents

                st.text('Document to GPT Done.')


    def initialize_retriever(documents):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(texts, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        return retriever
    
    def initialize_chain(retriever):

        system_template = """You are an expert in various scientific disciplines. Use the following context to answer the user's question.
        Given the follwing summaries of a long documnet and a question, create a final answer and referenced page number.
        If you don't know the answer, just say that "I don't know", don't try to make up an answer.
        ----------------
        {summaries}

        You MUST answer in Korean and in Markdown format:

        """

        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}")
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        chain_type_kwargs = {"prompt": prompt}
        llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0)

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )

        return chain

    # try:
    if st.session_state["api_key_check"]:
        retriever = initialize_retriever(st.session_state["documents"])

        chain = initialize_chain(retriever)

        def generate_response(input_text):
            result = chain(input_text)
            return result['answer']

        st.subheader('질문을 적어 주세요')

        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "질문을 적어 주세요 무엇을 도와 드릴까요?"}]

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            msg =  generate_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)

        if st.button("초기화"):
            retriever = initialize_retriever(st.session_state["documents"])
            chain = initialize_chain(retriever)
            st.session_state["messages"] = [{"role": "assistant", "content": "질문을 적어 주세요 무엇을 도와 드릴까요?"}]
            st.write("대화가 초기화되었습니다.")
            st.write(st.session_state["documents"])

        # else:
        #     st.warning("Please enter a valid OpenAI API Key")


    # except:
    #     st.warning("Please enter a valid OpenAI API Key")

    
