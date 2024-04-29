import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import openai
import pandas as pd
import os
import json
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from duckduckgo_search import DDGS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

def fileReader(prompt:str, file_path:str)->str:
    documents = SimpleDirectoryReader(file_path).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query(prompt)
    return response

def paulInformation(file_path:str)->str:
    context = ''
    with open(file_path, 'r') as file:
        for line in file:
            context += line.strip() + '\n'
    return context

def rolePlay(prompt, recording, character):
    paul_information = paulInformation('Paul_info.txt')
    memory = fileReader(prompt, 'dune')
    #memory = ''
    completion = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
        {"role": "system", "content": f'From now on, you are {character}. Remember, you are not the assistant, you are role playing! So do not say anything like \'How may I assist you today\'. Say what Paul will say.'},#This is your basic information {paul_information}
        {"role": "system", "content": f'This is your memory which contains the people you have met, the place you have been to, and everything you have experienced: {memory}. This is your chat history with others: {recording}. All of your answer should be based on your memory and the chat history.'},
        {"role": "system", "content": f'When you speak, you need to consider whether {character} has experienced this before. How would he perceive this matter? In what tone and manner would he express it? Then you need to mimic {character}\'s tone, communicating with others from his perspective.'},
        {"role": "user", "content": f"{prompt}"}
        ]
    )
    recording.append('User:'+ prompt)
    recording.append('System:'+ completion.choices[0].message.content)
    response = completion.choices[0].message.content
    return response, recording

def chatGPT(prompt: str, document: str)->str:
    """
    define an agent to read the document and complete the conversation or answer the question.
    :param prompt: user input like 'Give me a breif introduction of Dune 2'.
    :param document: the document that includes the necessary information to complete the conversation or answer the question .
    :return: the answer.
    """
    completion = openai.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": f"You will be given a document which consists of many movie related information. You need to use these information to finish the conversation or answer the question."},
        {"role": "system", "content": f"This is the document: {document}"},
        {"role": "user", "content": f"{prompt}"}
    ],
    temperature = 0.1
    )
    response = completion.choices[0].message.content
    return response

def summaryAssistant(prompt: str, document: list)->str:
    """
    define an agent to summarize the given content.
    :param prompt: user input like 'Give me a breif introduction of Dune 2'.
    :param document: the document that includes the necessary information to complete the conversation or answer the question .
    :return: the answer.
    """
    completion = openai.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": f"You will be given a document which contains the contents from different web pages about a specific prompt. Please give a summary of these contents."},
        {"role": "system", "content": f"This is the document: {document}"},
        {"role": "user", "content": f"{prompt}"}
    ],
    temperature = 1
    )
    response = completion.choices[0].message.content
    return response

def reviewSummary(prompt: str, document: list)->str:
    completion = openai.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": f"You will be given a list that contains many keywords which are extracted from the user reviews of the movie Dune. Please give a summarization of all the reviews based on these keywords."},
        {"role": "system", "content": f"This is the summary of the keywords: {document}"},
        {"role": "user", "content": f"{prompt}"}
    ],
    temperature = 1
    )
    response = completion.choices[0].message.content
    return response

def functionDetection(prompt: str):
    completion = openai.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": f"There are three different funtions of your app. You will be given a prompt which is the user input. You need to detect which funtion to use."},
        {"role": "system", "content": f"You only have three functions which are \'Movie Review\', \'Book Reading\', \'Online Searching\'."},
        {"role": "system", "content": f"\'Movie Review\': When users want to know how others view the movie, you need to choose this function."},
        {"role": "system", "content": f"\'Book Reading\': Some movies are adapted from books. If users want to know about the content of the original books, you need to choose this feature."},
        {"role": "system", "content": f"\'Online Searching\': When users want to know some basic information about the movie, you need to choose this function."},
        {"role": "system", "content": f"\'Online Searching\': If the users\' question is irrelated with movie, or books, or the actors/actress/director, please output \'Irrelevant\'."},
         {"role": "system", "content": f"You can only output one of \'Movie Review\', \'Book Reading\', \'Online Searching\', \'Irrelevant\'. Any other answers are totlly restricted"},
        {"role": "user", "content": f"{prompt}"}
    ],
    temperature = 0
    )
    response = completion.choices[0].message.content
    return response

def webSearch(prompt: str)->list:
    """
    use the API of Duckduckgo to search online
    :param prompt: user input like 'Give me a breif introduction of Dune 2'.
    :return: a list that contains the link of the 5 most related web pages.
    """
    results = DDGS().text(prompt, max_results=5)
    hrefs = [i['href'] for i in results]
    return hrefs

def singlePageReader(prompt: str, href: str)->str:
    """
    use LangChain to read the content on a given web page
    :param prompt: user input like 'Give me a breif introduction of Dune 2'.
    :href: the web link that we want the agent to read
    :return: answer of the prompt based on the web page.
    """
    loader = WebBaseLoader(href)
    docs = (loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(docs)
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=splits, embedding=embedding)
    llm = ChatOpenAI(temperature=1)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(), llm=llm
    )
    unique_docs = retriever_from_llm.get_relevant_documents(query=prompt)
    response = chatGPT(prompt, unique_docs)
    return response

def simpleMultiPagesReader(prompt: str, hrefs: list)->str:
    '''
    use LangChain to read the content on multiple web pages
    :param prompt: user input like 'Give me a breif introduction of Dune 2'.
    :href: the web link that we want the agent to read
    :return: answer of the prompt based on the web page.
    '''
    answer = ''
    for i, href in enumerate(hrefs[:2]):
        try:
          answer += f'Content of web page {i+1}: ' + singlePageReader(prompt, href)
        except:
          continue
    response = summaryAssistant(prompt, answer)
    return response

def complexMultiPagesReader(prompt: str, hrefs: list)->str:
    response = ''
    doc = ''
    Metadata = []
    for href in hrefs:
        loader = WebBaseLoader(href)
        docs = (loader.load())
        text = ' '.join(docs[0].page_content.split())
        doc += text
    response = summaryAssistant(prompt, doc)
    return response

def webHelper(prompt:str, mode = 'single_page', href = 'https://en.wikipedia.org/wiki/Dune_(2021_film)')->str:
    if mode == 'milti_page':
        hrefs = webSearch(prompt)
        response = simpleMultiPagesReader(prompt, hrefs)
        return response, hrefs
    elif mode == 'milti_page_pro':
        hrefs = webSearch(prompt)
        response = complexMultiPagesReader(prompt, hrefs)
        return response, hrefs
    elif mode == 'single_page':
        response = singlePageReader(prompt, href)
    return response

def reviewHelper(prompt):
    reviews = pd.read_csv('standard_Dune.csv')
    positve_phrase = reviews['positve phrase']
    negative_phrase = reviews['negative phrase']
    positve_review = []
    negative_review = []
    for i in positve_phrase:
        positve_review += eval(i)
    for i in negative_phrase:
        negative_review += eval(i)
    positive = reviewSummary(prompt, positve_review)
    negative = reviewSummary(prompt, negative_review)
    prompt2 = 'These are the negative reviews and positive reviews of the movie Dune from other users. Please merge them together and output a summary that includes both positive and negative reviews. And you need to explicitly seperate the positive reviews and negative review, which makes it more clear for user to read.'
    response = reviewSummary(prompt2, negative+positive)
    return response

def movieHelper(prompt:str):
    request_type = ''
    max_iter = 5
    iter = 0
    while request_type not in ['Book Reading', 'Irrelevant', "Movie Review", 'Online Searching']:
        if iter >= max_iter:
            request_type = 'Online Searching'
            break
        request_type = str(functionDetection(prompt))
        iter += 1
    if request_type == 'Book Reading':
        response = fileReader(prompt, 'dune')
    elif request_type == 'Irrelevant':
        response = 'It seems like that your question is irrelevant.'
    elif request_type == "Movie Review":
        response = reviewHelper(prompt)
    elif request_type == 'Online Searching':
        response, href = webHelper(prompt, mode = 'milti_page')
        return (response, href), request_type
    return response, request_type

# prompt = '沙丘的导演是谁'
# answer, request_type = movieHelper(prompt)[0], movieHelper(prompt)[1]
# print(request_type)
# print(answer)

# prompt = 'I want to know how bitcoin works and why it is important.'
# level = 'Green hand with 0 knowledge in Web 3 and block chain.'
# answers = webThreeHelper(prompt, mode = 'study_plan', level=level)
# print(answers)

# prompt = 'Can you give me some today\'s AI news?'
# href = webSearch(prompt)
# answer = complexMultiPagesReader(prompt, href)[0]
# print(href)
# print(answer)
