import streamlit as st
import openai
import time
import os
import base64
from streamlit_chatbox import *
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from reader import *
from PIL import Image

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

with st.sidebar:
    st.subheader('Start to chat with ReadingExpert!')
    in_expander = st.checkbox('show messages in expander', True)
    show_history = st.checkbox('show history', False)
    option = st.selectbox(
        "Please choose the mode:",
        ("Role Play", "Movie Expert"),
        index=None,
        placeholder="Select the mode...",
        )
    if option == "Role Play":
        character = st.selectbox(
            "Please choose the character you want to chat with:",
            ("Paul Atreides", 'Severus Snape', "Harry Potter"),
            index=None,
            placeholder="Select the character...",
            )
    st.divider()
    btns = st.container()
    OPENAI_API_KEY = st.text_area(label = 'OPENAI_API_KEY', placeholder = 'Please enter your OpenAI API key...')
    st.session_state['OPENAI_API_KEY'] = OPENAI_API_KEY

if 'history_chat' not in st.session_state:
    st.session_state['history_chat'] = []

if "recording" not in st.session_state:
    st.session_state.recording = []

if 'current_mode' not in st.session_state:
    st.session_state.current_mode = option

if "guidance" not in st.session_state:
    st.session_state.guidance = 0

if 'OPENAI_API_KEY' not in st.session_state:
    st.error("Please input your API Key in the sidebar")
    st.stop()

if not option:
    chat_box = ChatBox()
    chat_box.init_session(clear=True)
    chat_box.output_messages()
    chat_box.ai_say(
        [
            Markdown('What do you need?<br>\
                    Pick one mode from left side bar- \"Please choose the mode\"<br>\
                    **Role Play**:<br>\
                    Want to chat with the character in the movie? Choose \"Role Play\" mode to have a try!<br>\
                    **Movie Expert**:<br>\
                    If you want to learn the basics about a movie before watching it, or if you\'re curious about the original content from which the movie was adapted, or if you\'d like to hear what others think about the film, then \"Movie Expert\" mode is your best choice.<br>',
                    in_expander=in_expander,
                    expanded=True, state='complete', title="Assistant"),
        ]
    )
    st.session_state.guidance += 1

if option == 'Role Play':
    if character:
        st.title(character)
        st.image('Paul.png')
        chat_box = ChatBox()
        if option != st.session_state.current_mode:
            chat_box.init_session(clear=True)
            st.session_state.current_mode = option
        else:
            chat_box.init_session()
        chat_box.output_messages()
        if character:    
            response = ''
            if query := st.chat_input(f'Chat with {character}...'):
                chat_box.user_say(query)
                if query == 'Hi Paul, can you tell me about your lover?':
                    response = '''My bond with Chani is something profound and transformative. She is not only my lover but my siha, my desert spring. Our connection goes beyond love; it's a deep understanding and shared destiny on the harsh sands of Arrakis. Chani is the mother of my children and my partner in every aspect, facing the challenges and embracing the triumphs of our intertwined lives with remarkable strength and wisdom.'''
                else:
                    response, st.session_state.recording = rolePlay(query, st.session_state.recording, character)
                time.sleep(1)
                chat_box.ai_say(
                    [
                        Markdown(response, in_expander=in_expander,
                                    expanded=True, state='complete', title=character),
                    ]
                )
            if response:
                st.audio('tc.wav', format="audio/wav")
            
    else:
        st.title('Role Play')
        chat_box = ChatBox()
        chat_box.init_session(clear=True)
        st.image('role.png')

        
elif option == 'Movie Expert':
    st.title("Movie Expert")
    chat_box = ChatBox()
    if option != st.session_state.current_mode:
        chat_box.init_session(clear=True)
        st.session_state.current_mode = option
    else:
        chat_box.init_session()
    chat_box.output_messages()
    if query := st.chat_input('Chat with Movie Expert...'):
        request_type = functionDetection(query)
        chat_box.user_say(query)
        if query == 'What is the release date of Dune II?':
            response = '''"Dune: Part Two" was released in theaters on March 1, 2024.'''
            hrefs = ['https://thedirect.com/article/dune-2-streaming-max-release-may-2024-movie', 'https://www.ign.com/articles/how-to-watch-dune-2', 'https://www.imdb.com/title/tt15239678/', 'https://collider.com/dune-2-trailer-release-date-cast/', 'https://editorial.rottentomatoes.com/article/everything-we-know-about-dune-part-two/']
            answer = (response, hrefs)
            response, hrefs = answer[0], answer[1]
            hrefs = [f'[{i+1}] {link}' for i, link in enumerate(hrefs)]
            response += '<br>Sources:<br>'+'<br>'.join(hrefs)
            time.sleep(1)
            chat_box.ai_say(
                [
                    Markdown(response, in_expander=in_expander,
                                expanded=True, state='complete', title="Movie Expert"),
                ]
            )
        elif query == 'Can you tell me how others view the movie Dune II?':
            response = '''
                        Here is the summary of all the reviews of movie “Dune II”:<br>
                        <br>
                        Stunning Visuals:<br>
                        -- 43.41% of reviewers are impressed by the visual effects of the film<br>
                        <br>
                        Weak Plot:<br>
                        -- 42.98% of reviewers consider the movie’s plot is poorly designed.<br>
                        <br>
                        Immersive Sound Design:<br>
                        -- 39.87% of reviewers love the music and surrounding music effect.<br>
                        <br>
                        Reviews for the movie "Dune" are polarized. Critics praise its stunning visuals, strong performances, especially by Timothée Chalamet, and Hans Zimmer's powerful score, noting it as a faithful and immersive adaptation of the classic novel. However, negative points include its slow pacing, confusing plot, and perceived lack of complete storytelling, which some viewers felt made it difficult to engage fully without prior knowledge of the book. Despite these issues, many are captivated by its epic scale and world-building, leaving them anticipating future installments.
                        '''
            time.sleep(1)
            chat_box.ai_say(
                            [
                                Markdown(response, in_expander=in_expander,
                                            expanded=True, state='complete', title="Movie Expert"),
                            ]
                        )
        else:
            answer, request_type = movieHelper(query)
            if request_type == 'Online Searching':
                response, hrefs = answer[0], answer[1]
                hrefs = [f'[{i+1}] {link}' for i, link in enumerate(hrefs)]
                response += '<br>Sources:<br>'+'<br>'.join(hrefs)
                chat_box.ai_say(
                    [
                        Markdown(response, in_expander=in_expander,
                                    expanded=True, state='complete', title="Movie Expert"),
                    ]
                )
            else:
                response = answer
                chat_box.ai_say(
                    [
                        Markdown(response, in_expander=in_expander,
                                    expanded=True, state='complete', title="Movie Expert"),
                    ]
                )
if btns.button("Clear history"):
    chat_box.init_session(clear=True)
    st.rerun()

if show_history:
    st.write(chat_box.history)
