import streamlit as st
import base64
from streamlit_extras.switch_page_button import switch_page

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
bin_str = get_base64("imdb.png")
background = """
                <style>
                .stApp {
                    background-image: url("data:image/png;base64,%s");
                    background-size: cover; /* 调整背景图像的大小以覆盖整个屏幕 */
                }
                </style>
             """% bin_str
st.markdown(background, unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .fixed-box {
        position: fixed; /* 将框固定在页面上方 */
        top: 0; /* 距离页面顶部位置 */
        width: 50%; /* 宽度占满整个页面 */
        height: 205px;
        padding: 10px; /* 内边距 */
    }
    </style>
    <p style='font: serif; text-shadow: 2px 2px 4px #c1e6f5; color: #4abded; font-size: 130px; text-align:center; margin-bottom: -20px; margin-top: -80px'><b></b></p>
    """
    , unsafe_allow_html=True
)