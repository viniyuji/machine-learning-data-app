import streamlit as st
from modules import *

if __name__ == "__main__":
    st.set_page_config(page_title="Dashboard Product Search Corpus", page_icon=":bar_chart:", layout="wide")

    st.image("https://www.datarobot.com/wp-content/uploads/2022/06/huggingface-logo-light.png", width = 250)
    
    authenticator = BuildAuthenticator()
    name, authentication_status, username = authenticator.login('Login', 'main')

    if authentication_status:
        st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1946/1946429.png", width = 100)
        st.sidebar.title(f"Welcome {name}")
        selected_column = st.sidebar.selectbox("Select a column:", ["title", "text", "category", "material", "features"])
        authenticator.logout('Logout', 'sidebar')

        st.title("Dashboard Product Search Corpus")
        st.header("Word Embedding Chart:")

        with st.spinner('Wait until we classify the data...'):
            open_ai_client = build_open_ai_client()
            product_search_corpus = get_data()
            product_search_corpus = classify_data(data = product_search_corpus)

        with st.spinner('Wait until we build the model...'):
            model = build_model(product_search_corpus, selected_column)
            st.pyplot(tsne_plot(model))
            
    elif authentication_status == False:
        st.error('Username or password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')


