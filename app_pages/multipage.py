import streamlit as st 


class MultiPage:
    """
    A class to create a multi-page Streamlit application.
    """

    def __init__(self, app_name):
        self.pages = []
        self.app_name = app_name

        st.set_page_config(
            page_title=self.app_name,
            page_icon="🏦",
        )

    def add_page(self, title, func):
        """
        Add a new page to the application.
        """
        self.pages.append({"title": title, "function": func})

    def run(self):
        """
        Run the application.
        """
        st.title(self.app_name)
        st.divider()
        page = st.sidebar.radio(
            "Menu",
            self.pages,
            format_func=lambda page: page["title"])
        page["function"]()