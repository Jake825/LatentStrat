"""LatentStrat Streamlit research workbench entrypoint."""

import streamlit as st

from latentstrat.app.ui import PAGES, configure_page, render_sidebar

configure_page()
render_sidebar()

navigation = st.navigation(
    {
        "Overview": [
            st.Page(PAGES["Workspace"], title="Workspace", icon=":material/home:", default=True),
            st.Page(PAGES["Data"], title="Data", icon=":material/table_view:"),
        ],
        "Understand": [
            st.Page(
                PAGES["Runs + Evaluation"],
                title="Runs + Evaluation",
                icon=":material/monitoring:",
            ),
            st.Page(PAGES["Model"], title="Model", icon=":material/model_training:"),
            st.Page(PAGES["Embeddings"], title="Embeddings", icon=":material/scatter_plot:"),
            st.Page(PAGES["Match"], title="Match", icon=":material/scoreboard:"),
        ],
    },
    position="sidebar",
)
navigation.run()
