import streamlit as st
from streamlit_option_menu import option_menu

def renderIntroduction():
    #Header
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    with c1:
        st.title('Présentation du projet')
        st.subheader('*Rakuten France Multimodal Product Data Classification*')
    with c2:
        st.image('assets/rakuten.png', width=200)
    with c3:
        st.image('assets/mines.png', width=210)
    with c4:
        st.image('assets/datascientest.png', width=100)
    st.divider()        
    
    option = option_menu(None, ["Membres du projet", 'Objectif'], 
        icons=['people', "rocket-takeoff"], 
        menu_icon="cast", default_index=0, orientation="horizontal")
    
    if option == 'Objectif':
        st.header('Objectif')
        st.markdown("""
        L'objectif du projet est de cataloguer des produits selon un code type désignant le produit.
        La prédiction du type doit se faire à partir de données textuelles 
        (désignation et description du produit) ainsi que de données visuelles (image du produit).

        Étant réalisé dans le cadre de la formation Datascientest, 
        c'était l'opportunité pour nous de découvrir et mettre en application 
        des techniques de machine learning avancées telles que:

        - Computer vision
        - Réseaux de neurones convolutifs
        - NLP
        - Modèles multimodaux
        - Deep learning
        """)
    
    if option == 'Membres du projet':
        st.header('Membres du projet')
        st.subheader('_Promotion sep25_alt1_mle_')

        with st.container():
            col1,col2,col3=st.columns(3)
            with col1:
                st.image('assets/Picture1.jpg', width=225)
                st.header('Moussa BA')
                st.markdown(
                    """
                    - Ingénieur en IA 
                    - LLMs & Automatisation
                    - [Linkedin](https://www.linkedin.com/in/moussa-ba-615a901a9/)
                    - [Github](https://github.com/mobadata)
                    """
                    )
            with col2:
                st.image('assets/Picture2.png', width=225)
                st.header('Ben djabir Chipinda')
                st.markdown(
                    """
                    - AI Engineer
                    - Enterprise GenAI & LLM Systems | RAG • Multi-Agent AI • MLOps | Azure • AWS • GCP
                    - [Linkedin](https://www.linkedin.com/in/ben-djabir-chipinda-57b020223/?originalSubdomain=fr)
                    """
                    )
            with col3:
                st.image('assets/Picture3.png', width=225)
                st.header('NEDJAI Azeddine')
                st.markdown(
                    """
                    - Machine Learning Engineer | MLOps
                    - Traitement de signal et de l'image | Computer Vision | Deep Learning
                    - [Linkedin](https://www.linkedin.com/in/nedjai-azzedine/)
                    - [Github](https://github.com/AzzedineNed)
                    """
                    )