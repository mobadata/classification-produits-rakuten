import streamlit as st
from streamlit_option_menu import option_menu

def renderModelisation():
    st.title('Modélisation')
    st.divider()

    option = option_menu(None, ["Texte", 'Images', 'Fusion'], 
        icons=['chat-text', "images", "file-richtext"], 
        menu_icon="cast", default_index=0, orientation="horizontal")
    
    if option == 'Texte':
        st.header('Modèle de classification de texte')
        st.subheader("*Étude en 2 phases*")
        with st.expander('Phase 1 : Modèles Classiques'):
            st.header('Modèles Classiques')
            st.markdown("""
            Application de techniques statistiques et d'apprentissage machine traditionnel pour la 
            classification de texte, avec une préparation spécifique du texte.

            1. Préparation du Texte
                - Filtrage stop Words
                - SnowballStemmer
                - TF-IDF
                - CBOW
                - Skip Gram
                - FastText
                - Tokenisé
     
            2. Modèles testés : 
                - Random Forest
                - Régression Logistique
                - SVM (Linear)
                - Multinomial Naive Bayes
            3. Performance 
                Accuracy: Varie de 0,69 à 0,78
            """)

            st.subheader("Conclusion")
            st.markdown("""
            Importance du préprocessing et des paramètres d'entrainement sur les résultats
            Meilleur score obtenu avec SVM et Filtrage stop words + SnowballStemmer + TF-IDF
            """)
            col1, col2 = st.columns([2, 3])
            with col1:
                st.subheader('Rapport de classification')
                st.image('assets/cf-svm-10.PNG', width=400)
            with col2:
                st.subheader('Matrice de confusion')
                st.image('assets/confusion-matrix-svm-10.png', width=600)
             
        with st.expander('Phase 2 : Transfert Learning'):
            st.header('Transfert Learning')
            st.markdown("""
            Utilisation de modèles pré-entraînés adaptés à de nouvelles tâches, 
            permettant d'exploiter des connaissances linguistiques complexes acquises sur de vastes corpus.

            1. Préparation du Texte
                - Tokeniser
                - encoder avec CamemBERT
     
            2. Modèles testés : 
                - CamemBERT

            3. Performance 
                - Accuracy: Jusqu'à 0,89
            """)

            st.subheader("Conclusion")
            st.markdown("""
            L'utilisation de transfert learning permet d'obtenir des résultats significativement meilleurs, 
            malgrès le peu d'ajustements sur les paramètres d'entraînement.
            """)
            col1, col2 = st.columns([2, 3])
            with col1:
                st.subheader('Résumé')
                st.image('assets/cf-CamemBERT-28.png', width=400)
            with col2:
                st.subheader('Matrice de confusion')
                st.image('assets/cmcammebert.jpeg', width=600)
       
    if option == 'Images':
            st.header("Modèles d'images")
            st.subheader("*Trois architectures CNN testées*")
            
            with st.expander('Modèle d\'image LeNet (Baseline)'):
                st.header('Modèle LeNet - Baseline CNN')
                st.markdown("""
                Architecture CNN simple utilisée comme baseline pour établir les performances minimales.
                
                **Architecture:**
                - 2 couches Conv2D (6 et 16 filtres)
                - AveragePooling2D
                - Dense layers (120, 84, 27)
                
                **Configuration:**
                - Images: 256×256 pixels
                - Batch size: 128
                - Époques: 20 (early stopping)
                - Optimizer: Adam (LR=0.001, decay 0.9 tous les 3 epochs)
                - Data augmentation: rotation, shift, shear, zoom, horizontal flip
                
                **Données:**
                - Dataset rééquilibré (undersampling + oversampling)
                - Images zoomées (innerimageratio ≤ 0.8)
                - Train: 109,566 | Val: 8,492 | Test: 8,492
                """)

                st.subheader("Résultats")
                st.markdown("""
                **Performance:**
                - **F1-Score: 0.31**
                - Accuracy: 0.31
                
                Les performances obtenues avec LeNet confirment qu'une architecture simple 
                ne suffit pas pour cette tâche complexe de classification multiclasse (27 catégories).
                """)

            with st.expander('Modèle VGG16 Transfer Learning'):
                st.header('Modèle VGG16 avec Transfer Learning')
                st.markdown("""
                Pour améliorer significativement les performances, nous avons utilisé le transfer learning 
                avec VGG16 pré-entraîné sur ImageNet.

                **Stratégie d'entraînement en 2 phases:**
                
                **Phase 1: VGG16 Gelé (14 époques)**
                - Toutes les couches VGG16 gelées
                - Entraînement uniquement du classifier head
                - LR: 0.001
                - Val F1: ~0.56
                
                **Phase 2: Fine-tuning (10 époques)**
                - Dégel des 4 dernières couches VGG16
                - Fine-tuning avec LR réduit: 0.0001
                - Val F1: 0.60 → Test F1: **0.5986**
                
                **Architecture:**
                - Base: VGG16 (ImageNet weights)
                - Images: 224×224 (taille native VGG16)
                - Batch size: 64
                - Dropout: 0.2
                - Classifier: GlobalAvgPool → Dense(1024) → Dropout → Dense(512) → Dropout → Dense(27)
                """)

                st.subheader("Résultats")
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown("""
                        **Performance:**
                        - **Test F1-Score: 0.5986**
                        - Test Accuracy: 0.5909
                        - Amélioration vs LeNet: **+93%**
                        
                        #### Rapport de classification
                    """)
                    st.image("assets/crvgg16.PNG", width=400)
                
                with c2:
                    st.markdown("""
                        #### Matrice de confusion
                    """)
                    st.image("assets/mcvgg16.png")

            with st.expander('Modèle EfficientNetB3 (Meilleur Résultat)'):
                st.header('Modèle EfficientNetB3 Transfer Learning')
                st.markdown("""
                Pour dépasser les performances de VGG16, nous avons implémenté EfficientNetB3, 
                une architecture moderne (2019) utilisant le compound scaling.
                
                **Améliorations clés vs VGG16:**
                - **Images 300×300** (vs 224×224) → plus de détails produit
                - **Dropout 0.3** (vs 0.2) → meilleure régularisation
                - **Brightness augmentation** → robustesse aux variations d'éclairage
                - **Architecture plus efficace** (12M params vs 138M pour VGG16)
                
                **Stratégie d'entraînement en 2 phases:**
                
                **Phase 1: EfficientNet Gelé (~12-15 époques)**
                - Toutes les couches EfficientNet gelées
                - Entraînement classifier head
                - LR: 0.001
                - Val F1: 0.5901
                
                **Phase 2: Fine-tuning (~6-8 époques)**
                - Dégel des **20 dernières couches** (vs 4 pour VGG16)
                - Fine-tuning avec LR: 0.0001
                - Val F1: 0.6439 → Test F1: **0.6552**
                
                **Configuration:**
                - Base: EfficientNetB3 (ImageNet weights)
                - Images: 300×300 pixels
                - Batch size: 32
                - Dropout: 0.3
                - Early stopping: monitor val_loss, patience=5
                """)

                st.subheader("Résultats Finaux")
                st.markdown("""
                **Performance:**
                - **Test F1-Score: 0.6552** 🎯
                - Test Accuracy: 0.6537
                - Amélioration vs LeNet: **+111%**
                - Amélioration vs VGG16: **+9.2%**
                
                **Classes les plus performantes (F1 > 0.80):**
                - Cartes collectionnables: 0.93
                - Mobilier de chambre: 0.83
                - Équipement de piscine: 0.83
                
                **Généralisation excellente:** Test F1 > Val F1 (pas d'overfitting)
                """)
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown("""
                        #### Rapport de classification
                    """)
                    st.image("assets/crenb3.PNG", width=400)
                
                with c2:
                    st.markdown("""
                        #### Matrice de confusion
                    """)
                    st.image("assets/cmenb3.png")

            # Comparison summary
            st.subheader("📊 Comparaison des Modèles")
            comparison_data = {
                'Modèle': ['LeNet', 'VGG16', 'EfficientNetB3'],
                'F1-Score': [0.31, 0.5986, 0.6552],
                'Accuracy': [0.31, 0.5909, 0.6537],
                'Images': ['256×256', '224×224', '300×300'],
                'Params': ['~60K', '~138M', '~12M'],
                'Temps (heures)': ['~6', '~4-5', '~5']
            }
            st.table(comparison_data)
            
            st.success("✅ **Meilleur modèle: EfficientNetB3** avec F1=0.6552 (+9.2% vs VGG16)")
    if option == 'Fusion':
        st.header("Fusion des modèles texte et image")

        st.markdown("""
        L'objectif de cette étape est de combiner les prédictions de deux modèles spécialisés
        afin d'améliorer la performance globale de classification des produits.

        Les deux modèles utilisés sont :

        • **EfficientNetB3** : modèle de vision par ordinateur utilisé pour l'analyse
        des images de produits.

        • **CamemBERT** : modèle de traitement du langage naturel (NLP) utilisé pour
        analyser les descriptions textuelles des produits.

        Ces deux modèles capturent des **informations complémentaires** :
        - les caractéristiques visuelles via l'image
        - les informations sémantiques via le texte

        La fusion permet donc d'exploiter ces deux sources d'information
        afin d'améliorer la précision des prédictions.
        """)

        st.subheader("Stratégies de fusion testées")

        st.markdown("""
        Deux méthodes de fusion ont été expérimentées :

        **1. Fusion intelligente simple**

        Cette approche sélectionne la prédiction du modèle le plus confiant
        entre le modèle image et le modèle texte.

        **2. Fusion probabiliste pondérée**

        Les probabilités des deux modèles sont combinées à l'aide d'une
        moyenne pondérée :

        P_final = w_image × P_image + w_texte × P_texte

        Cette méthode permet de contrôler l'influence relative
        de chaque modèle dans la décision finale.
        """)

        st.subheader("Architecture de la fusion")

        st.image("assets/fusion-architecture.jpeg", width=700)

        