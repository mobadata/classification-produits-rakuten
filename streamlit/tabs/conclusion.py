import streamlit as st

def renderConclusion():
    st.title('Conclusion')
    st.divider()
    
    st.header('Difficultés rencontrées')
    st.markdown("""
    - **Preprocessing essentiel** : L'ordre des étapes (zoom images, rééquilibrage classes) 
    impacte directement les performances.
    
    - **Ressources GPU limitées** : Même avec H100 sur Colab, entraînement long 
    (~5h images, ~2-3h texte).
    
    - **Gestion Drive/SSD** : Copie images sur SSD local (10-20 min) indispensable 
    pour éviter 10x ralentissement.
    
    - **Résilience aux interruptions** : Système de checkpoints nécessaire pour 
    reprendre l'entraînement.
    
    - **Compatibilité Keras 3** : Adaptation des métriques F1 via callbacks au lieu 
    de métriques natives.
    """)

    st.header('Résultats Obtenus')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Modèles Image")
        st.markdown("""
        - LeNet (baseline) : **F1 = 0.31**
        - VGG16 : **F1 = 0.60** (+93%)
        - **EfficientNetB3 : F1 = 0.66** (+113%)
        
        *Meilleure que l'équipe originale grâce à images 300×300 et architecture moderne*
        """)
    
    with col2:
        st.subheader("📝 Modèle Texte")
        st.markdown("""
        **CamemBERT fine-tuné :**
        - **F1 weighted : 0.8935**
        - **F1 macro : 0.8796**
        - **Accuracy : 0.8940**
        
        *Performance excellente sur descriptions produits*
        """)
    
    st.subheader("🔗 Fusion Multimodale")
    st.markdown("""
    Architecture complète combinant EfficientNetB3 (image) + CamemBERT (texte) 
    pour exploiter la complémentarité des modalités visuelles et textuelles.
    """)

    st.header('Pistes d\'Amélioration')
    st.markdown("""
    **Images** : Ensemble models, TTA, Vision Transformers, EfficientNetB4/B5
    
    **Texte** : Modèles plus récents, enrichissement métadonnées, data augmentation
    
    **Fusion** : Late fusion optimisée (stacking), early fusion (embeddings), 
    attention cross-modale, modèles natifs (CLIP, ViLT)
    """)