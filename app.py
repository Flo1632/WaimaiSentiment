import streamlit as st
import joblib
import jieba
# Übersetzer
from deep_translator import GoogleTranslator

# Seitenkonfiguartion
st.set_page_config(page_title="Waimai Analyzer (Chinesische Essensrezensionen)", page_icon="🥡")

# Modell mit Caching laden
@st.cache_resource
def load_model():
    model = joblib.load("model/waimai_model.pkl")
    return model

# Text wie im Training vorbereiten
def prepare_text(text):
    tokens = jieba.lcut(str(text))
    return ' '.join(tokens)

# UI Aufbau

st.title("Waimai Analyzer (Chinesische Essensrezensionen)")
st.markdown("Dieses Tool analysiert chinesische Bewertungen (auf der Platform Waimai) und erkennt, ob die Rezensionen negativ oder positiv sind.")

# Model im Hintegrund laden
model = load_model()

user_input = st.text_input("Bitte Rezension auf Chinesisch oder einer anderen Sprache eingeben und übersetzen lassen:", "Das Essen war kalt und der Fahrer unhöflich.")
# Checkbox für Übersetzung
translate_option = st.checkbox("Text automatisch ins Chinesische übersetzen?", value=True)

if st.button('Analysieren'):
    if user_input:

        # --- Übersetzen (falls gewünscht) ---
        final_text = user_input

        if translate_option:
            with st.spinner('Übersetze...'):
                try:
                    # source='auto' erkennt die Sprache automatisch
                    # target='zh-CN' ist vereinfachtes Chinesisch (für Mainland China)
                    translator = GoogleTranslator(source='auto', target='zh-CN')
                    translated_text = translator.translate(user_input)

                    st.info(f"🔤 Übersetzung: {translated_text}")
                    final_text = translated_text  # Wir arbeiten mit dem chinesischen Text weiter
                except Exception as e:
                    st.error(f"Fehler bei der Übersetzung: {e}")

        # Was aktuell passiert, Visualisierung für den Nutzer
        st.subheader("1. Wie der Computer den Text sieht (Jieba Segmentation):")
        processed_input = prepare_text(final_text)
        tokens = jieba.lcut(final_text)
        st.write(tokens)

        # Vorhersage machen
        prediction = model.predict([processed_input])[0]
        probability = model.predict_proba([processed_input])[0]

        # Ergebnis anzeigen
        st.subheader("2. Ergebnis:")

        # Wahrscheinlichkeiten für Negativ (Index 0) und Positiv (Index 1)
        prob_neg = probability[0]
        prob_pos = probability[1]

        if prediction == 1:
            st.success(f"Positiv! (Zu {prob_pos:.1%} sicher)")
            st.balloons()  # visueller Effekt
        else:
            st.error(f"Negativ! (Zu {prob_neg:.1%} sicher)")

            # Balkendiagramm für die Details
            st.write("Detaillierte Wahrscheinlichkeit:")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Positiv 😋", f"{prob_pos:.2%}")
            with col2:
                st.metric("Negativ 😡", f"{prob_neg:.2%}")

    else:
        st.warning("Bitte gib erst einen Text ein.")