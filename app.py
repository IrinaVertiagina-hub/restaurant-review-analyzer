import streamlit as st
import pickle
import pandas as pd
from scipy.sparse import hstack
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка страницы
st.set_page_config(
    page_title="Restaurant Review Analyzer",
    page_icon="🍽️",
    layout="wide"
)

# Загружаем модели (кэшируем, чтобы загружать один раз)
@st.cache_resource
def load_models():
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open('models/xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/aspect_keywords.pkl', 'rb') as f:
        aspect_keywords = pickle.load(f)
    
    with open('models/model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    return vectorizer, model, aspect_keywords, metadata

# Функция для извлечения дополнительных признаков
def extract_features(text):
    features = {
        'text_length': len(text),
        'word_count': len(text.split()),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'capital_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
        'avg_word_length': len(text) / len(text.split()) if len(text.split()) > 0 else 0
    }
    return pd.DataFrame([features])

# Функция для предсказания общего sentiment
def predict_sentiment(text, vectorizer, model):
    # TF-IDF
    text_vector = vectorizer.transform([text])
    
    # Дополнительные признаки
    extra_features = extract_features(text)
    
    # Объединяем
    combined = hstack([text_vector, extra_features.values])
    
    # Предсказание
    prediction = model.predict(combined)[0]
    probabilities = model.predict_proba(combined)[0]
    
    return prediction, probabilities

# Функция для извлечения предложений с аспектом
# def extract_aspect_sentences(text, aspect_words):
#     sentences = re.split(r'[.!?]+', text.lower())
#     relevant_sentences = []
#     for sentence in sentences:
#         if any(word in sentence for word in aspect_words):
#             relevant_sentences.append(sentence.strip())
#     return relevant_sentences
def extract_aspect_sentences(text, aspect_words):
    """Извлекает контекст вокруг аспекта, избегая противоположных мнений"""
    text_lower = text.lower()
    sentences = re.split(r'[.!?]+', text_lower)
    relevant = []
    
    # Слова-разделители (but, however, although)
    separators = ['but', 'however', 'although', 'though', 'yet']
    
    for sentence in sentences:
        words = sentence.split()
        
        for i, word in enumerate(words):
            if any(aspect_word in word for aspect_word in aspect_words):
                # Найдём ближайший separator ПЕРЕД аспектом
                separator_idx = -1
                for j in range(i-1, -1, -1):
                    if words[j] in separators:
                        separator_idx = j
                        break
                
                # Начинаем ПОСЛЕ separator (или с начала если separator нет)
                start = max(0, separator_idx + 1)
                
                # Берём до конца или до следующего separator
                end = len(words)
                for j in range(i+1, len(words)):
                    if words[j] in separators:
                        end = j
                        break
                
                context = ' '.join(words[start:end])
                relevant.append(context.strip())
                break
    
    return relevant

# Функция для анализа sentiment конкретного аспекта (из Jupyter cell 19)
def analyze_aspect_sentiment(text, aspect_words, vectorizer, model):
    """Анализирует sentiment для конкретного аспекта"""
    sentences = extract_aspect_sentences(text, aspect_words)
    
    if not sentences:
        return None  # аспект не упоминается
    
    # Объединяем все предложения об аспекте
    aspect_text = ' '.join(sentences)
    
    # 1. TF-IDF вектор
    aspect_tfidf = vectorizer.transform([aspect_text])
    
    # 2. Дополнительные признаки
    extra_features = pd.DataFrame({
        'text_length': [len(aspect_text)],
        'word_count': [len(aspect_text.split())],
        'exclamation_count': [aspect_text.count('!')],
        'question_count': [aspect_text.count('?')],
        'capital_ratio': [sum(1 for c in aspect_text if c.isupper()) / len(aspect_text) if len(aspect_text) > 0 else 0],
        'avg_word_length': [len(aspect_text) / len(aspect_text.split()) if len(aspect_text.split()) > 0 else 0]
    })
    
    # 3. Объединяем
    aspect_vector = hstack([aspect_tfidf, extra_features.values])
    
    # Предсказываем sentiment
    sentiment = model.predict(aspect_vector)[0]
    sentiment_proba = model.predict_proba(aspect_vector)[0]
    
    return {
        'sentiment': sentiment,
        'confidence': sentiment_proba[sentiment],
        'text_sample': sentences[0] if sentences else ''
    }

# Функция для анализа всех аспектов
def analyze_aspects(text, vectorizer, model, aspect_keywords):
    """Анализирует все аспекты отзыва"""
    results = {}
    
    for aspect_name, keywords in aspect_keywords.items():
        result = analyze_aspect_sentiment(text, keywords, vectorizer, model)
        
        if result:
            results[aspect_name] = {
                'sentiment': 'Positive' if result['sentiment'] == 1 else 'Negative',
                'confidence': result['confidence'],
                'sample': result['text_sample']
            }
    
    return results

# Загружаем модели
vectorizer, model, aspect_keywords, metadata = load_models()

# === UI ===

st.title("🍽️ Restaurant Review Analyzer")
st.markdown("### Analyze restaurant reviews with AI-powered sentiment analysis")

# Sidebar с информацией о модели
with st.sidebar:
    st.header("📊 Model Information")
    st.metric("Accuracy", f"{metadata['baseline_accuracy']*100:.1f}%")
    st.metric("Training Samples", f"{metadata['training_samples']:,}")
    st.metric("Features", metadata['features'])
    st.markdown("---")
    st.markdown("**Aspects Analyzed:**")
    st.markdown("🍕 Food • 👥 Service • 🏠 Ambiance • 💰 Price")

# Главная секция
tab1, tab2 = st.tabs(["📝 Analyze Review", "ℹ️ About"])

with tab1:
    st.markdown("### Enter a restaurant review:")
    
    # Примеры отзывов
    example_reviews = {
        "Positive Example": "The food was absolutely delicious! Great service and wonderful atmosphere. Highly recommend!",
        "Negative Example": "Terrible service, waited 45 minutes. Food was cold and overpriced. Never coming back.",
        "Mixed Example": "Food was amazing but service was really slow. Prices are reasonable though."
    }
    
    selected_example = st.selectbox("Or select an example:", ["Custom"] + list(example_reviews.keys()))
    
    if selected_example == "Custom":
        user_review = st.text_area("Review text:", height=150, placeholder="Type or paste a restaurant review here...")
    else:
        user_review = st.text_area("Review text:", value=example_reviews[selected_example], height=150)
    
    if st.button("🔍 Analyze Review", type="primary"):
        if user_review:
            with st.spinner("Analyzing..."):
                # Общий sentiment
                prediction, probabilities = predict_sentiment(user_review, vectorizer, model)
                
                # Результат
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### Overall Sentiment")
                    if prediction == 1:
                        st.success(f"✅ **POSITIVE** ({probabilities[1]*100:.1f}% confidence)")
                    else:
                        st.error(f"❌ **NEGATIVE** ({probabilities[0]*100:.1f}% confidence)")
                
                with col2:
                    # Круговая диаграмма
                    fig, ax = plt.subplots(figsize=(3, 3))
                    colors = ['#90EE90' if prediction == 1 else '#FFB6C6', 
                             '#FFB6C6' if prediction == 1 else '#90EE90']
                    ax.pie(probabilities, labels=['Negative', 'Positive'], 
                           autopct='%1.1f%%', colors=colors, startangle=90)
                    ax.set_title('Confidence')
                    st.pyplot(fig)
                
                # Анализ по аспектам
                st.markdown("---")
                st.markdown("### 📊 Aspect-Based Analysis")
                
                aspect_results = analyze_aspects(user_review, vectorizer, model, aspect_keywords)
                
                if aspect_results:
                    cols = st.columns(len(aspect_results))
                    
                    for idx, (aspect, result) in enumerate(aspect_results.items()):
                        with cols[idx]:
                            emoji = {'food': '🍕', 'service': '👥', 'ambiance': '🏠', 'price': '💰'}
                            st.markdown(f"**{emoji.get(aspect, '📌')} {aspect.capitalize()}**")
                            
                            if result['sentiment'] == 'Positive':
                                st.success(f"✅ Positive")
                            else:
                                st.error(f"❌ Negative")
                            
                            st.caption(f"{result['confidence']*100:.0f}% confident")
                            with st.expander("View excerpt"):
                                st.write(f"_{result['sample'][:100]}..._")
                else:
                    st.info("No specific aspects detected in this review.")
        else:
            st.warning("⚠️ Please enter a review to analyze.")

with tab2:
    st.markdown("""
    ### About This Project
    
    This AI-powered tool analyzes restaurant reviews using Machine Learning to provide:
    
    - **Overall Sentiment**: Determines if a review is positive or negative
    - **Aspect-Based Analysis**: Breaks down sentiment by Food, Service, Ambiance, and Price
    - **Confidence Scores**: Shows how certain the model is about its predictions
    
    #### Model Performance
    - **Accuracy**: 95.5% on 17,728 test reviews
    - **Training Data**: 70,910 Yelp restaurant reviews
    - **Algorithm**: Logistic Regression with TF-IDF features
    
    #### Key Insights from Analysis
    - 📊 Price and Service are the most common complaint areas (22% and 21.7% negative)
    - 🏆 Ambiance receives the most positive feedback (88.4% positive)
    - 🍕 Food quality has 82% positive sentiment overall
    
    #### Model Limitations
    - Works best with complete, detailed reviews (50+ words)
    - May struggle with very short phrases or strong slang terms
    - Aspect analysis extracts relevant sentences but may be less accurate for brief mentions
    
    ---
    *Built with Python, scikit-learn, and Streamlit*
    """)

# Footer
st.markdown("---")
st.markdown("*Restaurant Review Analyzer • Built by Irina Vertiagina*")



