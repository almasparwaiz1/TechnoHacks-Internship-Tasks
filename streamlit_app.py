import os
import joblib
import random
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
VECTORIZER_PATH = r'tfidf_vectorizer.pkl'
MODEL_PATH = r'logistic_regression_sentiment_model.pkl'

# Aesthetic Palette (Warm Beige, Sage Green, Dark Brown)
BG_CREAM = "#FDFBF7"
SIDEBAR_CREAM = "#F4EFE6"
PRIMARY_SAGE = "#87A987"
SAGE_DARK = "#557155"
TEXT_BROWN = "#4A3B32"
CORAL_MUTED = "#C08A80"

SAMPLE_POSITIVE_REVIEWS = [
    "The doctors and nurses were incredibly attentive and kind during my stay.",
    "Excellent service, clean facilities, and minimal waiting time at the clinic.",
    "Highly professional medical staff. They answered all my questions patiently.",
    "The treatment was highly successful, and the discharge process was super smooth.",
    "Very clean environment and friendly front desk staff. I felt well cared for.",
    "The pediatric unit was wonderful with my daughter. Exceptional care!"
]

SAMPLE_NEGATIVE_REVIEWS = [
    "Long waiting hours even with a prior appointment. Extremely frustrating.",
    "The billing department was completely unhelpful and unorganized.",
    "The room was noisy, making it impossible to rest and recover properly.",
    "Staff seemed short-tempered and completely overwhelmed today.",
    "Food quality was terrible and cleanliness in the bathroom could be improved.",
    "Difficult to get a follow-up appointment. Communication needs work."
]

# ==========================================
# CORE PROCESSING LOGIC
# ==========================================
@st.cache_resource
def load_sentiment_artifacts(vectorizer_path: str, model_path: str):
    """Loads and caches the pre-trained vectorizer and model pipeline."""
    try:
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        return vectorizer, model
    except Exception as e:
        st.error(f"Critical Error: Failed to load pipeline artifacts. Details: {e}")
        return None, None

def process_sentiment_predictions(df: pd.DataFrame, vectorizer, model) -> pd.DataFrame:
    """Handles vectorization and model inference on input dataframe."""
    processed_df = df.copy()
    processed_df['text'] = processed_df['text'].fillna('')
    
    X_predict = vectorizer.transform(processed_df['text'])
    processed_df['sentiment_prediction'] = model.predict(X_predict)
    processed_df['sentiment_probability'] = model.predict_proba(X_predict)[:, 1]
    processed_df['sentiment_label'] = processed_df['sentiment_prediction'].map({0: 'Negative', 1: 'Positive'})
    return processed_df

def generate_wordcloud(text_data: str, color_palette: str):
    """Generates a styled matplotlib figure containing a theme-compliant word cloud."""
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color=BG_CREAM, 
        colormap=color_palette,
        max_words=100
    ).generate(text_data)
    
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG_CREAM)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

def generate_random_data(num_records: int, bias: str) -> pd.DataFrame:
    """Generates synthetic patient feedback reviews matching custom structural criteria."""
    data = []
    base_date = datetime.now() - timedelta(days=num_records // 2)
    
    for i in range(num_records):
        if bias == "Mostly Positive":
            pool = SAMPLE_POSITIVE_REVIEWS if random.random() < 0.8 else SAMPLE_NEGATIVE_REVIEWS
        elif bias == "Mostly Negative":
            pool = SAMPLE_NEGATIVE_REVIEWS if random.random() < 0.8 else SAMPLE_POSITIVE_REVIEWS
        else:
            pool = SAMPLE_POSITIVE_REVIEWS if random.random() < 0.5 else SAMPLE_NEGATIVE_REVIEWS
            
        review_text = random.choice(pool)
        review_date = base_date + timedelta(days=i // 2)
        
        data.append({
            "text": review_text,
            "date": review_date.strftime("%Y-%m-%d")
        })
        
    return pd.DataFrame(data)

# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================
def main():
    st.set_page_config(
        page_title="Patient Feedback Analytics",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Global Custom CSS Theme Injector
    st.markdown(f"""
        <style>
            .stApp {{
                background-color: {BG_CREAM};
                color: {TEXT_BROWN};
            }}
            h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, [data-testid="stMetricLabel"] {{
                color: {TEXT_BROWN} !important;
            }}
            [data-testid="stSidebar"] {{
                background-color: {SIDEBAR_CREAM} !important;
                border-right: 1px solid {PRIMARY_SAGE};
            }}
            .stButton>button {{
                background-color: {PRIMARY_SAGE} !important;
                color: white !important;
                border-radius: 6px;
                border: none;
                font-weight: 500;
            }}
            .stButton>button:hover {{
                background-color: {SAGE_DARK} !important;
            }}
            hr {{
                border-top: 1px solid {PRIMARY_SAGE} !important;
            }}
        </style>
    """, unsafe_allow_html=True)

    # Header section
    st.title("🏥 Patient Feedback Dashboard")
    st.markdown("Analyze system generated clinical feedback, evaluate sentiments, and monitor operational satisfaction metrics.")
    st.write("---")

    # Sidebar Controls Panel
    st.sidebar.header("Dashboard Settings")
    st.sidebar.markdown("Configure data parameters to run simulated inference tasks through the pipeline models.")
    
    sample_size = st.sidebar.slider("Number of Random Reviews", min_value=10, max_value=200, value=60, step=10)
    sentiment_bias = st.sidebar.selectbox("Simulated Sentiment Bias", ["Balanced", "Mostly Positive", "Mostly Negative"])
    
    # Core Model Validation Engine
    tfidf_vectorizer, lr_model = load_sentiment_artifacts(VECTORIZER_PATH, MODEL_PATH)
    if tfidf_vectorizer and lr_model:
        st.sidebar.success("✅ Models Operational")
    else:
        st.sidebar.error("❌ Machine Learning Models Offline.")
        return

    # Data Flow Generation & Pipeline Processing Matrix
    df = generate_random_data(sample_size, sentiment_bias)
    with st.spinner("Processing clinical sentiment predictions..."):
        df_analyzed = process_sentiment_predictions(df, tfidf_vectorizer, lr_model)

    # ----------------------------------------------------
    # METRICS SUMMARY PANEL
    # ----------------------------------------------------
    total_reviews = len(df_analyzed)
    pos_reviews = len(df_analyzed[df_analyzed['sentiment_prediction'] == 1])
    pos_percentage = (pos_reviews / total_reviews) * 100 if total_reviews > 0 else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Simulated Records", f"{total_reviews:,}")
    m2.metric("Positive Sentiments", f"{pos_reviews:,}")
    m3.metric("Patient Satisfaction Rate", f"{pos_percentage:.1f}%")
    st.write("###")

    # ----------------------------------------------------
    # INTERACTIVE ANALYTICS VISUALIZATIONS
    # ----------------------------------------------------
    col_chart, col_time = st.columns([4, 5])

    with col_chart:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df_analyzed['sentiment_label'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Volume']
        
        fig_pie = px.pie(
            sentiment_counts, 
            values='Volume', 
            names='Sentiment', 
            color='Sentiment',
            color_discrete_map={'Positive': PRIMARY_SAGE, 'Negative': CORAL_MUTED},
            hole=0.4
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color=TEXT_BROWN,
            margin=dict(l=10, r=10, t=10, b=10), 
            height=280
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_time:
        st.subheader("Longitudinal Sentiment Trends")
        df_analyzed['date'] = pd.to_datetime(df_analyzed['date'])
        time_df = df_analyzed.groupby(['date', 'sentiment_label']).size().unstack(fill_value=0).reset_index()
        
        fig_trend = px.line(
            time_df, 
            x='date', 
            y=list(time_df.columns[1:]),
            color_discrete_map={'Positive': PRIMARY_SAGE, 'Negative': CORAL_MUTED},
            labels={'value': 'Review Count', 'date': 'Timeline', 'variable': 'Sentiment'}
        )
        fig_trend.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color=TEXT_BROWN,
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#EFEAE2')
        )
        fig_trend.update_traces(line=dict(width=3, shape='spline'))
        st.plotly_chart(fig_trend, use_container_width=True)

    # ----------------------------------------------------
    # TEXTUAL PROCESSING (WORD CLOUDS)
    # ----------------------------------------------------
    st.write("---")
    st.subheader("Key Unstructured Sentiment Themes")
    cloud_pos_col, cloud_neg_col = st.columns(2)

    positive_text = " ".join(df_analyzed[df_analyzed['sentiment_prediction'] == 1]['text'])
    negative_text = " ".join(df_analyzed[df_analyzed['sentiment_prediction'] == 0]['text'])

    with cloud_pos_col:
        st.markdown("**Positive Highlights**")
        if positive_text.strip():
            fig_pos = generate_wordcloud(positive_text, 'summer')
            st.pyplot(fig_pos)
            plt.close(fig_pos)
        else:
            st.caption("No positive feedback found in current batch parameters.")

    with cloud_neg_col:
        st.markdown("**Areas for Improvement**")
        if negative_text.strip():
            fig_neg = generate_wordcloud(negative_text, 'copper')
            st.pyplot(fig_neg)
            plt.close(fig_neg)
        else:
            st.caption("No negative feedback found in current batch parameters.")

    # ----------------------------------------------------
    # RAW REVIEWS INSPECTOR DATA MATRIX
    # ----------------------------------------------------
    st.write("---")
    st.subheader("Pipeline Input/Output Records Audit")
    st.dataframe(
        df_analyzed[['date', 'text', 'sentiment_label', 'sentiment_probability']], 
        use_container_width=True
    )

if __name__ == '__main__':
    main()
