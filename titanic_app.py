import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# ---------------------- App Settings ----------------------
st.set_page_config(
    page_title="🚢 Titanic Survival Explorer", 
    layout="wide",
    page_icon="🚢"
)

# ---------------------- Premium Custom Style ----------------------
st.markdown("""
    <style>
    :root {
        --primary: #2c3e50;
        --secondary: #3498db;
        --accent: #e74c3c;
        --light: #ecf0f1;
        --dark: #2c3e50;
    }
    
    .main {
        background-color: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    h1 {
        color: var(--primary);
        border-bottom: 2px solid var(--secondary);
        padding-bottom: 0.5rem;
    }
    
    h2 {
        color: var(--primary);
        margin-top: 1.5rem;
    }
    
    .stSelectbox, .stMultiselect, .stSlider {
        background-color: var(--light);
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    [data-testid="stMetricValue"] {
        color: var(--accent);
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: var(--dark);
        font-weight: 600;
    }
    
    .footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #eee;
        color: #7f8c8d;
        font-size: 0.9rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- Load Data ----------------------
@st.cache_data
def load_data():
    RAW_GITHUB_URL = "https://raw.githubusercontent.com/Rehmi-1/titanic-eda-app/main/train_cleaned.csv"
    try:
        df = pd.read_csv(RAW_GITHUB_URL)
        if df.empty or 'Survived' not in df.columns:
            st.error("❌ Invalid dataset structure")
            return None
        return df
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {str(e)}")
        return None

df = load_data()
if df is None:
    st.stop()

# ---------------------- Sidebar Filters ----------------------
st.sidebar.header("🔍 Filter Controls")
with st.sidebar.expander("Passenger Filters", expanded=True):
    sex_filter = st.multiselect("Gender", df['Sex'].unique(), default=df['Sex'].unique())
    pclass_filter = st.multiselect("Passenger Class", sorted(df['Pclass'].unique()), 
                                  default=sorted(df['Pclass'].unique()))
    embarked_filter = st.multiselect("Embarkation Port", df['Embarked'].unique(), 
                                   default=df['Embarked'].unique())
    
with st.sidebar.expander("Advanced Filters"):
    age_range = st.slider("Age Range", 
                         min_value=int(df['Age'].min()), 
                         max_value=int(df['Age'].max()),
                         value=(int(df['Age'].min()), int(df['Age'].max())))
    
    fare_range = st.slider("Fare Range ($)",
                          min_value=int(df['Fare'].min()),
                          max_value=int(df['Fare'].max()),
                          value=(int(df['Fare'].min()), int(df['Fare'].max())))
    
    family_filter = st.checkbox("Only passengers with family")

# Apply filters
df = df[
    (df['Sex'].isin(sex_filter)) &
    (df['Pclass'].isin(pclass_filter)) &
    (df['Embarked'].isin(embarked_filter)) &
    (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
    (df['Fare'] >= fare_range[0]) & (df['Fare'] <= fare_range[1])
]

if family_filter:
    df = df[(df['SibSp'] + df['Parch']) > 0]

# ---------------------- Fare Binning ----------------------
def fare_bin(fare):
    if fare <= 50: return "0–50"
    elif fare <= 100: return "51–100"
    elif fare <= 150: return "101–150"
    else: return "151+"

fare_bins = ["0–50", "51–100", "101–150", "151+"]
df['Fare_Bin'] = pd.Categorical(df['Fare'].apply(fare_bin), categories=fare_bins, ordered=True)

# ---------------------- Main Content ----------------------

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">

<style>
/* Keyframe for flipping animation */
@keyframes flipOnce {
  0% {
    transform: rotateY(90deg);
    opacity: 0;
  }
  60% {
    transform: rotateY(-10deg);
    opacity: 0.6;
  }
  100% {
    transform: rotateY(0);
    opacity: 1;
  }
}

/* Container for title bar flip */
.hero-flip {
    animation: flipOnce 1.5s ease-in-out 0.2s forwards;
    transform-style: preserve-3d;
    perspective: 1200px;
}
</style>

<div class="hero-flip" style="
    font-family: 'Orbitron', sans-serif;
    font-size: 2.6rem;
    text-align: center;
    color: #00cec9;
    margin-top: -20px;
    margin-bottom: 10px;
    text-shadow: 0 0 8px rgba(0, 206, 201, 0.4);
">
     <br>   
🧭 SurvivorLens: Titanic Data Explorer
</div>

<div class="hero-flip" style="
    position: relative;
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    padding: 1.5rem;
    border-radius: 12px;
    font-family: 'Inter', sans-serif;
    font-size: 1.05rem;
    line-height: 1.8;
    color: #ecf0f1;
    margin-bottom: 2rem;
    text-shadow: 0px 0px 5px rgba(255, 255, 255, 0.05);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    overflow: hidden;
">

<div style="max-width: 70%;">
    <span style="color: #00cec9; font-weight: 600; font-size: 1.3rem;">Uncover the hidden patterns of survival.</span><br><br>
    This interactive explorer lets you dive deep into<br>
    <span style="color: #81ecec; font-weight: 500;">passenger demographics</span>,<br>
    <span style="color: #fab1a0; font-weight: 500;">   journey classes</span>,<br>
    and <span style="color: #ffeaa7; font-weight: 500;">survival odds</span> all at your fingertips.
</div>
""", unsafe_allow_html=True)


# Key Metrics Flip Cards - Modern Enhanced
st.markdown("""
<style>
/* FLIP CARD CONTAINER */
.flip-card {
  background: transparent;
  width: 100%;
  height: 140px;
  perspective: 1200px;
}

/* INNER WRAPPER */
.flip-card-inner {
  position: relative;
  width: 100%;
  height: 100%;
  transition: transform 0.8s cubic-bezier(0.4, 0.2, 0.2, 1.1);
  transform-style: preserve-3d;
}

/* FLIP ON HOVER */
.flip-card:hover .flip-card-inner {
  transform: rotateY(180deg);
}

/* FRONT & BACK SIDE */
.flip-card-front, .flip-card-back {
  position: absolute;
  width: 100%;
  height: 100%;
  border-radius: 14px;
  padding: 1.2rem;
  box-shadow: 0 10px 25px rgba(0,0,0,0.2);
  backface-visibility: hidden;
  display: flex;
  flex-direction: column;
  justify-content: center;
  background: linear-gradient(135deg, #0f2027, #203a43);
  border: 1px solid rgba(255,255,255,0.15);
  transition: all 0.4s ease;
}

/* FRONT SIDE STYLING */
.flip-card-front {
  color: white;
  border: 1px solid rgba(255,255,255,0.08);
}

/* BACK SIDE */
.flip-card-back {
  transform: rotateY(180deg);
  color: white;
}

/* HOVER GLOW EFFECT */
.flip-card:hover .flip-card-front,
.flip-card:hover .flip-card-back {
  box-shadow: 0 0 20px rgba(85,239,196,0.4);
  border-color: rgba(85,239,196,0.5);
}

/* TEXT STYLES */
.metric-title {
  font-size: 0.95rem;
  color: rgba(255,255,255,0.65);
  font-weight: 500;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

.metric-sub {
  font-size: 0.8rem;
  color: rgba(255,255,255,0.4);
  margin-top: 0.4rem;
}

.metric-value {
  font-size: 2rem;
  font-weight: bold;
  letter-spacing: 1px;
  margin-top: 0.3rem;
}

.metric-survival { color: #55efc4; }
.metric-age { color: #a29bfe; }
.metric-fare { color: #ffeaa7; }
</style>
""", unsafe_allow_html=True)

cols = st.columns(4)

with cols[0]:
    st.markdown(f"""
    <div class="flip-card">
      <div class="flip-card-inner">
        <div class="flip-card-front">
          <div class="metric-title">TOTAL PASSENGERS</div>
          <div class="metric-sub">Hover to see total</div>
        </div>
        <div class="flip-card-back">
          <div class="metric-value">{len(df)}</div>
          <div class="metric-sub">aboard Titanic</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with cols[1]:
    st.markdown(f"""
    <div class="flip-card">
      <div class="flip-card-inner">
        <div class="flip-card-front">
          <div class="metric-title">SURVIVAL RATE</div>
          <div class="metric-sub">Hover to see rate</div>
        </div>
        <div class="flip-card-back">
          <div class="metric-value metric-survival">{df['Survived'].mean():.1%}</div>
          <div class="metric-sub">chance to survive</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with cols[2]:
    st.markdown(f"""
    <div class="flip-card">
      <div class="flip-card-inner">
        <div class="flip-card-front">
          <div class="metric-title">AVERAGE AGE</div>
          <div class="metric-sub">Hover to see avg</div>
        </div>
        <div class="flip-card-back">
          <div class="metric-value metric-age">{df['Age'].mean():.1f}</div>
          <div class="metric-sub">years old</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with cols[3]:
    st.markdown(f"""
    <div class="flip-card">
      <div class="flip-card-inner">
        <div class="flip-card-front">
          <div class="metric-title">AVERAGE FARE</div>
          <div class="metric-sub">Hover to see fare</div>
        </div>
        <div class="flip-card-back">
          <div class="metric-value metric-fare">${df['Fare'].mean():.2f}</div>
          <div class="metric-sub">per passenger</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------- Custom Styles ----------------------
st.markdown("""
<style>
/* Gradient divider */
.modern-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(169,220,247,0.3), transparent);
    margin: 1.5rem 0;
}

/* Metric card style */
.metric-card {
    background: rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 1.2rem;
    border: 1px solid rgba(255,255,255,0.1);
    transition: all 0.3s ease;
    margin-bottom: 1rem;
    color: white;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
}
.metric-title {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.8);
    margin-bottom: 0.5rem;
    font-weight: 500;
    letter-spacing: 0.4px;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 600;
    color: #55efc4;
}

/* Modern selectbox override */
div[data-baseweb="select"] {
    background-color: #0f2027 !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 12px !important;
    padding: 6px 12px !important;
    font-size: 1rem !important;
    color: white !important;
}
div[data-baseweb="select"] * {
    color: white !important;
}
ul[role="listbox"] {
    background-color: #0f2027 !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
}

/* Slider styling */
.stSlider > div > div > div {
    background-color: #ffffff;
}
.stSlider > div > div > div > div {
    background-color: #ff4d4d;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* ===================== GENERAL FORM ELEMENT FIXES ===================== */

/* Label colors for selectboxes and sliders */
.stSelectbox label,
.stSlider label {
    color: #2c3e50 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.3rem !important;
}

/* Selectbox text (selected value) */
div[data-baseweb="select"] span {
    color: #2c3e50 !important;
    font-weight: 500;
}

/* Dropdown options list */
ul[role="listbox"] {
    background-color: #ffffff !important;
    border: 1px solid #dee2e6 !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15) !important;
}
ul[role="listbox"] li {
    color: #2c3e50 !important;
    padding: 10px 14px !important;
    font-size: 0.95rem !important;
}
ul[role="listbox"] li:hover {
    background-color: #dfe6e9 !important;
    color: #2d3436 !important;
    border-radius: 6px !important;
}

/* Tags inside multiselect */
.stMultiSelect span[data-baseweb="tag"] {
    background-color: #3498db !important;
    color: white !important;
    border-radius: 8px;
    font-weight: 500;
}

/* ===================== SLIDER CUSTOMIZATION ===================== */

/* Track color */
.stSlider > div > div > div > div {
    background: #3498db !important; /* Active track */
    border-radius: 4px;
}

/* Inactive track */
.stSlider > div > div > div {
    background: #dee2e6 !important;
}

/* Slider ticks and number labels */
.stSlider span,
.stSlider label,
.stSlider .css-1dp5vir,
.stSlider .css-1c5rxus {
    color: #2c3e50 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

/* Handle */
.stSlider div[data-baseweb="slider"] > div > div {
    background-color: #2c3e50 !important;
    border: 3px solid #3498db !important;
}

/* ===================== GENERAL FORM SHAPE ===================== */

.stSelectbox,
.stSlider {
    background-color: #f8f9fa !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
    transition: all 0.2s ease-in-out;
}

.stSelectbox:hover,
.stSlider:hover {
    box-shadow: 0 0 0 3px rgba(52,152,219,0.2);
    border-color: #3498db !important;
}

/* Reduce padding around widget if in cards */
.block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ---------------------- Color Palettes ----------------------
COLOR_SCALE = ['#ff7675', '#74b9ff', '#55efc4', '#a29bfe', '#ffeaa7', '#fd79a8']
SURVIVAL_COLORS = {0: '#ff7675', 1: '#55efc4'}
GENDER_COLORS = {'male': '#74b9ff', 'female': '#fd79a8'}
CLASS_COLORS = {1: '#a29bfe', 2: '#74b9ff', 3: '#55efc4'}

# ---------------------- Visualization Selector ----------------------
plot_options = [
    "📈 Survival Rate by Fare",
    "👑 Survival by Class & Gender",
    "📊 Age Distribution by Survival",
    "📍 Age vs Fare Scatter",
    "🎭 Passenger Demographics",
    "📐 Correlation Heatmap",
    "🧮 Survival Probability"
]

# Selector Section
st.markdown('<div class="metric-card">', unsafe_allow_html=True)
st.markdown('<div class="metric-title">SELECT VISUALIZATION</div>', unsafe_allow_html=True)
plot_type = st.selectbox("", plot_options, index=0, label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# Title for the Plot
st.markdown(f"""
<div style="
    font-size: 1.4rem;
    font-weight: 600;
    margin: 1rem 0 0.5rem 0;
    color: #00cec9;">
    {plot_type[2:]}
</div>
<div class="modern-divider"></div>
""", unsafe_allow_html=True)

# Visualization Logic Container
with st.container():
    if plot_type == "📈 Survival Rate by Fare":
        fig = px.bar(df.groupby('Fare_Bin')['Survived'].mean().reset_index(),
                     x='Fare_Bin', y='Survived', color='Survived',
                     color_continuous_scale=[SURVIVAL_COLORS[0], SURVIVAL_COLORS[1]])
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "👑 Survival by Class & Gender":
        fig = px.bar(df.groupby(['Pclass', 'Sex'])['Survived'].mean().reset_index(),
                     x='Pclass', y='Survived', color='Sex', barmode='group',
                     color_discrete_map=GENDER_COLORS)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "📊 Age Distribution by Survival":
        fig = px.histogram(df, x='Age', color='Survived', nbins=20,
                           facet_col='Sex', barmode='overlay',
                           color_discrete_map=SURVIVAL_COLORS, opacity=0.8)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "📍 Age vs Fare Scatter":
        hover_data = ['Sex', 'Embarked']
        if 'Name' in df.columns:
            hover_data.insert(0, 'Name')
        fig = px.scatter(df, x='Age', y='Fare', color='Survived',
                         size='Pclass', hover_data=hover_data,
                         color_discrete_map=SURVIVAL_COLORS, size_max=15)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "🎭 Passenger Demographics":
     col1, col2 = st.columns(2)
    
     with col1:
        fig1 = px.pie(df, names='Sex',
                      color_discrete_sequence=[GENDER_COLORS['male'], GENDER_COLORS['female']])
        fig1.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)', 
            font_color='white', 
            showlegend=False
        )
        st.plotly_chart(fig1, use_container_width=True)

     with col2:
        fig2 = px.pie(df, names='Pclass',
                      color_discrete_sequence=[
                          CLASS_COLORS[1],
                          CLASS_COLORS[2],
                          CLASS_COLORS[3]
                      ])
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)', 
            font_color='white', 
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    elif plot_type == "📐 Correlation Heatmap":
        numeric = df[['Age', 'Fare', 'Pclass', 'SibSp', 'Parch', 'Survived']]
        corr = numeric.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax,
                    annot_kws={"size": 10}, cbar_kws={"shrink": 0.8})
        ax.set_facecolor('#0f2027')
        fig.patch.set_facecolor('#0f2027')
        ax.tick_params(colors='white')
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(ax.get_xticklabels(), color="white")
        plt.setp(ax.get_yticklabels(), color="white")
        st.pyplot(fig)

    elif plot_type == "🧮 Survival Probability":
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">SURVIVAL PROBABILITY ESTIMATOR</div>
            <div style="color: rgba(255,255,255,0.7); font-size: 0.95rem; margin-bottom: 1rem;">
                Estimate survival probability based on passenger characteristics
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            user_sex = st.selectbox("Gender", ["female", "male"], key='sex')
            user_age = st.slider("Age", 0, 100, 30, key='age')
        with col2:
            user_pclass = st.selectbox("Passenger Class", [1, 2, 3], key='class')
            user_fare = st.slider("Fare ($)", 0, 600, 50, key='fare')

        similar = df[
            (df['Sex'] == user_sex) &
            (df['Pclass'] == user_pclass) &
            (df['Age'].between(user_age-5, user_age+5)) &
            (df['Fare'].between(user_fare-20, user_fare+20))
        ]

        if len(similar) > 0:
            prob = similar['Survived'].mean()
            st.markdown(f"""
            <div class="metric-card" style="background: rgba(85, 239, 196, 0.1); border-color: rgba(85, 239, 196, 0.3);">
                <div class="metric-title">ESTIMATED SURVIVAL PROBABILITY</div>
                <div class="metric-value">{prob:.1%}</div>
                <div style="color: rgba(255,255,255,0.6); font-size: 0.9rem;">
                    Based on {len(similar)} similar passengers in the dataset
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="background: rgba(255, 118, 117, 0.1); border-color: rgba(255, 118, 117, 0.3);">
                <div style="color: #ff7675; font-weight: 600;">No similar passengers found</div>
                <div style="color: rgba(255,255,255,0.6); font-size: 0.9rem;">
                    Try adjusting your criteria to find matches
                </div>
            </div>
            """, unsafe_allow_html=True)


# Data Download
st.markdown("---")
st.download_button(
    label="📥 Download Filtered Data",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name='titanic_filtered.csv',
    mime='text/csv',
    use_container_width=True
)


st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    .footer-container {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, rgba(23,32,42,0.95), rgba(44,62,80,0.95));
        color: #ecf0f1;
        padding: 2.5rem 1rem;
        border-top-left-radius: 2rem;
        border-top-right-radius: 2rem;
        box-shadow: 0 -6px 25px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        margin-top: 3rem;
    }

    .footer-content {
        max-width: 1000px;
        margin: auto;
        text-align: center;
    }

    .footer-content h4 {
        font-size: 1.5rem;
        margin-bottom: 0.8rem;
        font-weight: 600;
        color: #ffeaa7;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    .footer-content p {
        font-size: 1rem;
        line-height: 1.6;
        color: #bdc3c7;
        margin: 0.4rem 0;
    }

    .footer-line {
        margin: 1.8rem auto;
        width: 100px;
        height: 3px;
        background: linear-gradient(90deg, transparent, #ffeaa7, transparent);
        border-radius: 3px;
    }

    .footer-credits {
        font-size: 1.1rem;
        color: #ffffff;
        margin-top: 1.5rem;
        padding: 0.8rem 1.5rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50px;
        display: inline-block;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }

    .footer-credits:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }

    .footer-credits strong {
        color: #ffeaa7;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    .heart {
        color: #ff6b6b;
        font-size: 1.2em;
        vertical-align: middle;
        animation: pulse 1.5s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.3); }
        100% { transform: scale(1); }
    }
    </style>

    <div class="footer-container">
        <div class="footer-content">
            <h4>🚢 Titanic Survival Explorer</h4>
            <div class="footer-line"></div>
            <p>This interactive dashboard visualizes insights from the famous Titanic passenger dataset,</p>
            <p>offering data exploration, survival analytics, and more — with an engaging UI.</p>
            <div class="footer-credits">
                Built with <span class="heart">❤ </span><strong>Talha Abdul Rauf</strong> & <strong>Abdul Rehman</strong>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)