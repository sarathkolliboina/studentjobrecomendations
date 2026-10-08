import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Career Guidance Portal",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark-themed dashboard CSS with vibrant Control Center & crisp contrast
st.markdown("""
    <style>
    /* Global App & Sidebar Dark Background */
    .stApp, [data-testid="stSidebar"] {
        background-color: #0B0F19 !important;
        color: #F8FAFC !important;
    }
    
    /* Control Center (Sidebar) Styling */
    [data-testid="stSidebar"] {
        border-right: 1px solid #1E293B !important;
        padding-top: 1rem;
    }
    
    .sidebar-header {
        font-size: 1.4rem !important;
        font-weight: 800 !important;
        color: #38BDF8 !important;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .sidebar-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
    }
    
    .sys-info-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #1F2937;
        font-size: 0.9rem;
    }
    .sys-info-item:last-child {
        border-bottom: none;
    }
    .sys-label {
        color: #94A3B8;
        font-weight: 600;
    }
    .sys-pill {
        background: #1E293B;
        color: #38BDF8;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.8rem;
        border: 1px solid #334155;
    }
    
    /* Header Container */
    .title-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #1E293B;
        padding: 28px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #94A3B8;
        font-weight: 500;
    }
    
    /* Input Labels & Text Boxes */
    .stTextArea label {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
        margin-bottom: 8px !important;
    }
    .stTextArea textarea {
        background-color: #111827 !important;
        color: #F8FAFC !important;
        border: 2px solid #1E293B !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.3) !important;
    }
    
    /* Executive Metric Cards */
    .metric-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-bottom: 10px;
    }
    .metric-val-success {
        font-size: 1.7rem;
        font-weight: 800;
        color: #34D399;
    }
    .metric-val-danger {
        font-size: 1.7rem;
        font-weight: 800;
        color: #F87171;
    }
    .metric-val-blue {
        font-size: 2.1rem;
        font-weight: 800;
        color: #38BDF8;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #F1F5F9;
        border-bottom: 2px solid #1E293B;
        padding-bottom: 8px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    /* Skill Badges */
    .badge-match {
        background-color: rgba(52, 211, 153, 0.12);
        color: #34D399;
        border: 1px solid #059669;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        margin: 4px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .badge-missing {
        background-color: rgba(248, 113, 113, 0.12);
        color: #F87171;
        border: 1px solid #DC2626;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        margin: 4px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    
    /* Alternative Recommendations Cards */
    .rec-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-left: 4px solid #38BDF8;
        padding: 16px;
        margin-bottom: 12px;
        border-radius: 10px;
    }
    .rec-title {
        font-weight: 700;
        font-size: 1.1rem;
        color: #38BDF8;
        margin-bottom: 4px;
    }
    .rec-desc {
        font-size: 0.92rem;
        color: #CBD5E1;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. MODEL LOADING WITH CACHING
# ------------------------------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        clf = joblib.load('career_recommendation_classifier.pkl')
        reg = joblib.load('career_match_score_regressor.pkl')
        vec = joblib.load('tfidf_vectorizer.pkl')
        return clf, reg, vec, None
    except Exception as e:
        return None, None, None, str(e)

clf, reg, vec, error = load_models()

# ------------------------------------------------------------------------------
# 3. HEADER BANNER
# ------------------------------------------------------------------------------
st.markdown("""
    <div class="title-container">
        <div class="main-title">🎯 AI Career Guidance Portal</div>
        <div class="sub-title">Automated Skill Gap Analytics & Recommendation Engine</div>
    </div>
""", unsafe_allow_html=True)

if error:
    st.error(f"⚠️ Model Loading Failure: {error}")
    st.info("Ensure the model binary files (`.pkl`) exist in your working directory.")
    st.stop()

# ------------------------------------------------------------------------------
# 4. CONTROL CENTER (SIDEBAR)
# ------------------------------------------------------------------------------
st.sidebar.markdown('<div class="sidebar-header">⚙️ Control Center</div>', unsafe_allow_html=True)

preset_choice = st.sidebar.selectbox(
    "📌 Quick Test Presets:",
    [
        "Custom Skills Input",
        "Data Scientist Profile",
        "Frontend Developer Profile",
        "Full Match Alignment"
    ]
)

# Corrected preset values
if preset_choice == "Data Scientist Profile":
    default_user = "Python, Machine Learning, Data Science, SQL, Pandas"
    default_job = "Python, SQL, AWS, Spark, Data Engineering, Docker"
elif preset_choice == "Frontend Developer Profile":
    default_user = "HTML, CSS, JavaScript, React, Tailwind CSS"
    default_job = "HTML, CSS, JavaScript, React, Tailwind CSS, TypeScript"
elif preset_choice == "Full Match Alignment":
    default_user = "Python, Machine Learning, PyTorch, Deep Learning, MLOps"
    default_job = "Python, Machine Learning, PyTorch, Deep Learning, MLOps"
else:
    default_user = "Python, Machine Learning, Data Science, SQL"
    default_job = "Python, Machine Learning, SQL, AI, Cloud"

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-header">ℹ️ System Details</div>', unsafe_allow_html=True)

st.sidebar.markdown("""
    <div class="sidebar-card">
        <div class="sys-info-item">
            <span class="sys-label">Model Engine</span>
            <span class="sys-pill">Dual Random Forest</span>
        </div>
        <div class="sys-info-item">
            <span class="sys-label">Vector Space</span>
            <span class="sys-pill">TF-IDF Cosine</span>
        </div>
        <div class="sys-info-item">
            <span class="sys-label">Training Size</span>
            <span class="sys-pill">100,000 Records</span>
        </div>
        <div class="sys-info-item">
            <span class="sys-label">System Status</span>
            <span style="color: #34D399; font-weight: 700; font-size: 0.85rem;">Operational 🟢</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 5. INPUT FORM SECTION
# ------------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-header">👤 Candidate Profile</div>', unsafe_allow_html=True)
    user_skills_input = st.text_area(
        "User Skills (comma-separated):",
        value=default_user,
        height=130,
        help="Input applicant skills."
    )

with col2:
    st.markdown('<div class="section-header">💼 Target Job Role Requirements</div>', unsafe_allow_html=True)
    job_reqs_input = st.text_area(
        "Job Requirements (comma-separated):",
        value=default_job,
        height=130,
        help="Input target role requirements."
    )

st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("🚀 Execute AI Recommendation", use_container_width=True, type="primary")

# Taxonomy for alternative role suggestions when target job doesn't match
JOB_TAXONOMY = {
    "Data Scientist / ML Engineer": ["python", "machine learning", "pytorch", "deep learning", "mlops", "data science", "sql"],
    "Frontend Developer": ["html", "css", "javascript", "react", "tailwind css", "typescript", "vue"],
    "Data Analyst": ["python", "sql", "excel", "pandas", "tableau", "power bi"],
    "DevOps / Cloud Engineer": ["aws", "docker", "kubernetes", "cloud", "linux", "python", "terraform"],
    "Backend Engineer": ["python", "java", "sql", "node.js", "postgres", "api", "docker"]
}

# ------------------------------------------------------------------------------
# 6. INFERENCE & VISUALIZATION LOGIC
# ------------------------------------------------------------------------------
if analyze_btn:
    if not user_skills_input.strip() or not job_reqs_input.strip():
        st.warning("⚠️ Please provide valid inputs for both skill fields.")
    else:
        with st.spinner("Processing feature extraction & running model inference..."):
            time.sleep(0.2)
            
            # Feature extraction
            u_set = set([s.strip().lower() for s in user_skills_input.split(',') if s.strip()])
            j_set = set([s.strip().lower() for s in job_reqs_input.split(',') if s.strip()])
            
            intersection = u_set.intersection(j_set)
            union = u_set.union(j_set)
            
            jaccard_sim = len(intersection) / len(union) if len(union) > 0 else 0.0
            overlap_count = len(intersection)
            u_count = len(u_set)
            j_count = len(j_set)
            coverage_ratio = len(intersection) / len(j_set) if len(j_set) > 0 else 0.0
            
            # TF-IDF Cosine similarity
            u_vec = vec.transform([user_skills_input])
            j_vec = vec.transform([job_reqs_input])
            
            u_n = u_vec.multiply(1.0 / np.sqrt(u_vec.multiply(u_vec).sum() + 1e-9))
            j_n = j_vec.multiply(1.0 / np.sqrt(j_vec.multiply(j_vec).sum() + 1e-9))
            tfidf_cos_sim = float(u_n.multiply(j_n).sum())
            
            features = pd.DataFrame([[jaccard_sim, overlap_count, u_count, j_count, coverage_ratio, tfidf_cos_sim]], 
                                    columns=['jaccard_sim', 'overlap_count', 'user_skill_count', 'job_req_count', 'skill_coverage_ratio', 'tfidf_cos_sim'])
            
            # Model Predictions
            raw_pred = clf.predict(features)[0]
            rec_prob = clf.predict_proba(features)[0][1]
            predicted_score = float(reg.predict(features)[0])
            match_score_clamped = min(max(predicted_score, 0.0), 1.0)
            
            # Deterministic Guardrail: If coverage or similarity is extremely high, enforce RECOMMENDED decision
            if coverage_ratio >= 0.8 or jaccard_sim >= 0.75:
                rec_pred = 1
                rec_prob = max(rec_prob, 0.95)
                match_score_clamped = max(match_score_clamped, coverage_ratio)
            else:
                rec_pred = raw_pred

        # --- EXECUTIVE METRICS DASHBOARD ---
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            status_html = '<div class="metric-val-success">✅ RECOMMENDED</div>' if rec_pred == 1 else '<div class="metric-val-danger">❌ NOT MATCHED</div>'
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-title">Model Decision</div>
                    {status_html}
                </div>
            ''', unsafe_allow_html=True)
            
        with m2:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-title">Predicted Match Score</div>
                    <div class="metric-val-blue">{match_score_clamped * 100:.1f}%</div>
                </div>
            ''', unsafe_allow_html=True)
            
        with m3:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-title">Classifier Confidence</div>
                    <div class="metric-val-blue">{rec_prob * 100:.1f}%</div>
                </div>
            ''', unsafe_allow_html=True)
            
        with m4:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-title">Requirement Coverage</div>
                    <div class="metric-val-blue">{coverage_ratio * 100:.1f}%</div>
                </div>
            ''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- VISUAL GRAPHICAL ANALYTICS ---
        st.markdown('<div class="section-header">📈 Feature Representation & Skill Gap Visualizations</div>', unsafe_allow_html=True)
        
        g1, g2 = st.columns(2)
        
        with g1:
            fig1, ax1 = plt.subplots(figsize=(6, 3.8))
            fig1.patch.set_facecolor('#111827')
            ax1.set_facecolor('#111827')
            
            feature_names = ['Jaccard Sim', 'Skill Coverage', 'TF-IDF Cosine']
            feature_vals = [jaccard_sim, coverage_ratio, tfidf_cos_sim]
            
            bars = ax1.bar(feature_names, feature_vals, color=['#38BDF8', '#818CF8', '#34D399'], width=0.45)
            ax1.set_ylim(0, 1.08)
            ax1.set_ylabel("Similarity Score (0 - 1)", color="#94A3B8", fontweight="bold")
            ax1.set_title("Extracted Vector Similarities", color="#F8FAFC", fontweight="bold", pad=12)
            ax1.tick_params(colors='#94A3B8')
            ax1.grid(axis='y', linestyle='--', alpha=0.15)
            
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.2f}',
                         ha='center', va='bottom', color='#FFFFFF', fontweight='bold')
                         
            st.pyplot(fig1)

        with g2:
            fig2, ax2 = plt.subplots(figsize=(6, 3.8))
            fig2.patch.set_facecolor('#111827')
            
            overlap_num = len(intersection)
            missing_num = len(j_set - u_set)
            
            sizes = [overlap_num, max(missing_num, 0)]
            labels = ['Matching Skills', 'Missing Gap']
            colors = ['#34D399', '#F87171']
            
            if sum(sizes) > 0:
                wedges, texts, autotexts = ax2.pie(
                    sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                    startangle=140, textprops=dict(color="#FFFFFF", fontweight="bold"),
                    wedgeprops=dict(width=0.4, edgecolor='#111827')
                )
                plt.setp(autotexts, size=10, weight="bold")
                ax2.set_title("Target Requirement Fulfillment", color="#F8FAFC", fontweight="bold", pad=12)
            else:
                ax2.text(0.5, 0.5, "No Requirements Detected", ha='center', color="#94A3B8")
                
            st.pyplot(fig2)

        # --- GRANULAR SKILL BREAKDOWN ---
        st.markdown('<div class="section-header">🔍 Granular Skill Gap Analysis</div>', unsafe_allow_html=True)
        
        b1, b2 = st.columns(2)
        
        with b1:
            st.markdown("#### 🟢 Overlapping Matched Skills")
            if intersection:
                html_badges = "".join([f'<span class="badge-match">✓ {s.title()}</span>' for s in intersection])
                st.markdown(html_badges, unsafe_allow_html=True)
            else:
                st.info("No matching skills detected between profile and role requirements.")

        with b2:
            st.markdown("#### 🔴 Missing Skill Requirements")
            missing = j_set - u_set
            if missing:
                html_badges = "".join([f'<span class="badge-missing">✕ {s.title()}</span>' for s in missing])
                st.markdown(html_badges, unsafe_allow_html=True)
            else:
                st.success("Perfect alignment! Candidate meets all target criteria.")

        # --- ALTERNATIVE ROLE RECOMMENDATIONS (IF NOT MATCHED) ---
        if rec_pred == 0:
            st.markdown('<div class="section-header">💡 Alternative Recommended Roles</div>', unsafe_allow_html=True)
            st.info("The candidate profile does not meet the full criteria for the target role. Based on the candidate's skill set, here are alternative recommended roles:")
            
            matched_roles = []
            for role_name, role_skills in JOB_TAXONOMY.items():
                role_set = set(role_skills)
                common = u_set.intersection(role_set)
                if common:
                    score = (len(common) / len(role_set)) * 100
                    matched_roles.append((role_name, score, common))
            
            matched_roles.sort(key=lambda x: x[1], reverse=True)
            
            if matched_roles:
                for role, score, common_skills in matched_roles[:3]:
                    skills_str = ", ".join([s.title() for s in common_skills])
                    st.markdown(f'''
                        <div class="rec-card">
                            <div class="rec-title">🎯 Suggested Role: {role} ({score:.0f}% Match)</div>
                            <div class="rec-desc"><b>Key Matching Skills:</b> {skills_str}</div>
                        </div>
                    ''', unsafe_allow_html=True)
            else:
                st.write("No direct taxonomy matches found for current skills.")

# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 0.85rem; padding-bottom: 20px;'>"
    "<b>AI-Powered Career Guidance Portal</b> | Developed with Streamlit & Scikit-Learn"
    "</div>", 
    unsafe_allow_html=True
)