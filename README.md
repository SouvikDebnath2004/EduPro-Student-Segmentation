# EduPro: Student Segmentation & Personalized Recommendation Engine

An end-to-end Machine Learning and interactive web application designed to transform learner engagement on digital education platforms through unsupervised behavioral clustering and hybrid content-collaborative filtering recommendations.

---

## Project Overview
Modern e-learning ecosystems suffer from choice paralysis, often relying on static, platform-wide popular course listings that fail to engage diverse user demographics. **EduPro** addresses this bottleneck by:
1. Grouping learners into actionable behavioral personas using unsupervised machine learning.
2. Delivering contextually relevant, personalized course recommendations via a hybrid scoring engine.
3. Providing stakeholders with an interactive, multi-tab analytics dashboard built with Streamlit.

---

## Technical Architecture & Methodology

### 1. Data Pipeline & Feature Engineering
* **Data Sources:** Merged transactional enrollments, user demographic profiles, and course catalog metadata ($3,000$ learners, $10,000$ transactions, $60$ courses).
* **Engineered Features:** `Total Spending ($)`, `Courses Enrolled`, `Avg Spend/Course`, `Avg Course Rating`, and `Category Diversity Score`.

### 2. Unsupervised Student Segmentation ($K=4$)
* **Algorithm:** Applied **K-Means Clustering** on normalized behavioral feature vectors.
* **Validation:** Optimal cluster boundaries verified via Elbow Method inertia analysis and Silhouette coefficient evaluation.
* **Identified Personas:**
  * **Budget Learners (34.7%):** Cost-conscious users preferring discounted or entry-level modules.
  * **Casual Learners (33.2%):** Introductory consumers exploring basic content.
  * **Explorer Learners (15.1%):** High-engagement power users with broad multi-domain interests.
  * **Premium Specialists (17.0%):** High-value professionals targeting expensive, niche certifications.

### 3. Hybrid Recommendation Engine
* **Content Similarity ($S_{\text{content}}$):** Computed via TF-IDF vectorization and cosine similarity across course metadata (`Category`, `Type`, `Level`).
* **Hybrid Ranking Formula:** Combines content similarity, normalized user rating history, and intra-cluster course popularity affinity to compute a precise match score for top-$N$ filtering.

---

## Interactive Dashboard Features (`app.py`)
* **Tab 1: Learner Profile & Recommendations:** Interactive dropdown for selecting any user ID, displaying real-time KPI overview metrics alongside a dynamically filtered top-$N$ recommended course matrix.
* **Tab 2: Cluster Analytics Dashboard:** Multi-dimensional scatter plot highlighting user distribution (`Spend` vs `Enrollments` scaled by `Diversity`) alongside a cluster volume breakdown donut chart.
* **Tab 3: Segment Comparison Panel:** Executive aggregation table and bar chart contrasting total spending dynamics against unit costs per course across all segments.

---

## Tech Stack
* **Language:** Python 3.11
* **Machine Learning & Data:** `pandas`, `numpy`, `scikit-learn` (K-Means, TfidfVectorizer, Cosine Similarity)
* **Data Visualization:** `plotly.express`
* **Web Framework:** `Streamlit`

---

## Installation & Running Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/SouvikDebnath2004/EduPro-Student-Segmentation.git](https://github.com/SouvikDebnath2004/EduPro-Student-Segmentation.git)
   cd EduPro-Student-Segmentation
