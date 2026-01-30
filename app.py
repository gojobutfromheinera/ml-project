from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # allows frontend to call backend

# =========================
# LOAD MODEL (RUN ONCE)
# =========================
tfidf = pickle.load(open("models/tfidf.pkl", "rb"))
tfidf_matrix = pickle.load(open("models/tfidf_matrix.pkl", "rb"))
df = pickle.load(open("models/courses_df.pkl", "rb"))

# =========================
# PREDEFINED IMAGES FOR COURSES
# =========================
courses_img = {
    "python":[
        "https://img.freepik.com/free-vector/programming-concept-illustration_114360-1351.jpg",
        "https://images.unsplash.com/photo-1624953587687-daf255b6b80a?q=80&w=1074&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://images.unsplash.com/photo-1649180556628-9ba704115795?q=80&w=1162&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://images.unsplash.com/photo-1690683790356-c1edb75e3df7?q=80&w=2109&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://images.unsplash.com/photo-1526379095098-d400fd0bf935?q=80&w=1632&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://plus.unsplash.com/premium_vector-1682310979404-d8b3a7789482?q=80&w=1008&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://plus.unsplash.com/premium_vector-1726307125386-a76f9b9b21b8?q=80&w=1267&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://plus.unsplash.com/premium_vector-1711987684064-d3a0ffb6790e?q=80&w=1021&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://img.freepik.com/free-vector/coding-concept-illustration_114360-939.jpg",
        "https://img.freepik.com/free-vector/app-development-concept_114360-5161.jpg"
    ],
    "ai":[
        "https://img.freepik.com/free-vector/machine-learning-concept_114360-1000.jpg",
        "https://img.freepik.com/free-vector/deep-learning-concept_114360-1022.jpg",
        "https://img.freepik.com/free-vector/neural-network-concept_114360-2571.jpg"
    ],
    "web":[
        "https://img.freepik.com/free-vector/web-design-concept_114360-1083.jpg",
        "https://img.freepik.com/free-vector/programmer-concept_114360-2284.jpg"
        "https://plus.unsplash.com/premium_vector-1682310664746-f934119dffd6?q=80&w=1014&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        "https://plus.unsplash.com/premium_vector-1734127305687-4440bad6d7a7?q=80&w=1025&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        "https://plus.unsplash.com/premium_vector-1734421474117-e5e4c5f47bfb?q=80&w=977&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        "https://plus.unsplash.com/premium_vector-1733932446246-6333cc2ccc16?q=80&w=2148&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    ],
    "data":[
        "https://img.freepik.com/free-vector/data-analytics-concept_114360-5211.jpg",
        "https://img.freepik.com/free-vector/big-data-concept_114360-4180.jpg",
        "https://img.freepik.com/free-vector/database-concept_114360-579.jpg"
        "https://images.unsplash.com/photo-1666875753105-c63a6f3bdc86?q=80&w=1173&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        "https://plus.unsplash.com/premium_photo-1661878265739-da90bc1af051?q=80&w=1386&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        "https://plus.unsplash.com/premium_photo-1661964014072-2a78dd2232cd?q=80&w=1086&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        "https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    ],
    "design":[
        "https://img.freepik.com/free-vector/ui-ux-concept_114360-701.jpg",
        "https://img.freepik.com/free-vector/app-interface-concept_114360-472.jpg",
        "https://img.freepik.com/free-vector/design-thinking-concept_114360-285.jpg",
        "https://img.freepik.com/free-vector/creative-design-concept_114360-1745.jpg"
    ],
    "cloud":[
        "https://img.freepik.com/free-vector/cloud-computing-concept_114360-234.jpg",
        "https://img.freepik.com/free-vector/devops-concept-illustration_114360-1031.jpg",
        "https://img.freepik.com/free-vector/data-center-concept-illustration_114360-1406.jpg",
        "https://img.freepik.com/free-vector/blockchain-concept_114360-200.jpg"
    ]
}

# =========================
# RECOMMEND FUNCTION
# =========================
def recommend_courses(keyword, top_n=5):
    keyword_vec = tfidf.transform([keyword.lower()])
    similarities = cosine_similarity(keyword_vec, tfidf_matrix).flatten()

    top_indices = similarities.argsort()[-top_n:][::-1]

    results = df.iloc[top_indices][['course_title', 'course_url']]  # no category column needed
    results['score'] = similarities[top_indices]

    return results

# =========================
# API ENDPOINT
# =========================
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"success": False, "courses": []})

    results = recommend_courses(query)

    response_courses = []
    for idx, row in results.iterrows():
        title = row['course_title'].lower()
        link = row['course_url']

        # ===== Heuristic to pick image based on title keywords =====
        if 'python' in title:
            img_url = courses_img['python'][idx % len(courses_img['python'])]
        elif 'ai' in title or 'machine learning' in title or 'deep learning' in title:
            img_url = courses_img['ai'][idx % len(courses_img['ai'])]
        elif 'web' in title or 'html' in title or 'javascript' in title:
            img_url = courses_img['web'][idx % len(courses_img['web'])]
        elif 'data' in title or 'analytics' in title:
            img_url = courses_img['data'][idx % len(courses_img['data'])]
        elif 'ui' in title or 'ux' in title or 'design' in title:
            img_url = courses_img['design'][idx % len(courses_img['design'])]
        elif 'cloud' in title or 'aws' in title or 'devops' in title:
            img_url = courses_img['cloud'][idx % len(courses_img['cloud'])]
        else:
            # Fallback image for any other keyword
            img_url = "https://img.freepik.com/free-vector/online-learning-concept_114360-4766.jpg"


        response_courses.append({
            "name": row['course_title'],
            "img": img_url,
            "link": link
        })

    return jsonify({
        "success": True,
        "courses": response_courses
    })
@app.route("/")
def home():
    return "Flask ML API is running"
# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
    