import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
import PyPDF2
import random

# ---------------- Config ----------------
st.set_page_config(page_title="Smart Research Assistant", layout="wide")
st.title("📚 Smart Research Assistant")

@st.cache_resource
def load_models():
    summarizer_tokenizer = AutoTokenizer.from_pretrained("models/t5-small")
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("models/t5-small")

    qa_tokenizer = AutoTokenizer.from_pretrained("models/bert-squad2")
    qa_model = AutoModelForQuestionAnswering.from_pretrained("models/bert-squad2")

    embedder = SentenceTransformer("models/all-MiniLM-L6-v2")

    return summarizer_model, summarizer_tokenizer, qa_model, qa_tokenizer, embedder

summarizer_model, summarizer_tokenizer, qa_model, qa_tokenizer, embedder = load_models()

@st.cache_data
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def split_text(text, max_words=300):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

@st.cache_data
def embed_chunks(chunks):
    return embedder.encode(chunks, convert_to_tensor=True)

# ----------- Algorithm Detection -----------
def extract_algorithm_sentences(text):
    keywords = [
        "SVM", "support vector machine", "logistic regression", "decision tree",
        "random forest", "naive bayes", "neural network", "deep learning", "KNN", "k-means"
    ]
    lines = text.split("\n")
    matched_lines = []
    for i, line in enumerate(lines):
        if any(kw.lower() in line.lower() for kw in keywords):
            start = max(0, i - 1)
            end = min(len(lines), i + 2)
            matched_lines.extend([lines[j].strip() for j in range(start, end)])
    return list(set(matched_lines))

def extract_ml_techniques(lines):
    techniques = []
    for line in lines:
        line_lower = line.lower()
        if "svm" in line_lower: techniques.append("SVM")
        if "logistic regression" in line_lower: techniques.append("Logistic Regression")
        if "decision tree" in line_lower: techniques.append("Decision Tree")
        if "random forest" in line_lower: techniques.append("Random Forest")
        if "naive bayes" in line_lower or "naïve bayes" in line_lower: techniques.append("Naive Bayes")
        if "neural network" in line_lower: techniques.append("Neural Network")
        if "deep learning" in line_lower: techniques.append("Deep Learning")
        if "k-means" in line_lower: techniques.append("K-Means")
        if "knn" in line_lower: techniques.append("K-NN")
    return sorted(set(techniques))

def should_use_strict_filter(question: str) -> bool:
    return any(kw in question.lower() for kw in [
        "ml algorithm", "machine learning technique", "ml technique", "classification algorithm"
    ])

def summarize_chunk(chunk):
    inputs = summarizer_tokenizer("summarize: " + chunk, return_tensors="pt", truncation=True)
    with torch.no_grad():
        output = summarizer_model.generate(inputs["input_ids"], max_length=120, min_length=30, num_beams=4)
    return summarizer_tokenizer.decode(output[0], skip_special_tokens=True)

# ---------- Evaluation Logic ----------
def evaluate_answer(question, student_answer):
    if not student_answer.strip() or student_answer.lower() in ["none", "na"]:
        return "Rating: 1 - No answer provided. Try describing what you understood."

    word_count = len(student_answer.split())
    if word_count < 4:
        return random.choice([
            "Rating: 2 - Too short. Try writing a complete sentence.",
            "Rating: 2 - Needs more detail. Expand your answer with one or two key points."
        ])

    return random.choice([
        "Rating: 4 - Good answer. You understood the section.",
        "Rating: 5 - Excellent! Clear and relevant response."
    ])

# ----------- Main App -----------
uploaded_file = st.file_uploader("Upload a research PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    raw_text = extract_text_from_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else uploaded_file.read().decode("utf-8")

    if not raw_text.strip():
        st.error("❌ No useful content detected. Please upload a different file.")
    else:
        st.success("✅ File uploaded and processed successfully!")

        chunks = split_text(raw_text)
        summaries = []

        with st.spinner("🔍 Generating summary..."):
            for i, chunk in enumerate(chunks[:5]):
                try:
                    summary = summarize_chunk(chunk)
                    summaries.append(f"Section {i + 1}: {summary}")
                except Exception:
                    summaries.append(f"Section {i + 1}: ❌ Error summarizing this part.")

        st.subheader("📚 Summary Output")
        for summary in summaries:
            st.write(summary)

        # ---------- Ask Anything ----------
        st.subheader("❓ Ask Anything from the Document")
        user_query = st.text_input("Enter your question below:")

        if user_query:
            if should_use_strict_filter(user_query):
                st.markdown("🔍 Using ML-algorithm-specific lines...")
                matched_lines = extract_algorithm_sentences(raw_text)
                techniques = extract_ml_techniques(matched_lines)
                if techniques:
                    st.markdown(f"**Answer:** {', '.join(techniques)}")
                    with st.expander("🔍 Show matched lines"):
                        st.code("\n".join(matched_lines))
                else:
                    st.warning("⚠️ No specific ML techniques found.")
            else:
                st.markdown("🧠 Using full semantic search...")
                chunk_embeddings = embed_chunks(chunks)
                query_embedding = embedder.encode(user_query, convert_to_tensor=True)
                cos_scores = torch.nn.functional.cosine_similarity(query_embedding, chunk_embeddings)
                top_indices = torch.topk(cos_scores, k=3).indices.tolist()
                context = " ".join([chunks[i] for i in top_indices])

                try:
                    qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)
                    result = qa_pipeline(question=user_query, context=context)
                    st.markdown(f"**Answer:** {result['answer']}")
                    with st.expander("🔍 Show selected context"):
                        st.code(context)
                except Exception as e:
                    st.error(f"❌ Error during QA: {e}")

        # ---------- Challenge Me ----------
        st.subheader("🧠 Challenge: What did you learn?")
        if st.button("Challenge Me!"):
            questions = [f"What is the main focus of section {i + 1}?" for i in range(min(3, len(summaries)))]
            st.session_state["challenge_questions"] = questions
            st.session_state["user_answers"] = ["" for _ in questions]

        if "challenge_questions" in st.session_state:
            for idx, q in enumerate(st.session_state["challenge_questions"]):
                st.markdown(f"**{q}**")
                st.caption(f"_Hint: {summaries[idx]}_")
                st.session_state["user_answers"][idx] = st.text_input("Your Answer:", key=f"challenge_q{idx}")

            if st.button("Submit Answers"):
                st.subheader("📋 Feedback")
                total_score = 0
                for i, (q, ans) in enumerate(zip(st.session_state["challenge_questions"], st.session_state["user_answers"])):
                    feedback = evaluate_answer(q, ans)
                    st.markdown(f"**Q{i + 1} Feedback:** {feedback}")
                    try:
                        score = int(feedback.split("Rating: ")[1].split(" -")[0])
                        total_score += score
                    except:
                        pass
                st.markdown(f"**Your Total Score: {total_score} / {5 * len(st.session_state['challenge_questions'])}**")
