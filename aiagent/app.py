import os
import re
import uuid
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from openai import OpenAI
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

load_dotenv()

session_data_cache = {}
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

UPLOAD_FOLDER = './uploads'
RESULT_FOLDER = './results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def perform_advanced_analysis(df, query):
    """
    Recognizes and executes complex analysis commands like elbow, kmeans, dbscan, 
    hierarchical, silhouette score, and davies-bouldin index.
    """
    query = query.lower()
    
    # Define keywords that trigger this advanced analysis
    keywords = ["elbow", "k-means", "kmeans", "dbscan", "hierarchical", "agglomerative", "silhouette", "davies-bouldin", "davies bouldin"]
    if not any(keyword in query for keyword in keywords):
        return None

    try:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2: 
            return "Analisis tidak dapat dilakukan. Dibutuhkan setidaknya 2 kolom numerik pada data Anda."
        
        df_proc = df[numeric_cols].copy()
        df_cleaned = clean_and_fill_mean(df_proc)
        df_scaled = normalize(df_cleaned, df_cleaned.columns)

        if "elbow" in query:
            k = elbow_method(df_scaled)
            return f"Hasil perhitungan Elbow Method menunjukkan bahwa jumlah cluster (K) yang optimal untuk data Anda adalah **{k}**."
        
        if "silhouette" in query or "davies-bouldin" in query or "davies bouldin" in query:
            optimal_k = elbow_method(df_scaled)
            if optimal_k <= 1:
                return f"Tidak dapat menghitung skor validasi karena jumlah cluster optimal yang ditemukan hanya {optimal_k}. Dibutuhkan minimal 2 cluster."
            
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto').fit(df_scaled)
            labels = kmeans.labels_
            
            if "silhouette" in query:
                score = silhouette_score(df_scaled, labels)
                return f"**Silhouette Score** untuk data Anda (dengan K={optimal_k}) adalah **{score:.4f}**. Skor berkisar dari -1 hingga 1. Semakin mendekati 1, semakin baik clusternya terpisah."
            if "davies-bouldin" in query or "davies bouldin" in query:
                score = davies_bouldin_score(df_scaled, labels)
                return f"**Davies-Bouldin Index** untuk data Anda (dengan K={optimal_k}) adalah **{score:.4f}**. Semakin rendah skornya (mendekati 0), semakin baik model clusternya."

        # --- Clustering Algorithms ---
        if "k-means" in query or "kmeans" in query:
            num_search = re.search(r'\d+', query)
            k = int(num_search.group(0)) if num_search else elbow_method(df_scaled)
            model = KMeans(n_clusters=k, random_state=42, n_init='auto') 
            clusters = model.fit_predict(df_scaled)
            cluster_counts = pd.Series(clusters).value_counts().to_dict()
            return f"Analisis **K-Means** (menggunakan inisialisasi K-Means++) dengan **{k} cluster** telah dijalankan. Berikut adalah distribusi data di setiap cluster:\n{json.dumps(cluster_counts, indent=2)}"

        if "dbscan" in query:
            model = DBSCAN(eps=0.5, min_samples=5)
            clusters = model.fit_predict(df_scaled)
            cluster_counts = pd.Series(clusters).value_counts()
            noise_points = cluster_counts.get(-1, 0)
            num_clusters = len(cluster_counts) - (1 if -1 in cluster_counts.index else 0)
            return f"Analisis **DBSCAN** telah dijalankan dan menemukan **{num_clusters} cluster** dengan **{noise_points} titik noise** (data yang dianggap anomali)."

        if "hierarchical" in query or "agglomerative" in query:
            num_search = re.search(r'\d+', query)
            k = int(num_search.group(0)) if num_search else elbow_method(df_scaled)
            model = AgglomerativeClustering(n_clusters=k)
            clusters = model.fit_predict(df_scaled)
            cluster_counts = pd.Series(clusters).value_counts().to_dict()
            return f"Analisis **Hierarchical Clustering** dengan target **{k} cluster** telah dijalankan. Berikut adalah distribusi data di setiap cluster:\n{json.dumps(cluster_counts, indent=2)}"

    except Exception as e:
        return f"Terjadi kesalahan saat mencoba melakukan analisis: {str(e)}"

    return None

def perform_basic_data_analysis(df, query):
    """
    Parses a user's query for basic statistics (mean, sum, etc.).
    """
    query = query.lower()
    column_search = re.search(r"['\"](.*?)['\"]", query)
    actual_col = None
    if column_search:
        col_name = column_search.group(1)
        for c in df.columns:
            if c.lower() == col_name.lower():
                actual_col = c
                break
    
    try:
        if actual_col and pd.api.types.is_numeric_dtype(df[actual_col]):
            if "rata-rata" in query: return f"Rata-rata untuk '{actual_col}' adalah: {df[actual_col].mean():.2f}"
            if "total" in query: return f"Total untuk '{actual_col}' adalah: {df[actual_col].sum():,.2f}"
            if "median" in query: return f"Median untuk '{actual_col}' adalah: {df[actual_col].median():.2f}"
            if "minimum" in query: return f"Nilai minimum untuk '{actual_col}' adalah: {df[actual_col].min():.2f}"
            if "maksimum" in query: return f"Nilai maksimum untuk '{actual_col}' adalah: {df[actual_col].max():.2f}"
            if "statistik" in query: return f"Statistik deskriptif untuk '{actual_col}':\n{df[actual_col].describe().to_string()}"
        if "korelasi" in query:
            numeric_df = df.select_dtypes(include=np.number)
            if len(numeric_df.columns) < 2: return "Tidak cukup kolom numerik untuk korelasi."
            return f"Matriks korelasi:\n{numeric_df.corr().to_string()}"
    except Exception as e: return f"Kesalahan analisis: {str(e)}"
    return None

def clean_and_fill_mean(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]): df[col].fillna(df[col].mean(), inplace=True)
        else:
            mode_val = df[col].mode(); df[col].fillna(mode_val[0] if not mode_val.empty else 'Missing', inplace=True)
    return df
def normalize(df, feature_cols):
    scaler = StandardScaler(); return pd.DataFrame(scaler.fit_transform(df[feature_cols]), columns=feature_cols)
def encode_categorical(df, feature_cols):
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]
    encoded_df = pd.DataFrame()
    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        if not df[categorical_cols].empty:
            encoded_data = encoder.fit_transform(df[categorical_cols])
            encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
    numeric_df = df[numeric_cols].reset_index(drop=True)
    encoded_df = encoded_df.reset_index(drop=True)
    return pd.concat([numeric_df, encoded_df], axis=1)
def elbow_method(df_scaled, max_k=10):
    if len(df_scaled) < max_k: max_k = len(df_scaled) - 1 if len(df_scaled) > 1 else 1
    if max_k <= 1: return 1
    distortions = []
    K = range(1, max_k + 1)
    for k in K: kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto'); kmeans.fit(df_scaled); distortions.append(kmeans.inertia_)
    if len(distortions) < 3: return max_k if max_k > 0 else 1
    deltas = np.diff(distortions); second_deltas = np.diff(deltas)
    optimal_k = np.argmin(second_deltas) + 2
    return max(1, optimal_k)
def generate_summary(df, feature_cols, algorithm, n_clusters):
    summary = f"Model {algorithm.upper()} telah dijalankan dengan hasil {n_clusters} cluster berdasarkan fitur: {', '.join(feature_cols)}.\n"
    for cluster in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster]
        if cluster == -1: summary += f"- Cluster -1 (noise) berisi {len(cluster_data)} data.\n"
        else: summary += f"- Cluster {cluster} berisi {len(cluster_data)} data.\n"
    return summary
def query_openrouter(prompt):
    try:
        response = client.chat.completions.create(model="openai/gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content
    except Exception as e: return f"Gagal mengakses AI: {str(e)}"

@app.route('/')
def landing_page(): return render_template('landing.html')
@app.route('/sapadapa')
def sapadapa_page(): return render_template('sapadapa.html')
@app.route('/app')
def main_app(): return render_template('index.html')

@app.route('/upload_for_sapadapa', methods=['POST'])
def upload_for_sapadapa():
    session_id = request.form.get('session_id')
    if not session_id: return jsonify({'error': 'Session ID is required'}), 400
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    try:
        df = pd.read_csv(file, encoding='utf-8', sep=request.form.get('delimiter', ','))
        session_data_cache[session_id] = {"dataframe": df, "chat_history": []}
        ai_question = "Data berhasil diunggah. Anda sekarang dapat meminta analisis seperti 'hitung metode elbow', 'jalankan hierarchical clustering', atau 'hitung silhouette score'."
        return jsonify({ 'success': True, 'ai_question': ai_question })
    except Exception as e: return jsonify({'error': f'Gagal memproses file: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat requests with a multi-layered analysis logic."""
    data = request.json
    session_id = data.get('session_id')
    if not session_id: return jsonify({'reply': "Error: ID Sesi tidak ditemukan."}), 400
    user_message = data.get('message', '')
    if not user_message: return jsonify({'reply': "Tolong masukkan pertanyaan."})

    chat_context = data.get('context', 'general')
    current_session_data = session_data_cache.get(session_id, {})
    chat_history = current_session_data.get("chat_history", [])
    formatted_history = "\n".join([f"User: {turn['user']}\nAI: {turn['bot']}" for turn in chat_history])
    prompt = ""

    if chat_context == 'sapadapa_chat':
        analysis_result = None
        df = current_session_data.get("dataframe")

        if df is not None:
            analysis_result = perform_advanced_analysis(df, user_message)
            if not analysis_result:
                analysis_result = perform_basic_data_analysis(df, user_message)

        if analysis_result:
            prompt = f"Anda adalah AI analis data. Sistem telah melakukan perhitungan akurat. Tugas Anda adalah menjelaskan hasil berikut dengan cara yang mudah dimengerti.\n\nHasil Perhitungan:\n---\n{analysis_result}\n---\n\nPenjelasan Anda:"

        else:
            if 'sapadapa' in user_message.lower():
                prompt = "Anda adalah AI Agent Aurinova. Jelaskan secara ringkas keempat tahap kerangka kerja SAPADAPA: Situation, Problem, Decision, dan Potential Problem Analysis."
            elif df is not None:
                raw_data_head = df.head(3).to_string()
                prompt = f"Anda adalah AI Agent Aurinova. Konteks data user:\n{raw_data_head}\nRiwayat:\n{formatted_history}\n\nJawab: \"{user_message}\"."
            else:
                prompt = f"Anda adalah AI Agent Aurinova. Riwayat:\n{formatted_history}\n\nJawab pertanyaan umum: \"{user_message}\". Anda belum punya data."
    else: 
        summary = current_session_data.get("summary", "Belum ada hasil.")
        prompt = f"Anda adalah AI penganalisis hasil klasterisasi. Ringkasan:\n{summary}\nRiwayat:\n{formatted_history}\n\nJawab: \"{user_message}\"\nAI:"
            
    ai_reply = query_openrouter(prompt)
    chat_history.append({"user": user_message, "bot": ai_reply})
    current_session_data["chat_history"] = chat_history
    session_data_cache[session_id] = current_session_data
    return jsonify({'reply': ai_reply})


@app.route('/upload', methods=['POST'])
def upload_file():
    session_id = request.form.get('session_id')
    if not session_id: return jsonify({'error': 'Session ID is required'}), 400
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    try:
        unique_id = str(uuid.uuid4())
        img_path = os.path.join(RESULT_FOLDER, f'cluster_plot_{unique_id}.png')
        result_csv_path = os.path.join(RESULT_FOLDER, f'clustered_data_{unique_id}.csv')
        df = pd.read_csv(file, encoding='utf-8', sep=request.form.get('delimiter', ','))
        feature_cols = [f.strip() for f in request.form.get('features').split(',') if f.strip() in df.columns]
        if not feature_cols: raise ValueError("Tidak ada fitur valid yang dipilih.")
        df_cleaned = clean_and_fill_mean(df.copy()); df_encoded = encode_categorical(df_cleaned, feature_cols)
        if df_encoded.empty: raise ValueError("Tidak ada data valid setelah encoding.")
        df_scaled = normalize(df_encoded, df_encoded.columns)
        algo = request.form.get('algorithm', 'kmeans')
        n_clusters_req = request.form.get('n_clusters')
        if algo == 'kmeans':
            n_clusters = int(n_clusters_req) if n_clusters_req and n_clusters_req.isdigit() else elbow_method(df_scaled)
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        else: model = DBSCAN(eps=0.5, min_samples=5)
        clusters = model.fit_predict(df_scaled)
        df['cluster'] = clusters
        df.to_csv(result_csv_path, index=False)
        plot_url = ""
        numeric_features_for_plot = df_scaled.select_dtypes(include=np.number).columns
        if len(numeric_features_for_plot) >= 2:
            pca = PCA(n_components=2); pca_result = pca.fit_transform(df_scaled[numeric_features_for_plot])
            plt.figure(figsize=(10, 8)); unique_clusters = np.unique(clusters)
            cmap = cm.get_cmap('tab10', len(unique_clusters))
            for i, cluster_label in enumerate(unique_clusters):
                points = pca_result[clusters == cluster_label]
                label_text = f'Cluster {cluster_label}' if cluster_label != -1 else 'Noise'
                plt.scatter(points[:, 0], points[:, 1], color=cmap(i), label=label_text, alpha=0.8, s=50)
            plt.xlabel('Principal Component 1'); plt.ylabel('Principal Component 2'); plt.title(f'Hasil Klasterisasi ({algo.upper()})')
            plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(img_path); plt.close()
            plot_url = f'/download/plot/{unique_id}'
        summary = generate_summary(df, feature_cols, algo, len(np.unique(clusters)))
        session_data_cache[session_id] = {"summary": summary, "chat_history": []}
        return jsonify({'clusters': {str(k): int(v) for k, v in df['cluster'].value_counts().items()},'features': feature_cols, 'plot_url': plot_url, 'csv_url': f'/download/csv/{unique_id}'})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/download/plot/<uid>')
def download_plot(uid):
    filepath = os.path.join(RESULT_FOLDER, f'cluster_plot_{uid}.png')
    if not os.path.exists(filepath): return "Plot file not found", 404
    return send_file(filepath, mimetype='image/png')
@app.route('/download/csv/<uid>')
def download_csv(uid):
    filepath = os.path.join(RESULT_FOLDER, f'clustered_data_{uid}.csv')
    if not os.path.exists(filepath): return "CSV file not found", 404
    return send_file(filepath, mimetype='text/csv', as_attachment=True, download_name='clustered_data.csv')
@app.route('/get_cached_data/<session_id>', methods=['GET'])
def get_cached_data(session_id):
    cached_data = session_data_cache.get(session_id, {})
    return jsonify({'chat_history': cached_data.get('chat_history', [])})
@app.route('/reset_session', methods=['POST'])
def reset_session():
    session_id = request.json.get('session_id')
    if session_id in session_data_cache: del session_data_cache[session_id]; return jsonify({'status': 'success'})
    return jsonify({'status': 'failed'}), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)