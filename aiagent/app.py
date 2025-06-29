from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
import numpy as np
import requests
import uuid
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import json
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import matplotlib.cm as cm

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


def clean_and_fill_mean(df):
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            
            if not df[col].mode().empty:
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna('Missing', inplace=True) 
    return df

def normalize(df, feature_cols):
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[feature_cols]), columns=feature_cols)
    return df_scaled

def encode_categorical(df, feature_cols):
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]

    encoded_df = pd.DataFrame()
    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        if not df[categorical_cols].empty:
            encoded = encoder.fit_transform(df[categorical_cols])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

    numeric_df = df[numeric_cols].copy()
    df_encoded = pd.concat([numeric_df, encoded_df], axis=1)
    return df_encoded

def elbow_method(df_scaled, max_k=10):
    if len(df_scaled) < max_k: 
        max_k = len(df_scaled) -1 if len(df_scaled) > 1 else 1

    if max_k <= 1:
        return 1

    distortions = []
    K = range(1, max_k + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) 
        kmeans.fit(df_scaled)
        distortions.append(kmeans.inertia_)
    
    
    deltas = np.diff(distortions)
    if len(deltas) < 2: 
        return max_k 

    second_deltas = np.diff(deltas)
    

    optimal_k = np.argmin(second_deltas) + 2
    
    return max(1, optimal_k) 


def generate_summary(df, feature_cols, algorithm, n_clusters):
    summary = f"Model {algorithm.upper()} berhasil mengelompokkan data menjadi {n_clusters} cluster berdasarkan fitur {', '.join(feature_cols)}.\n"
    for cluster in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster]
        if cluster == -1:
            summary += f"- Cluster -1 (noise) mencakup {len(cluster_data)} data ({len(cluster_data)/len(df)*100:.1f}%).\n"
        else:
            summary += f"- Cluster {cluster} mencakup {len(cluster_data)} data ({len(cluster_data)/len(df)*100:.1f}%) dengan rata-rata:\n"
            for col in feature_cols:
                try:
                    if pd.api.types.is_numeric_dtype(cluster_data[col]):
                        mean_val = cluster_data[col].mean()
                        summary += f"   • {col}: {mean_val:.2f}\n"
                    else:
                        mode_val = cluster_data[col].mode().iloc[0] if not cluster_data[col].empty else "N/A"
                        summary += f"   • {col}: {mode_val} (mode)\n"
                except Exception as e:
                    summary += f"   • {col}: [error / non-numeric: {e}]\n"
    return summary

def query_openrouter(prompt):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Gagal mengakses OpenRouter: {str(e)}"

@app.route('/')
def landing_page():
    return render_template('landing.html')

@app.route('/sapadapa')
def sapadapa_page():
    return render_template('sapadapa.html')

@app.route('/app')
def main_app():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    chat_context = data.get('context', 'general') 
    session_id = data.get('session_id') 

    if not session_id:
        return jsonify({'reply': "Error: ID Sesi tidak ditemukan. Mohon refresh halaman."}), 400

    if not user_message:
        return jsonify({'reply': "Tolong masukkan pertanyaan."})

    # Dapatkan data sesi saat ini, atau buat entri baru jika tidak ada
    current_session_data = session_data_cache.get(session_id, {})
    
    # Ambil konteks yang relevan dari cache
    raw_data_head = current_session_data.get("raw_data_head", "")
    clustering_summary = current_session_data.get("summary", "")
    chat_history = current_session_data.get("chat_history", [])

    # Format riwayat obrolan untuk prompt
    formatted_history = "\n".join([f"User: {turn['user']}\nAI: {turn['bot']}" for turn in chat_history])
    prompt = "" 
    
    # Logika untuk konteks halaman SAPADAPA
    if chat_context == 'sapadapa_chat':
        # 1. Check for "SAPADAPA" keyword first
        if 'sapadapa' in user_message.lower():
            prompt = """
            Anda adalah AI Agent Aurinova. User bertanya secara spesifik tentang SAPADAPA.
            Jelaskan secara ringkas dan jelas keempat tahap dari kerangka kerja SAPADAPA:
            1.  **Situation Analysis (Analisis Situasi):** Jelaskan tujuannya untuk memahami konteks 'apa yang terjadi?'.
            2.  **Problem Analysis (Analisis Masalah):** Jelaskan tujuannya untuk menemukan akar penyebab 'mengapa ini terjadi?'.
            3.  **Decision Analysis (Analisis Keputusan):** Jelaskan tujuannya untuk memilih tindakan terbaik 'apa yang harus kita lakukan?'.
            4.  **Potential Problem Analysis (Analisis Potensi Masalah):** Jelaskan tujuannya untuk mengantisipasi risiko 'apa yang mungkin salah nanti?'.
            Tutup dengan ajakan untuk mendiskusikan salah satu tahap.
            AI:
            """
        # 2. If no keyword, check if data exists
        elif current_session_data.get("raw_data_head"):
            raw_data_head = current_session_data.get("raw_data_head")
            prompt = f"Anda adalah AI Agent Aurinova. Konteks data user:\n---\n{raw_data_head}\n---\nRiwayat percakapan:\n{formatted_history}\n\nJawab pertanyaan user: \"{user_message}\" berdasarkan data dan riwayat tersebut."
        # 3. If no data and no keyword, act as a general-purpose AI
        else:
            prompt = f"Anda adalah AI Agent Aurinova yang serba bisa. Riwayat percakapan:\n{formatted_history}\n\nJawab pertanyaan umum dari user: \"{user_message}\". Anda belum memiliki akses ke data apa pun."

    # --- Logic for the main clustering app page (index.html) ---
    else: 
        clustering_summary = current_session_data.get("summary")
        if not clustering_summary:
            prompt = f"User bertanya '{user_message}', tapi belum ada hasil klasterisasi. Beri tahu user untuk mengunggah file dan menjalankan analisis."
        else:
            prompt = f"Anda adalah AI yang menganalisis hasil klasterisasi. Ringkasan:\n{clustering_summary}\n\nRiwayat percakapan:\n{formatted_history}\n\nJawab pertanyaan user: \"{user_message}\"\nAI:"
            
    ai_reply = query_openrouter(prompt)
    chat_history.append({"user": user_message, "bot": ai_reply})
    current_session_data["chat_history"] = chat_history
    session_data_cache[session_id] = current_session_data
    return jsonify({'reply': ai_reply})

@app.route('/upload_for_sapadapa', methods=['POST'])
def upload_for_sapadapa():
    session_id = request.form.get('session_id')
    if not session_id:
        return jsonify({'error': 'Session ID is required'}), 400

    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    try:
        original_csv_path = os.path.join(UPLOAD_FOLDER, f'original_data_{session_id}.csv')
        file.save(original_csv_path)

        file_delimiter = request.form.get('delimiter', ',')
        df = pd.read_csv(original_csv_path, encoding='utf-8', sep=file_delimiter)
        
        session_data_cache[session_id] = {
            "raw_data_head": df.head(5).to_string(),
            "original_csv_url": f'/download/original_csv/{session_id}',
            "chat_history": []
        }
        ai_question = "Data Anda berhasil diunggah. Mari kita analisis menggunakan kerangka SAPADAPA. Apakah Anda ingin memulai dengan **Analisis Situasi** untuk memahami gambaran umum data Anda?"
        
        return jsonify({ 'success': True, 'ai_question': ai_question })

    except Exception as e:
        return jsonify({'error': f'Gagal memproses file: {str(e)}'}), 500
    
@app.route('/upload', methods=['POST'])
def upload_file():
    session_id = request.form.get('session_id')
    if not session_id:
        return jsonify({'error': 'Session ID is required'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        unique_id = str(uuid.uuid4())
        img_path = os.path.join(RESULT_FOLDER, f'cluster_plot_{unique_id}.png')
        result_csv_path = os.path.join(RESULT_FOLDER, f'clustered_data_{unique_id}.csv')
        
        file_delimiter = request.form.get('delimiter', ',')
        df = pd.read_csv(file, encoding='utf-8', sep=file_delimiter)

        feature_cols = request.form.get('features').split(',')
        feature_cols = [f.strip() for f in feature_cols if f.strip() in df.columns]

        if not feature_cols:
            raise ValueError("Tidak ada fitur valid yang dipilih.")

        # PROSES DATA
        df_cleaned = clean_and_fill_mean(df.copy())
        df_encoded = encode_categorical(df_cleaned, feature_cols)
        
        if df_encoded.empty or df_encoded.shape[1] == 0:
            raise ValueError("Tidak ada data valid setelah encoding.")
            
        df_scaled = normalize(df_encoded, df_encoded.columns)

        # PROSES KLASTERISASI
        algo = request.form.get('algorithm', 'kmeans')
        n_clusters_req = request.form.get('n_clusters')
        clusters = []

        if algo == 'kmeans':
            if n_clusters_req and n_clusters_req.isdigit() and int(n_clusters_req) > 0:
                n_clusters = int(n_clusters_req)
            else:
                n_clusters = elbow_method(df_scaled)
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = model.fit_predict(df_scaled)
        elif algo == 'dbscan':
            model = DBSCAN(eps=0.5, min_samples=5)
            clusters = model.fit_predict(df_scaled)
        
        df['cluster'] = clusters
        df.to_csv(result_csv_path, index=False)

        # BUAT PLOT (DENGAN PENJAGA)
        plot_url = "" # Defaultnya kosong
        numeric_features_for_plot = df_scaled.select_dtypes(include=np.number).columns
        if len(numeric_features_for_plot) >= 2:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df_scaled[numeric_features_for_plot])
            
            plt.figure(figsize=(10, 7))
            unique_clusters = np.unique(clusters)
            cmap = cm.get_cmap('tab10', len(unique_clusters))
            
            for i, cluster_label in enumerate(unique_clusters):
                points = pca_result[clusters == cluster_label]
                plt.scatter(points[:, 0], points[:, 1], color=cmap(i), label=f'Cluster {cluster_label}')

            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title(f'Hasil Klasterisasi ({algo.upper()})')
            plt.legend()
            plt.grid(True)
            plt.savefig(img_path)
            plt.close()
            plot_url = f'/download/plot/{unique_id}'

        # SIMPAN HASIL KE CACHE
        summary = generate_summary(df, feature_cols, algo, len(np.unique(clusters)))
        session_data_cache[session_id] = {
            "summary": summary,
            "chat_history": []
        }
        
        # KIRIM RESPON JSON YANG KONSISTEN
        return jsonify({
            'clusters': {str(k): int(v) for k, v in df['cluster'].value_counts().items()},
            'features': feature_cols,
            'plot_url': plot_url,
            'csv_url': f'/download/csv/{unique_id}'
        })

    except Exception as e:
        # Jika terjadi error di mana pun, kirim respons error yang jelas
        return jsonify({'error': str(e)}), 500

@app.route('/download/plot/<uid>')
def download_plot(uid):
    filepath = os.path.join(RESULT_FOLDER, f'cluster_plot_{uid}.png')
    if not os.path.exists(filepath):
        return jsonify({'error': 'Plot file not found'}), 404
    return send_file(filepath, mimetype='image/png', as_attachment=False)

@app.route('/download/csv/<uid>')
def download_csv(uid):
    filepath = os.path.join(RESULT_FOLDER, f'clustered_data_{uid}.csv')
    if not os.path.exists(filepath):
        return jsonify({'error': 'CSV file not found'}), 404
    return send_file(filepath, mimetype='text/csv', as_attachment=True, download_name='clustered_data.csv')

@app.route('/download/original_csv/<session_id>') # New endpoint for original CSV
def download_original_csv(session_id):
    filepath = os.path.join(UPLOAD_FOLDER, f'original_data_{session_id}.csv')
    if not os.path.exists(filepath):
        return jsonify({'error': 'Original CSV file not found'}), 404
    return send_file(filepath, mimetype='text/csv', as_attachment=True, download_name='original_data.csv')


@app.route('/get_cached_data/<session_id>', methods=['GET'])
def get_cached_data(session_id):
    cached_data = session_data_cache.get(session_id)
    if cached_data:
        return jsonify({
            'raw_data_head': cached_data.get('raw_data_head', ''),
            'summary': cached_data.get('summary', ''),
            'last_plot_uid': cached_data.get('last_plot_uid'),
            'original_csv_url': cached_data.get('original_csv_url'),
            'chat_history': cached_data.get('chat_history', [])
        })
    return jsonify({'error': 'Data not found in cache for this session.'}), 404

@app.route('/reset_session', methods=['POST'])
def reset_session():
    data = request.json
    session_id_to_reset = data.get('session_id')
    if session_id_to_reset and session_id_to_reset in session_data_cache:
        del session_data_cache[session_id_to_reset]
        return jsonify({'status': 'success', 'message': f'Session {session_id_to_reset} data cleared.'})
    return jsonify({'status': 'failed', 'message': 'Session not found or already cleared.'}), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)