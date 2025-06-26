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
    session_id = data.get('session_id', 'default_session')

    if not user_message:
        return jsonify({'reply': "Tolong masukkan pertanyaan."})

    current_session_data = session_data_cache.get(session_id, {})
    raw_data_head = current_session_data.get("raw_data_head", "")
    clustering_summary = current_session_data.get("summary", "")

    if chat_context == 'sapadapa_chat':
        if raw_data_head:
            prompt = f"""
            Anda adalah AI Agent Aurinova. User telah mengunggah data. Berikut adalah beberapa baris pertama dari data yang diunggah oleh user:
            ---
            {raw_data_head}
            ---
            User sekarang ingin bertanya: "{user_message}"
            Jawab pertanyaan user ini berdasarkan data yang disediakan di atas. Fokus pada memberikan wawasan atau informasi faktual dari data tersebut. Jika pertanyaan user bersifat umum dan tidak terkait langsung dengan data, jawablah secara umum dan fleksibel sebagai AI Agent Aurinova.
            AI:
            """
        else:
            prompt = f"""
            Anda adalah AI Agent Aurinova. User belum mengunggah data untuk sesi ini. User bertanya: "{user_message}"
            Jawab pertanyaan user ini secara umum dan fleksibel. Anda bisa menjelaskan tentang konsep analisis data, klasterisasi, atau kerangka SAPADAPA (Situation Analysis, Problem Analysis, Decision Analysis, Potential Problem Analysis) jika relevan, atau topik umum lainnya.
            AI:
            """
    # Logika chat untuk halaman utama (index.html), tetap fokus pada hasil klasterisasi
    else: # context is 'general' or from '/app'
        if not clustering_summary:
            prompt = f"""
            User belum mengunggah atau mengelompokkan data di aplikasi utama. Jawab pertanyaan user ({user_message})
            dengan menjelaskan bahwa chatbot ini akan sangat berguna setelah data diunggah dan dianalisis di halaman utama.
            Dorong user untuk mengunggah file CSV dan memilih fitur untuk memulai di halaman utama.
            AI:
            """
        else:
            prompt = f"""
            Berikut adalah ringkasan hasil clustering yang telah dilakukan:

            {clustering_summary}

            Sekarang user ingin bertanya:

            User: {user_message}
            AI: Jawab berdasarkan hasil clustering di atas secara ringkas dan jelas. Jika pertanyaan terkait Situation, Problem, Decision, atau Potential Problem Analysis, kaitkan dengan ringkasan clustering yang diberikan.
            """
    ai_reply = query_openrouter(prompt)
    return jsonify({'reply': ai_reply})

@app.route('/upload', methods=['POST'])
def upload_file():
    session_id = request.form.get('session_id', str(uuid.uuid4()))
    
    unique_id = str(uuid.uuid4()) # Ini UID untuk plot/CSV hasil klasterisasi
    img_path = os.path.join(RESULT_FOLDER, f'cluster_plot_{unique_id}.png')
    result_csv_path = os.path.join(RESULT_FOLDER, f'clustered_data_{unique_id}.csv')
    original_csv_path = os.path.join(UPLOAD_FOLDER, f'original_data_{session_id}.csv') # Simpan file asli

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Simpan file yang diunggah ke lokasi aslinya
    file.save(original_csv_path)

    file_delimiter = request.form.get('delimiter', ',') # Delimiter akan tetap default atau dideteksi

    try:
        df = pd.read_csv(original_csv_path, encoding='utf-8', sep=file_delimiter)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(original_csv_path, encoding='latin1', sep=file_delimiter)
        except Exception as e:
            return jsonify({'error': f'Failed to read CSV file with common encodings: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to read CSV file: {str(e)}'}), 400

  
    raw_data_head = df.head(5).to_string() 
    

    feature_cols = request.form.get('features')
    if feature_cols:
        feature_cols = [f.strip() for f in feature_cols.split(',')]
        feature_cols = [f for f in feature_cols if f in df.columns]
    else:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not feature_cols: 
            all_cols = df.columns.tolist()
            if len(all_cols) > 0:
                feature_cols = all_cols
            else:
                return jsonify({'error': 'Tidak ada kolom yang terdeteksi di file CSV.'}), 400


    # Jika setelah deteksi otomatis masih kurang dari 2 fitur (untuk PCA)
    if len(feature_cols) < 2 and len(df) > 1: # Minimal 2 fitur untuk PCA dan klasterisasi
         # Try to pick 2 columns if possible
         if len(df.columns) >= 2:
             feature_cols = df.columns[:2].tolist() # Just pick first two columns
         else:
             # Handle case with only one column or empty df for clustering, though AI can still read it
             pass # Will proceed with less than 2 features for AI, but clustering might fail


    # --- Bagian Klasterisasi (Tetap Berjalan di Backend untuk cache Index.html) ---
    df_encoded = encode_categorical(df.copy(), feature_cols) # Gunakan copy untuk menghindari SettingWithCopyWarning
    df_encoded = clean_and_fill_mean(df_encoded)
    
    # Handle case where df_encoded might become empty or have no columns after encoding
    if df_encoded.empty or df_encoded.shape[1] == 0:
        # If encoding resulted in an empty dataframe (e.g., all invalid columns),
        # then we cannot perform clustering, but AI can still read raw_data_head.
        # Store basic info to cache, but indicate clustering not performed.
        session_data_cache[session_id] = {
            "raw_data_head": raw_data_head,
            "original_csv_url": f'/download/original_csv/{session_id}', # New: link to original CSV
            "summary": "Tidak dapat melakukan klasterisasi dengan fitur yang tersedia. Silakan coba di aplikasi utama.",
            "features": [], # No features used for clustering
            "algorithm": "N/A",
            "n_clusters": 0,
            "last_plot_uid": None # No plot generated
        }
        return jsonify({
            'success': True,
            'message': 'Data berhasil diunggah. AI siap menjawab pertanyaan umum tentang data Anda.',
            'raw_data_head': raw_data_head,
            'session_id': session_id,
            'original_csv_url': f'/download/original_csv/{session_id}'
        })

    df_scaled = normalize(df_encoded, df_encoded.columns)

    if df_scaled.isnull().values.any():
        # Handle NaN after normalization - means some columns were problematic
        session_data_cache[session_id] = {
            "raw_data_head": raw_data_head,
            "original_csv_url": f'/download/original_csv/{session_id}',
            "summary": "Terdapat nilai yang hilang atau tidak valid setelah normalisasi. Klasterisasi tidak dapat dilakukan.",
            "features": feature_cols,
            "algorithm": "N/A",
            "n_clusters": 0,
            "last_plot_uid": None
        }
        return jsonify({
            'success': True,
            'message': 'Data berhasil diunggah, namun ada masalah normalisasi. AI siap menjawab pertanyaan umum.',
            'raw_data_head': raw_data_head,
            'session_id': session_id,
            'original_csv_url': f'/download/original_csv/{session_id}'
        })
    
    # Default parameters for backend clustering (since sapadapa.html doesn't provide them)
    algo = 'kmeans'
    n_clusters_auto = elbow_method(df_scaled)
    clusters = []
    
    # Only perform clustering if there are enough samples and features for meaningful clustering
    if len(df_scaled) >= 2 and df_scaled.shape[1] >= 2: # At least 2 samples and 2 features for PCA/KMeans
        try:
            model = KMeans(n_clusters=n_clusters_auto, random_state=42, n_init=10)
            clusters = model.fit_predict(df_scaled)
            df['cluster'] = clusters
            
            # Generate plot if clustering was successful
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df_scaled)

            plt.figure(figsize=(10, 7))
            unique_clusters = np.unique(clusters)
            num_clusters_to_color = len(unique_clusters)
            cmap = cm.get_cmap('tab10', num_clusters_to_color)

            for i, cluster_label in enumerate(unique_clusters):
                cluster_points = pca_result[clusters == cluster_label]
                legend_label = f'Cluster {cluster_label}'
                if cluster_label == -1:
                    legend_label = 'Noise (-1)'
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    color=cmap(i),
                    label=legend_label,
                    s=50,
                    alpha=0.8
                )

            plt.xlabel('PCA 1')
            plt.ylabel('PCA 2')
            plt.title(f'Clustering Result ({algo}, k={len(set(clusters))})')
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=max(1, min(len(unique_clusters), 4)), fancybox=True, shadow=True)
            plt.subplots_adjust(bottom=0.25, right=0.95)
            plt.grid(True)
            plt.savefig(img_path)
            plt.close()

            df.to_csv(result_csv_path, index=False)
            
            # Store full clustering results
            session_data_cache[session_id] = {
                "raw_data_head": raw_data_head,
                "original_csv_url": f'/download/original_csv/{session_id}',
                "summary": generate_summary(df, feature_cols, algo, len(set(clusters))),
                "features": feature_cols,
                "algorithm": algo,
                "n_clusters": len(set(clusters)),
                "last_plot_uid": unique_id,
                "clustered_csv_url": f'/download/csv/{unique_id}', # Link to clustered CSV
                "plot_url": f'/download/plot/{unique_id}' # Link to plot
            }
            message_for_sapadapa = 'Data Anda berhasil diunggah. AI siap menjawab pertanyaan tentang data Anda dan hasil klasterisasi telah disiapkan untuk aplikasi utama!'
            
        except Exception as e:
            # If clustering fails for any reason, store only raw data info
            session_data_cache[session_id] = {
                "raw_data_head": raw_data_head,
                "original_csv_url": f'/download/original_csv/{session_id}',
                "summary": f"Klasterisasi gagal: {str(e)}. Silakan coba di aplikasi utama jika Anda ingin mengklaster data.",
                "features": feature_cols,
                "algorithm": "N/A",
                "n_clusters": 0,
                "last_plot_uid": None
            }
            message_for_sapadapa = f'Data Anda berhasil diunggah, namun klasterisasi gagal. AI siap menjawab pertanyaan tentang data Anda.'
    else:
        # If not enough features/samples for meaningful clustering
        session_data_cache[session_id] = {
            "raw_data_head": raw_data_head,
            "original_csv_url": f'/download/original_csv/{session_id}',
            "summary": "Data terlalu kecil atau tidak memiliki cukup fitur numerik untuk klasterisasi.",
            "features": feature_cols,
            "algorithm": "N/A",
            "n_clusters": 0,
            "last_plot_uid": None
        }
        message_for_sapadapa = 'Data Anda berhasil diunggah, namun tidak ada cukup fitur untuk klasterisasi. AI siap menjawab pertanyaan tentang data Anda.'

    # Return only what sapadapa.html needs
    return jsonify({
        'success': True,
        'message': message_for_sapadapa,
        'raw_data_head': raw_data_head,
        'session_id': session_id,
        'original_csv_url': f'/download/original_csv/{session_id}' # Provide original CSV download
    })


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
            'summary': cached_data.get('summary', ''), # Still return summary for index.html context
            'last_plot_uid': cached_data.get('last_plot_uid'),
            'original_csv_url': cached_data.get('original_csv_url') # Return original CSV URL
        })
    return jsonify({'error': 'Data not found in cache for this session.'}), 404


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)