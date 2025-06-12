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
import matplotlib.cm as cm # Import matplotlib.cm untuk colormaps

load_dotenv()
clustering_summary_cache = ""
app = Flask(__name__)
CORS(app)

# Konfigurasi batas ukuran konten untuk upload (misal: 100 MB)
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
            df[col].fillna(df[col].mode()[0], inplace=True)
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
        encoded = encoder.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

    numeric_df = df[numeric_cols].copy()
    df_encoded = pd.concat([numeric_df, encoded_df], axis=1)
    return df_encoded

def elbow_method(df_scaled, max_k=10):
    distortions = []
    K = range(1, max_k+1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        distortions.append(kmeans.inertia_)
    deltas = np.diff(distortions)
    second_deltas = np.diff(deltas)
    if len(second_deltas) == 0:
        return 1
    optimal_k = np.argmin(second_deltas) + 2
    return max(1, optimal_k)

def generate_summary(df, feature_cols, algorithm, n_clusters):
    summary = f"Model {algorithm.upper()} berhasil mengelompokkan data menjadi {n_clusters} cluster berdasarkan fitur {', '.join(feature_cols)}.\n"
    for cluster in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster]
        if cluster == -1:
            summary += f"- Cluster -1 (noise) mencakup {len(cluster_data)} data ({len(cluster_data)/len(df)*100:.1f}%)\n"
        else:
            summary += f"- Cluster {cluster} mencakup {len(cluster_data)} data ({len(cluster_data)/len(df)*100:.1f}%) dengan rata-rata:\n"
            for col in feature_cols:
                try:
                    mean_val = cluster_data[col].astype(float).mean()
                    summary += f"   • {col}: {mean_val:.2f}\n"
                except:
                    summary += f"   • {col}: [bukan numerik]\n"
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

@app.route('/app')
def main_app():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'reply': "Tolong masukkan pertanyaan."})

    global clustering_summary_cache
    prompt = f"""
    Berikut adalah ringkasan hasil clustering yang telah dilakukan:

    {clustering_summary_cache}

    Sekarang user ingin bertanya:

    User: {user_message}
    AI: Jawab berdasarkan hasil clustering di atas secara ringkas dan jelas.
    """
    ai_reply = query_openrouter(prompt)
    return jsonify({'reply': ai_reply})

@app.route('/upload', methods=['POST'])
def upload_file():
    # Generate unique_id per request
    unique_id = str(uuid.uuid4())
    img_path = os.path.join(RESULT_FOLDER, f'cluster_plot_{unique_id}.png')
    result_csv_path = os.path.join(RESULT_FOLDER, f'clustered_data_{unique_id}.csv')

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save file to temporary location first
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)

    # Dapatkan delimiter dari frontend (default koma)
    file_delimiter = request.form.get('delimiter', ',')
    
    try:
        # Coba membaca CSV dengan encoding UTF-8, jika gagal coba latin1
        df = pd.read_csv(filename, encoding='utf-8', sep=file_delimiter)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filename, encoding='latin1', sep=file_delimiter)
        except Exception as e:
            return jsonify({'error': f'Failed to read CSV file with common encodings: {str(e)}'}), 400
    except Exception as e: # Catch error selain UnicodeDecodeError (misal ParserError)
        return jsonify({'error': f'Failed to read CSV file: {str(e)}'}), 400

    features = request.form.get('features', None)
    if not features:
        return jsonify({'error': 'No features specified'}), 400
    feature_cols = [f.strip() for f in features.split(',')]

    for f in feature_cols:
        if f not in df.columns:
            return jsonify({'error': f'Feature \"{f}\" not found'}), 400

    df_encoded = encode_categorical(df, feature_cols)
    df_encoded = clean_and_fill_mean(df_encoded)
    df_scaled = normalize(df_encoded, df_encoded.columns)

    if df_scaled.isnull().values.any():
        return jsonify({'error': 'Terdapat nilai NaN setelah normalisasi'}), 400

    algo = request.form.get('algorithm', 'kmeans').lower()
    n_clusters = request.form.get('n_clusters', None)
    if n_clusters:
        try:
            n_clusters = int(n_clusters)
        except ValueError: # Tangani jika konversi ke int gagal
            n_clusters = None # Kembali ke auto-detection

    if algo == 'kmeans':
        if not n_clusters:
            n_clusters = elbow_method(df_scaled)
        model = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = model.fit_predict(df_scaled)
    elif algo == 'dbscan':
        model = DBSCAN(eps=0.5, min_samples=5)
        clusters = model.fit_predict(df_scaled)
    else:
        return jsonify({'error': f'Algorithm {algo} not supported'}), 400

    df['cluster'] = clusters
    global clustering_summary_cache
    clustering_summary_cache = generate_summary(df, feature_cols, algo, len(set(clusters)))

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)

    plt.figure(figsize=(10, 7)) # Ukuran figure lebih lebar untuk legenda

    unique_clusters = np.unique(clusters)
    num_clusters_to_color = len(unique_clusters)
    
    # Menggunakan colormap 'tab10' untuk warna yang lebih distinct
    cmap = cm.get_cmap('tab10', num_clusters_to_color) 

    for i, cluster_label in enumerate(unique_clusters):
        cluster_points = pca_result[clusters == cluster_label]
        legend_label = f'Cluster {cluster_label}'
        if cluster_label == -1: # Handle klaster noise dari DBSCAN (label -1)
            legend_label = 'Noise (-1)'
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            color=cmap(i), # Ambil warna dari colormap berdasarkan indeks
            label=legend_label,
            s=50, # Ukuran marker (opsional)
            alpha=0.8 # Transparansi (opsional)
        )

    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title(f'Clustering Result ({algo}, k={len(set(clusters))})') 

    # >>>>>> MODIFIKASI UNTUK PENEMPATAN LEGEND <<<<<<
    plt.legend(
        loc='lower center', # Posisikan legenda di bagian bawah tengah
        bbox_to_anchor=(0.5, -0.25), # Geser legenda ke bawah dari bawah plot
        ncol=max(1, min(len(unique_clusters), 4)), # Batasi kolom legenda maksimal 4
        fancybox=True, # Efek box yang lebih bagus
        shadow=True # Efek bayangan
    )
    
    # Sesuaikan margin bawah dan kanan plot agar ada ruang untuk legenda
    plt.subplots_adjust(bottom=0.25, right=0.95) 
    plt.grid(True) 
    # >>>>>> AKHIR MODIFIKASI UNTUK PENEMPATAN LEGEND <<<<<<

    plt.savefig(img_path) 
    plt.close()

    df.to_csv(result_csv_path, index=False)

    cluster_counts = {str(c): int(sum(clusters == c)) for c in set(clusters)}

    return jsonify({
        'clusters': cluster_counts,
        'plot_url': f'/download/plot/{unique_id}',
        'csv_url': f'/download/csv/{unique_id}', 
        'features': feature_cols,
        'uid': unique_id
    })


@app.route('/download/plot/<uid>')
def download_plot(uid):
    filepath = os.path.join(RESULT_FOLDER, f'cluster_plot_{uid}.png')
    if not os.path.exists(filepath):
        return jsonify({'error': 'Plot file not found'}), 404
    return send_file(filepath, mimetype='image/png', as_attachment=True, download_name='cluster_plot.png')

@app.route('/download/csv/<uid>')
def download_csv(uid):
    filepath = os.path.join(RESULT_FOLDER, f'clustered_data_{uid}.csv')
    if not os.path.exists(filepath):
        return jsonify({'error': 'CSV file not found'}), 404
    return send_file(filepath, mimetype='text/csv', as_attachment=True, download_name='clustered_data.csv')

@app.route('/summary', methods=['POST'])
def clustering_summary():
    data = request.get_json()
    try:
        df = pd.DataFrame(data['data'])
        feature_cols = data['features']
        algorithm = data['algorithm']
        n_clusters = len(set(df['cluster']))
        summary = generate_summary(df, feature_cols, algorithm, n_clusters)
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)