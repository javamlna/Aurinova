Aplikasi web ini dikembangkan untuk melakukan analisis clustering data secara interaktif menggunakan algoritma K-Means atau DBSCAN. Dilengkapi dengan fitur AI Agent Chatbot untuk membantu pengguna memahami hasil klasterisasi dan memberikan insight data secara otomatis.

Fitur
- User Upload file CSV berisi data pelanggan atau data lain.
- User dapat memilih algoritma K-MEANS atau DBSCAN.
- User memilih fitur apa saja yang ingin digunakan sesuai dengan dataset
- Backend Python (Flask) bersihkan dan normalisasi data.
- Jika fitur yang dipilih data kategorikal maka backend Python akan melakukan encoding.
- Menjalankan klasterisasi K-Means dengan jumlah cluster optimal.
- Pengguna juga dapat memilih jumlah cluster yang diinginkan.
- Menampilkan deskripsi jumlah data per cluster.
- Visualisasi hasil klaster menggunakan scatter plot.
- User dapat download hasil file CSV yang sudah ada label cluster.
- Frontend menggunakan HTML + JavaScript sederhana.
- Chatbot interaktif berbasis Natural Language
- AI Agent Menjawab pertanyaan sesuai user input
- User dapat insight dari AI Agent
- AI Agent menggunakan OpenRouter openai/gpt-4o-mini
