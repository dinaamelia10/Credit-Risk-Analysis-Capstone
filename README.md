# Analisis Risiko Kredit Menggunakan Machine Learning dan AI

## Overview
Project ini merupakan capstone untuk menganalisis risiko kredit berdasarkan dataset publik. Tujuan utama adalah memprediksi risiko gagal bayar (default) nasabah menggunakan model machine learning (Logistic Regression) dan clustering (KMeans) untuk segmentasi nasabah. Selain itu, AI (LLM seperti Granite dari Replicate dan OpenRouter) digunakan untuk interpretasi hasil clustering, memberikan insight pola nasabah dan faktor risiko dalam bahasa yang mudah dipahami.

Analisis mencakup:
- Preprocessing data (handling missing values, encoding categorical features).
- Modeling prediksi risiko dengan Logistic Regression (akurasi ~80% berdasarkan evaluasi).
- Clustering nasabah menjadi 3 cluster menggunakan KMeans.
- Interpretasi AI untuk pola dan faktor risiko tiap cluster.

## Dataset Link
- Sumber: [Credit Risk Dataset di Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- Deskripsi: Dataset berisi 32.581 baris data nasabah dengan fitur seperti usia, pendapatan, jumlah pinjaman, suku bunga, dll. Target: `loan_status` (0: non-default, 1: default).

## Insight & Findings
- **Distribusi Data**: Usia nasabah rata-rata 27-29 tahun, pendapatan bervariasi dari rendah hingga tinggi. Risiko default lebih tinggi pada nasabah dengan pendapatan rendah dan suku bunga tinggi.
- **Model Prediksi**: Logistic Regression mencapai akurasi ~80%, precision ~70% untuk kelas default. Fitur utama: `loan_percent_income`, `loan_int_rate`, `person_income`.
- **Clustering (3 Cluster)**:
  - **Cluster 0**: Nasabah muda (usia ~27 tahun), pendapatan rendah (~Rp56 juta), pinjaman kecil (~Rp7 juta), suku bunga rendah (~8%). Pola: Konservatif, tapi rentan karena pendapatan terbatas.
  - **Cluster 1**: Nasabah agak lebih tua (~29 tahun), pendapatan tinggi (~Rp103 juta), pinjaman besar (~Rp17 juta), suku bunga sedang (~12%). Pola: Ambisius, tapi beban utang tinggi.
  - **Cluster 2**: Nasabah muda (~27 tahun), pendapatan rendah (~Rp47 juta), pinjaman kecil (~Rp7 juta), suku bunga tinggi (~13%). Pola: Berisiko tinggi, kemampuan finansial lemah.
- **Faktor Risiko Utama**: Pendapatan rendah, suku bunga tinggi, dan beban pinjaman besar meningkatkan risiko default. Cluster 2 paling berisiko.

Visualisasi: Lihat notebook untuk plot seperti distribusi usia, correlation heatmap, dan scatter plot clustering.

## AI Support Explanation
- **Model AI Digunakan**:
  - Granite 3.3-8B-Instruct dari Replicate (untuk interpretasi dasar seperti penjelasan credit risk).
  - DeepSeek R1-Distill-Llama-70B via OpenRouter (untuk interpretasi clustering: pola nasabah dan faktor risiko).
- **Cara Kerja**: Prompt dikirim ke LLM dengan summary statistik cluster (rata-rata fitur). AI menghasilkan penjelasan runtut dalam bahasa Indonesia, memastikan lengkap untuk semua cluster.
- **Manfaat**: AI membantu menerjemahkan data numerik menjadi insight naratif yang mudah dipahami, menghemat waktu analisis manual.
- **Limitasi**: Respons AI bergantung pada prompt; saya gunakan temperature 0.7 untuk kreativitas tapi tetap faktual.

## Cara Run Notebook
- Buka di Google Colab.
- Instal library: `langchain_community`, `replicate`, dll.
- Set API key untuk Replicate dan OpenRouter.
- Jalankan seluruh cell untuk replikasi hasil.

Kontak: Dina Amelia (dina.amelia@example.com)
