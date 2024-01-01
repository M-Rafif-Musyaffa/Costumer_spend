# Laporan Proyek Machine Learning

### Nama : Muhammad Rafif Musyaffa

### Nim : 211351097

### Kelas : Pagi A

## Domain Proyek

Pada proyek ini bertujuan untuk membantu memahami pola pengeluaran uang pelanggan dan mengidentifikasi kelompok pelanggan dengan perilaku pengeluaran yang serupa. Melalui teknik clustering, kami akan mengelompokkan pelanggan berdasarkan umur, pola pendapatan,dan pola pengeluaran. Hasil analisis klaster ini akan membantu untuk merancang strategi pemasaran yang lebih efektif, menyesuaikan penawaran produk, dan meningkatkan pengalaman pelanggan.

## Business Understanding

Pada proyek ini, masalah bisnis yang ingin diselesaikan adalah untuk memahami perilaku pelanggan berdasarkan pola pendapatan,dan pola pengeluaran mereka. Dengan memahami perilaku pelanggan, perusahaan dapat mengembangkan strategi pemasaran yang lebih efektif.

### Problem Statementsbard

- Bagaimana cara mengkelompokan pelanggan berdasarkan pola pengeluaran dan pola pendapatan?
- Bagaimana pola pengeluaran pelanggan dapat membantu dalam pengambilan keputusan bisnis?
- Bagaimana perusahaan mengukur efektivitas strategi pemasaran yang ada dalam meningkatkan pengeluaran pelanggan?
- Apakah ada peluang untuk meningkatkan penjualan atau kepuasan pelanggan berdasarkan data pengeluaran?

### Goals

- Segmentasi yang tepat dapat membantu perusahaan dalam menyesuaikan strategi pemasaran dan penawaran produk sesuai dengan preferensi dan kebutuhan setiap kelompok pelanggan.
- Memungkinkan perusahaan untuk mengidentifikasi peluang pertumbuhan, meningkatkan efisiensi pemasaran, dan mengoptimalkan strategi penjualan berdasarkan pemahaman yang mendalam tentang perilaku pelanggan.
- Memberikan dasar untuk mengevaluasi dan menyesuaikan strategi pemasaran, mengidentifikasi taktik yang paling sukses, dan meningkatkan pengalaman pelanggan secara keseluruhan.
- Menyediakan landasan untuk pengembangan strategi baru, penyesuaian penawaran produk, dan peningkatan layanan pelanggan untuk meningkatkan performa bisnis secara keseluruhan.

  ### Solution statements

  - Menerapkan algoritma clustering, seperti K-Means atau Hierarchical Clustering, pada dataset pola pengeluaran dan pola pendapatan untuk mengidentifikasi kelompok pelanggan dengan perilaku serupa.
  - Menerapkan strategi pemasaran yang lebih terfokus berdasarkan dari pola pengeluaran, termasuk pengoptimalan penawaran produk dan program loyalitas.
  - Pemanfaatan analisis pola pengeluaran untuk mengidentifikasi peluang pengembangan produk yang lebih sesuai dengan kebutuhan pelanggan.

## Data Understanding

Untuk Proyek ini saya menggunakan data yang berasal dari kaggle.Dataset ini memiliki 8 Atribut,tetapi yang digunakan dalam proyek ini hanya 3.

Dataset yang digunakan: [Customer Spending](https://www.kaggle.com/datasets/goyaladi/customer-spending-dataset).

### Variabel-variabel pada Customer Spending Dataset adalah sebagai berikut:

| Nomer | Variabel           | Tipe Data | Keterangan                           |
| ----- | ------------------ | :-------: | ------------------------------------ |
| 1     | name               |  String   | Berisi Nama nama pelanggan           |
| 2     | age                |  Integer  | Berisi umur pelanggan                |
| 3     | gender             |  String   | Berisi jenis kelamin pelanggan       |
| 4     | education          |  String   | Berisi Tingkat pendidikan pelanggan  |
| 5     | income             |  Integer  | Berisi pendapatan pelanggan          |
| 6     | country            |   Float   | Berisi negara pelanggan              |
| 7     | purchase_frequency |   Float   | Berisi frekuensi pembelian pelanggan |
| 8     | spending           |   Float   | berisi pengeluaran pelanggan         |

#### Variabel-variabel pada Customer Spending Dataset yang digunakan dalam proyek ini adalah sebagai berikut:

| Nomer | Variabel | Tipe Data | Keterangan                   |
| ----- | -------- | :-------: | ---------------------------- |
| 1     | age      |  Integer  | Berisi umur pelanggan        |
| 2     | income   |  Integer  | Berisi pendapatan pelanggan  |
| 3     | spending |   Float   | berisi pengeluaran pelanggan |

**Rubrik/Kriteria Tambahan (Opsional)**:

- Visualisai Data
  ![Visualisasi menggunakan Seaborn](Gambar\Vsd1.png)
  <br>
  ![Visualisasi menggunakan Seaborn](Gambar\Vsd2.png)
  <br>
  ![Visualisasi menggunakan Seaborn](Gambar\Vsd3.png)
  <br>
  ![Visualisasi menggunakan Seaborn](Gambar\Vsd4.png)

## Data Preparation

Beberapa penggunaan algoritma K-Means yang saya gunakan untuk mengelompokan Customer berdasarkan Spending dan Income, teknik persiapan data mencakup langkah-langkah yang saya lakukan ialah :

1. Mencari dataset yang berisi informasi tentang .
2. Mendownload dataset yang sudah dicari dan menload dataset yang akan digunakan.

```python
  !kaggle datasets download -d goyaladi/customer-spending-dataset
  df = pd.read_csv('customer_dataset/customer_data.csv')
```

```python
  !mkdir customer_dataset
  !unzip customer-spending-dataset.zip -d customer_dataset
```

```python
  df = pd.read_csv('customer_dataset/customer_data.csv')
```

3. Memilih library yangrelevan untuk prediksi kualitas air.

```python
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  from sklearn.preprocessing import StandardScaler
  from sklearn.cluster import KMeans
  import plotly.express as px
```

4. Menghapus dataset yang tidak digunakan

```python
  X = df.drop(['name','country','gender','education','purchase_frequency'],axis=1)
```

5. Mengstandarisasi Data

```python
  scaler = StandardScaler()
  scaler.fit(X)
```

```python
  standarized_data = scaler.transform(X)
  print(standarized_data)
```

6. Menentukan jumlah cluster dengan elbow

```python
  k_values = list(range(1, 11))
    inertia_values = []

  for best_k in k_values:
    kmeans = KMeans(n_clusters=best_k,init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=45)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

  # Plot the elbow curve to find the optimal k value
  plt.plot(k_values, inertia_values, marker='o')
  plt.xlabel('Number of Clusters (k)')
  plt.ylabel('Inertia')
  plt.title('Elbow Curve')
  plt.show()
```

7. Proses clustering

```python
  best_k =3
  kmeans = KMeans(n_clusters=best_k,init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=45).fit(X)
  X["Cluster"] = kmeans.labels_
```

8. Proses viusalisasi hasil clustering

```python
  fig = px.scatter_3d(x=X["age"], y=X["income"], z=X["spending"], color=X["Cluster"])

  fig.update_layout(
    title="K-means Clustering",
    scene=dict(
       xaxis_title="Age",
       yaxis_title="Income",
       zaxis_title="Spending",),)
```

## Modeling

Algoritma K-Means adalah salah satu algoritma dalam analisis clustering yang digunakan untuk mengelompokkan data ke dalam kategori yang berbeda secara otomatis. Tujuan algoritma ini adalah untuk mempartisi himpunan data ke dalam kelompok-kelompok (klaster) yang berbeda berdasarkan kemiripan antar data.

Selain itu, tujuan lain dari algoritma K-Means adalah untuk meminimalkan nilai fungsi inersia sehingga klaster-klaster yang dihasilkan memiliki kompak dan terpisah dengan jelas.

```python
  best_k =3
  kmeans = KMeans(n_clusters=best_k,init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=45).fit(X)
```

<br>
- Berikut terdapat beberapa kelebihan yang ada pada algoritma K-Means:

- Algoritma K-Means relatif sederhana dan mudah dipahami. Konsepnya intuitif dan tidak memerlukan pengetahuan matematika atau statistik yang mendalam untuk mengimplementasikannya.
- Algoritma K-Means mampu mengelola dataset dengan jumlah data yang besar dengan efisien. Kompleksitas waktunya berbanding linear dengan jumlah data, sehingga lebih mudah diterapkan pada dataset yang besar.
- K-Means cenderung efisien dalam mengelola dataset dengan dimensi yang tinggi. Meskipun dimensi yang tinggi dapat mempengaruhi kinerja algoritma, K-Means masih mampu memberikan hasil yang dapat diterima dalam dimensi yang cukup tinggi.
- K-Means dapat ditingkatkan dengan menggunakan teknik paralel atau algoritma yang dioptimalkan untuk mengatasi masalah skalabilitas. Ini memungkinkan penggunaan K-Means pada dataset yang semakin besar.
- Hasil dari algoritma K-Means mudah diinterpretasikan. Setiap klaster memiliki pusat klaster yang mewakili kelompok data dalam klaster tersebut. Ini memungkinkan pemahaman yang lebih baik tentang struktur data dan kemiripannya di antara klaster-klaster.
- K-Means dapat diterapkan pada berbagai jenis data dan masalah clustering. Ini tidak terbatas pada jenis data tertentu atau domain tertentu, sehingga dapat digunakan dalam berbagai aplikasi dan bidang.
- Dalam banyak kasus, algoritma K-Means mencapai konvergensi yang cepat. Ini berarti algoritma tersebut cenderung memberikan hasil yang baik dalam waktu yang relatif singkat.
- K-Means dapat membantu mengenali pola dan hubungan antar data. Dengan mengelompokkan data ke dalam klaster, pola atau karakteristik yang serupa dapat ditemukan dalam klaster yang sama.

<br>
- Berikut terdapat beberapa kekurangan dalam penggunaan algoritma K-Means:

- Algoritma K-Means sangat sensitif terhadap inisialisasi awal titik pusat klaster. Untuk mengatasi masalah ini, sering kali dilakukan beberapa percobaan dengan inisialisasi yang berbeda.
- Algoritma K-Means membutuhkan jumlah klaster yang diinginkan (K) sebagai parameter masukan. Pemilihan jumlah klaster yang salah dapat menghasilkan partisi yang tidak memadai atau tidak relevan.
- K-Means mengasumsikan bahwa klaster-klaster memiliki bentuk yang mirip dengan bola dan ukuran yang serupa. Oleh karena itu, algoritma ini tidak efektif untuk mengatasi klaster yang memiliki bentuk yang kompleks, seperti klaster yang berbentuk tidak teratur, elips, atau klaster dengan ukuran yang sangat berbeda-beda.
- K-Means sangat sensitif terhadap adanya outlier dalam data. Algoritma K-Means mungkin tidak mengenali outlier sebagai kelompok terpisah dan cenderung mengintegrasikan mereka ke dalam klaster yang ada.
- Ketika data memiliki dimensi yang tinggi, perhitungan jarak Euclidean yang digunakan dalam algoritma K-Means dapat menjadi tidak efektif. Konsep jarak yang bermakna dalam dimensi yang tinggi menjadi kabur dan dapat menghasilkan partisi yang tidak optimal.
- Algoritma K-Means dapat menghasilkan solusi yang berbeda-beda pada setiap run, terutama ketika terdapat titik pusat klaster yang sama atau jarak antara data yang sama. Ini dapat membuat interpretasi hasil yang sulit dan mengharuskan percobaan ulang dengan inisialisasi yang berbeda.
- K-Means cenderung menghasilkan klaster-klaster yang memiliki ukuran yang seimbang. Jika ada perbedaan yang signifikan dalam jumlah data antara klaster, algoritma ini mungkin tidak mampu menghasilkan partisi yang memadai.
- K-Means memerlukan skala yang seragam antara atribut-atribut yang digunakan. Jika atribut memiliki skala yang berbeda, atribut dengan skala besar akan memiliki pengaruh yang lebih besar pada perhitungan jarak dan dapat mendominasi dalam pembentukan klaster.

## Evaluation

Inertia (juga dikenal sebagai WCSS - Within-Cluster Sum of Squares) adalah metrik evaluasi yang umum digunakan dalam algoritma k-means untuk mengukur sejauh mana titik data dalam suatu kelompok berada dari pusat kelompoknya. Inertia dihitung dengan menjumlahkan kuadrat jarak setiap titik data dalam kelompoknya terhadap pusat kelompoknya.

Secara matematis, inersia untuk kelompok k dihitung dengan rumus berikut:

![Rumus inertia](Gambar\inertia.png)

Tujuan dalam k-means adalah untuk meminimalkan nilai inersia. Semakin kecil inersia, semakin baik, karena ini menunjukkan bahwa titik data dalam kelompok cenderung berada lebih dekat satu sama lain dan dengan pusat kelompok.

Cara umum untuk menggunakan inersia dalam pemilihan jumlah klaster yang optimal adalah dengan melihat "elbow point" pada grafik inersia terhadap jumlah klaster. "Elbow point" adalah titik di mana penurunan inersia tidak lagi signifikan atau menunjukkan penurunan yang lebih lambat. Pada titik ini, penambahan klaster tidak memberikan penurunan inersia yang besar lagi, dan itu bisa menjadi indikator jumlah klaster yang optimal untuk data Anda.

## Deployment

Link Streamlit : [Clustering Customer Spending](https://m-rafif-musyaffa-costumer-spend-k-means-i29pwd.streamlit.app/)
![Streamlit](Gambar\Streamlit.png)
