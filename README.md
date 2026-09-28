# MIR — Harmonics on Cretan Music

Music information retrieval study of harmonic structure in Cretan music, set against broader Greek repertoire and current trending recordings. The analysis lives in [`harmony_comparison_analysis.ipynb`](harmony_comparison_analysis.ipynb). It downloads YouTube audio, represents each excerpt with a short-time Fourier transform, reduces that representation to harmonic descriptors, and compares the three playlists in that feature space.

The notebook title block frames the work as an STFT study of harmonic structure for the School of Electrical and Computer Engineering at the National Technical University of Athens.

## Scope

The question is comparative: do traditional Cretan recordings, other Greek recordings, and trending recordings occupy different regions of a harmonic feature space?

Three playlists are loaded from text files of YouTube URLs:

| List | File | Role in the comparison |
| --- | --- | --- |
| Κρητική Μουσική | `1_cretan.txt` | Traditional Cretan repertoire |
| Ελληνική Μουσική | `2_greek.txt` | Broader Greek repertoire |
| Trending Μουσική | `3_trending.txt` | Contemporary trending recordings |

Each item is one YouTube recording. The unit of analysis after feature extraction is one track (a vector of scalar descriptors), labelled by playlist. What is measured is how pitch-class energy, spectral shape, and the balance of harmonic versus percussive energy differ across the three lists.

Tracks enter the corpus only when their duration $T$ satisfies

$$
90\,\mathrm{s} \le T \le 1500\,\mathrm{s}.
$$

Shorter clips are too brief for a stable harmonic summary. Recordings longer than 25 minutes are excluded before download. Unavailable URLs are dropped.

With the notebook defaults (`SAMPLE_SIZE = 0`, `SAMPLE_SIZE_PER_LIST = 0`) every URL that passes the duration gate is analysed. A positive sample size keeps a window of tracks around the median duration of that list.

## Methodology

### Acquisition and temporal coverage

Metadata is read with `yt-dlp` before any media is fetched. Audio that passes the gate is downloaded and decoded to a waveform at the analyser sample rate.

Two listening windows are implemented. The running configuration uses smart sampling (`USE_SMART_SAMPLING = True`). The legacy path keeps only the first $180\,\mathrm{s}$.

Smart sampling draws several excerpts from the body of the track. On recordings longer than six minutes the clip lengths are chosen so that the excerpts add up to about three minutes. Let $T$ be the duration in seconds and $D_{\star} = 180$. The number of clips $M$ and the clip length $d$ (seconds) are

$$
(M, d) =
\begin{cases}
(1,\; T) & T \le 180, \\
\bigl(2,\; \max(30,\, \min(60,\, \lfloor T/2 \rfloor))\bigr) & 180 < T \le 240, \\
\bigl(3,\; \max(30,\, \min(60,\, \lfloor T/3 \rfloor))\bigr) & 240 < T \le 360, \\
\bigl(M_T,\; \max(30,\, \min(60,\, \lfloor D_{\star} / M_T \rfloor))\bigr) & T > 360,
\end{cases}
$$

with $M_T = \max\bigl(3,\, \min(5,\, \lfloor T/90 \rfloor)\bigr)$. Clips are at least $30\,\mathrm{s}$ and at most $60\,\mathrm{s}$. For $T > 180$ the first and last 5% of the recording are left unused, and the $M$ starts are spaced evenly across the remaining span:

$$
t_i = 0.05\,T + i\,\frac{0.90\,T - d}{M - 1}, \qquad i = 0,\ldots,M-1.
$$

Coverage of a track is stored as a percentage of its duration:

$$
\mathrm{coverage} = \frac{100}{T}\sum_{m=1}^{M} d_m.
$$

Each clip is analysed on its own. Scalar descriptors are then combined by an unweighted mean across the $M$ clips (vector descriptors are averaged only when every clip yields the same shape):

$$
\bar f = \frac{1}{M}\sum_{m=1}^{M} f^{(m)}.
$$

### Time–frequency representation

Harmonic descriptors are computed from the short-time Fourier transform of a clip $x[n]$. With window $w$, FFT length $N$, and hop $H$,

$$
X(t, k) = \sum_{n=0}^{N-1} x[n + tH]\, w[n]\, e^{-j 2\pi k n / N}.
$$

The pipeline constructs `HarmonyAnalyzer()` with no arguments, so the transform that is actually applied uses the class defaults:

| Parameter | Value used by the pipeline |
| --- | --- |
| Sample rate $f_s$ | $22050\,\mathrm{Hz}$ |
| FFT length $N$ | $2048$ |
| Hop $H$ | $512$ samples |
| Window | Hann, length $N$ |

Frequency bin spacing and frame rate under those defaults are

$$
\Delta f = \frac{f_s}{N} \approx 10.8\,\mathrm{Hz}, \qquad r = \frac{f_s}{H} \approx 43.1\,\mathrm{frames/s}.
$$

The notebook also defines `OPTIMAL_STFT_PARAMS` ($f_s = 48000\,\mathrm{Hz}$, $N = 4096$, $H = 512$, window length $2048$, Hamming). That dictionary is printed as a recommended higher-resolution setup. The analyser is constructed without it, so the run uses the class defaults. Passing `OPTIMAL_STFT_PARAMS` into the constructor applies this setup instead.

### Harmonic descriptors

`extract_harmonic_features` builds the following quantities from $X$ and from the waveform. `compute_harmony_metrics` collapses them to one number per clip.

**Spectral centroid.** Brightness of frame $t$, with bin frequencies $f_k$:

$$
C(t) = \frac{\sum_k f_k\, |X(t,k)|}{\sum_k |X(t,k)|}.
$$

The track stores the mean and standard deviation of $C(t)$.

**Spectral flux.** Euclidean change of the magnitude spectrum between successive frames:

$$
F(t) = \sqrt{\sum_k \bigl(|X(t,k)| - |X(t-1,k)|\bigr)^2}.
$$

**Spectral rolloff.** The frequency below which 85% of the frame energy lies (librosa’s default rolloff). The metric is the mean over frames.

**Zero-crossing rate.** The fraction of successive samples that change sign, hopped with the same $H$. The metric is the mean rate.

**Chromagram.** A 12-bin pitch-class distribution $C_p(t)$, $p \in \{0,\ldots,11\}$, from `chroma_stft`. Three summaries are taken from the per-class temporal standard deviation $\sigma_p = \mathrm{std}_t\, C_p(t)$:

$$
\mathrm{HC} = \frac{1}{12}\sum_{p=0}^{11} \sigma_p,
\qquad
D = \bigl|\{ p : \sigma_p > \bar\sigma \}\bigr|.
$$

$\mathrm{HC}$ is harmonic complexity. $D$ is tonal diversity, a count of pitch classes whose chroma varies more than the average class.

**Harmonic stability.** Total absolute frame-to-frame change of the chromagram, scaled by the number of frames $T_c$:

$$
S = \left(1 + \frac{1}{T_c}\sum_{p,t} \bigl|C_p(t+1) - C_p(t)\bigr|\right)^{-1}.
$$

$S$ is near $1$ when the pitch-class profile barely moves, and falls as the harmony changes.

**Harmonic–percussive balance.** Harmonic and percussive waveforms $h$ and $p$ come from median-filter HPSS (`librosa.effects.hpss`). Their mean-square energies are

$$
E_h = \frac{1}{L_h}\sum_n h[n]^2, \qquad
E_p = \frac{1}{L_p}\sum_n p[n]^2, \qquad
R_h = \frac{E_h}{E_h + E_p}.
$$

$R_h$ is the harmonic ratio: close to $1$ when the sustained tonal component dominates the percussive residual.

**Spectral entropy.** An unnormalised entropy of the time-averaged magnitude spectrum $\bar S_k = \mathrm{mean}_t \lvert X(t,k)\rvert$. The magnitudes stay on their original scale:

$$
H = -\sum_k \bar S_k \log(\bar S_k + 10^{-10}).
$$

**Cepstral and pitch-class statistics.** Thirteen MFCCs are computed. For the chromagram and the MFCCs the selector also keeps per-bin mean and standard deviation; for centroid, rolloff, and zero-crossing rate it keeps mean, standard deviation, median, and range. The tonal centre is the pitch class with the largest mean chroma energy:

$$
p^{\star} = \arg\max_p \,\mathrm{mean}_t\, C_p(t).
$$

The scalar set used for list comparison is

`mean_spectral_centroid`, `std_spectral_centroid`, `mean_spectral_flux`, `std_spectral_flux`, `mean_spectral_rolloff`, `mean_zcr`, `harmonic_complexity`, `tonal_diversity`, `harmonic_stability`, `harmonic_energy`, `harmonic_ratio`, `spectral_entropy`, and `duration`.

### Quality filters after extraction

A second cleaning pass runs on the assembled table:

1. Drop tracks whose duration column is below $90\,\mathrm{s}$. Under smart sampling that column is the full recording length.
2. Drop tracks with more than 20% missing values among the numeric descriptors.
3. On spectral centroid, harmonic complexity, harmonic ratio, and spectral entropy, drop extreme outliers outside the $3\times\mathrm{IQR}$ fences

$$
\bigl[Q_1 - 3\,\mathrm{IQR},\; Q_3 + 3\,\mathrm{IQR}\bigr], \qquad \mathrm{IQR} = Q_3 - Q_1.
$$

Points between $1.5\,\mathrm{IQR}$ and $3\,\mathrm{IQR}$ are reported and kept.

### Feature selection

Four selectors each keep up to 15 columns. The label is the playlist name.

- **Harmony-based.** The scalar harmony keys above, in that priority order.
- **Mutual information.** $I(X_j; y)$ via `mutual_info_classif`, after standardising columns.
- **ANOVA.** The $F$ statistic of `f_classif` on the same standardised matrix.
- **Random forest.** Mean decrease in impurity from a forest of 100 trees (`random_state = 42`).

Standardisation of column $j$ before the supervised selectors is

$$
z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}.
$$

Agreement of two selected sets $A$ and $B$ is the Jaccard index

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}.
$$

A descriptor is a consensus feature when at least three of the four methods select it. The downstream plots use the consensus list, or the harmony-based list when no feature reaches that agreement. At most 15 consensus features are kept.

### How the lists are compared

Descriptive statistics (mean, standard deviation, minimum, maximum) are grouped by playlist for spectral centroid, harmonic complexity, harmonic ratio, spectral entropy, harmonic stability, and tonal diversity.

Pearson correlation is computed on that same block of columns.

Principal component analysis is applied twice. The visualisation stage standardises the selected numeric features and keeps

$$
n_{\mathrm{PC}} = \min(n_{\mathrm{tracks}},\, n_{\mathrm{features}},\, 10)
$$

components, then scatters the first two. The statistical stage fits a two-component PCA on those six descriptors after filling missing entries with zero, and reports the variance those two axes retain.

K-means uses $k = 3$, one cluster per playlist, with `random_state = 42`. Clustering is done in the six-descriptor space. Cluster centres are projected with that PCA only for the plot. Agreement with the playlist labels is a majority-cluster purity. For playlist $\ell$ with assignments $c_i$,

$$
\mathrm{Acc}(\ell) = \frac{1}{|S_\ell|}\,\bigl|\{ i \in S_\ell : c_i = \mathrm{mode}(c_{S_\ell}) \}\bigr|,
$$

and the reported score is the unweighted mean of $\mathrm{Acc}(\ell)$ over the three lists.

Similarity between playlists uses the mean descriptor vector of each list. Cosine similarity and Euclidean distance are

$$
\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}\cdot\mathbf{v}}{\|\mathbf{u}\|\,\|\mathbf{v}\|}, \qquad
d_2(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2.
$$

The pair with the largest cosine is reported as the closest lists; the pair with the smallest cosine as the furthest.

The chromagram pass prints, for each track, the three strongest pitch classes, the standard deviation of the mean chroma vector, and a stability of the form $\bigl(1 + \sum \lvert \Delta C \rvert\bigr)^{-1}$. List-level bar charts show the mean pitch-class profile, with the top three classes highlighted.

## Analysis pipeline

The notebook runs as one sequence. Later cells assume the tables produced by earlier ones.

1. **Install and import.** `librosa`, `soundfile`, NumPy, Matplotlib, pandas, seaborn, scikit-learn, `yt-dlp`, and `ffmpeg-python`. SciPy supplies the pairwise distances.
2. **Record the STFT settings.** Print `OPTIMAL_STFT_PARAMS`. The analyser object still uses its own defaults, as noted above.
3. **Load the three URL lists** from `1_cretan.txt`, `2_greek.txt`, and `3_trending.txt` under `base_dir`. Lines may hold one URL or several separated by spaces. Only `youtube.com/watch` and `youtu.be` links are kept.
4. **Duration gate, before download.** Keep $90\,\mathrm{s} \le T \le 1500\,\mathrm{s}$. Optionally thin each list to a median-duration window.
5. **Download and clip.** Smart sampling extracts $M$ excerpts; the legacy branch loads the first $180\,\mathrm{s}$.
6. **Features per clip.** STFT, spectral shape, chromagram, HPSS, MFCCs, then the scalar harmony metrics.
7. **Aggregate.** Mean the clip-level descriptors into one row per track, tagged with playlist, duration, coverage, clip count, and URL.
8. **Write the raw table** to `harmony_analysis_smart_sampling.csv` (or `harmony_analysis_legacy.csv`).
9. **Clean the table.** Duration floor, missingness cap, extreme-outlier fences. Write `harmony_analysis_smart_sampling_cleaned.csv` (or the legacy cleaned name).
10. **Select features** with the four methods and keep the consensus subset.
11. **Plot.** Box plots of the leading features by playlist, a correlation heatmap, and a PCA scatter with an explained-variance bar chart.
12. **Summarise.** Group statistics, a second correlation heatmap, and the two-component PCA.
13. **Cluster.** K-means in the descriptor space, with purity against the playlist labels.
14. **Compare lists.** Cosine similarity and Euclidean distance between playlist centroids.
15. **Read the chromagrams.** Dominant pitch classes per track and the mean pitch-class profile of each list.
16. **Print the closing summary** of the STFT setup, the extracted descriptors, the comparison methods, and the ranges observed in the run.

## Getting started

1. Clone the repository:

   ```bash
   git clone https://github.com/panteleimon-a/MIR---Harmonics-on-Cretan-Music.git
   cd MIR---Harmonics-on-Cretan-Music
   ```

2. Install the Python dependencies and a system `ffmpeg` binary on `PATH` (the notebook’s download cells also look for `/opt/homebrew/bin/ffmpeg`):

   ```bash
   pip install librosa soundfile numpy matplotlib pandas seaborn scikit-learn yt-dlp ffmpeg-python scipy jupyter
   ```

3. Point `base_dir` in the URL-loading cell at a directory that contains `1_cretan.txt`, `2_greek.txt`, and `3_trending.txt`.

4. Start Jupyter and open the notebook:

   ```bash
   jupyter notebook harmony_comparison_analysis.ipynb
   ```

Run the cells from the top. The analysis cells expect `music_lists` and then `harmony_df` to exist. `USE_SMART_SAMPLING` and `USE_DATA_CLEANSING` are both on by default.

## Requirements

- Python 3
- Jupyter Notebook
- ffmpeg
- `librosa`, `soundfile`, `numpy`, `matplotlib`, `pandas`, `seaborn`, `scikit-learn`, `scipy`, `yt-dlp`, `ffmpeg-python`

## License

This project is currently unlicensed.
