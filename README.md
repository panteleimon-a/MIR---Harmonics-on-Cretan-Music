# MIR — Harmonics on Cretan Music

Music information retrieval study of harmonic structure in Cretan music, set against broader Greek repertoire and current trending recordings. The analysis lives in [`harmony_comparison_analysis.ipynb`](harmony_comparison_analysis.ipynb).

The notebook title block frames the work as an STFT study of harmonic structure for the School of Electrical and Computer Engineering at the National Technical University of Athens.

## Scope

Traditional music is fading. Playlists fill with whatever is easiest to stream, and the songs that once named a place, a dance, and a way of listening slip out of daily use. People prefer mainstream music, and the roots of our tradition go quiet: the tunes a village already knew, the modes a player was expected to travel through, the sound of a room rather than a studio.

In Crete that root has a voice. The λύρα is a small bowed instrument and a large inheritance. Its power is in the bow as much as in the string: a melody that leans, answers itself, and keeps moving. Under it, the λαούτο does not sit on one chord. Players pass through turns — εναλλαγές — that change the colour of the phrase while the dance stays intact. Heritage here is audible. It is richness of sound: partials, drones, open strings, and a harmony that shifts because the music has somewhere to go.

![A man plays the Cretan lyra at a home table, bow drawn across the strings, while two men sit with him.](images/cretan-lyra.jpg)

*A λύρα at the table. The bow, the instrument, and the room the music was made for.*

This study challenges the promotion of music lacking harmonic progression and chordal richness, as well as the dismissal of traditions that preserve both. By projecting Cretan pieces, other Greek recordings, and current trending tracks into a shared harmonic feature space, we quantify these differences. Specifically, we compare pitch-class energy, spectral envelope, and the ratio of sustained tone to percussive transients. The core question is whether these corpora separate into distinct clusters—that is, whether traditional practices occupy an acoustic domain that flatter, more homogeneous commercial repertoire does not.

Three playlists supply the recordings:

| List | File | Role in the comparison |
| --- | --- | --- |
| Κρητική Μουσική | `1_cretan.txt` | Traditional Cretan repertoire, λύρα and dance tunes among it |
| Ελληνική Μουσική | `2_greek.txt` | Broader Greek repertoire |
| Trending Μουσική | `3_trending.txt` | Contemporary trending recordings |

Each line item is one YouTube recording. After feature extraction the unit of analysis is one track: a vector of scalar descriptors labelled by playlist. A track enters the corpus only when its duration \(T\) satisfies

$$
90\,\mathrm{s} \le T \le 1500\,\mathrm{s}.
$$

## Methodology

The notebook downloads each accepted URL, listens to several excerpts rather than only the opening, and summarises the harmony of those excerpts.

Metadata is read with `yt-dlp` before any file is fetched. The running configuration (`USE_SMART_SAMPLING = True`) then cuts short clips from the body of the piece, skipping the first and last 5% on longer tracks. Clips last between \(30\,\mathrm{s}\) and \(60\,\mathrm{s}\). Short pieces are taken whole; longer ones contribute two to five clips, and past six minutes those clips add up to about three minutes of audio. Descriptors from the clips of one track are averaged into a single row.

Each clip is represented by a short-time Fourier transform. The analyser is constructed with its defaults: sample rate \(f_s = 22050\,\mathrm{Hz}\), FFT length \(N = 2048\), hop \(H = 512\), Hann window. With window \(w\),

$$
X(t, k) = \sum_{n=0}^{N-1} x[n + tH]\, w[n]\, e^{-j 2\pi k n / N}.
$$

From \(X\) and the waveform the notebook keeps a compact harmonic summary: spectral centroid and flux (brightness and how fast the spectrum moves), rolloff, zero-crossing rate, a 12-bin chromagram, thirteen MFCCs, and a harmonic–percussive split. Three of those numbers speak directly to εναλλαγές and chordal colour. Harmonic complexity is the average, over the twelve pitch classes, of how much each class varies in time. Harmonic stability falls as the chromagram changes from frame to frame. The harmonic ratio is the share of energy that stays in the sustained tonal component after the percussive residual is removed. Spectral entropy records how widely that energy is spread across frequency.

Tracks with more than 20% missing descriptors, or with extreme outliers on the main harmonic measures, are dropped. Four selectors (a fixed harmony priority list, mutual information, ANOVA \(F\), and a random forest) each propose up to 15 columns. A descriptor is kept for the plots when at least three methods agree.

The three lists are then compared on the retained features: group statistics, correlation, principal components, K-means with \(k = 3\), and the cosine similarity of each list’s mean vector,

$$
\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}\cdot\mathbf{v}}{\|\mathbf{u}\|\,\|\mathbf{v}\|}.
$$

A chromagram pass closes the notebook: the strongest pitch classes of each track, and the mean pitch-class profile of each playlist.

## Analysis pipeline

1. Load `1_cretan.txt`, `2_greek.txt`, and `3_trending.txt`.
2. Keep URLs whose duration lies between \(90\,\mathrm{s}\) and \(1500\,\mathrm{s}\), before downloading.
3. Download the audio and cut the smart-sampling clips.
4. Compute STFT features and harmony metrics on each clip, then average them per track.
5. Drop short, incomplete, or extreme rows and write the cleaned table to CSV.
6. Select the consensus features.
7. Compare the playlists with statistics, PCA, K-means, cosine similarity, and chromagram profiles.

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

3. Create the three track-list files described under Requirements, and point `base_dir` in the URL-loading cell at the directory that holds them.

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

The track lists are not shipped with the repository. Create these three text files and put them in the folder that `base_dir` points to:

| File | Playlist the notebook assigns |
| --- | --- |
| `1_cretan.txt` | Κρητική Μουσική |
| `2_greek.txt` | Ελληνική Μουσική |
| `3_trending.txt` | Trending Μουσική |

Add one or more YouTube tracks to each file. A line holds either a single URL or several URLs separated by spaces. Blank lines are skipped. A token is kept when it contains `youtube.com/watch` or `youtu.be/`; anything else on the line is ignored.

```text
https://www.youtube.com/watch?v=VIDEO_ID
https://youtu.be/VIDEO_ID

https://youtu.be/VIDEO_ID_1 https://www.youtube.com/watch?v=VIDEO_ID_2
```

Both forms can appear in the same file. Fill `1_cretan.txt` with Cretan repertoire, `2_greek.txt` with other Greek recordings, and `3_trending.txt` with trending tracks. With the notebook defaults every URL that passes the duration gate is analysed.

## License

This project is currently unlicensed.
