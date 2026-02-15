# Predicting Spotify Song Popularity Using Audio Features

**Can a song's audio characteristics predict how popular it will be?**

## Overview

This project investigates whether Spotify's audio features — such as tempo, energy, loudness, and danceability — can predict a song's popularity score. Using a dataset of **550K+ tracks**, we built and compared multiple regression models to understand which musical traits are most associated with listener interest.

The best model (Random Forest) achieved an **R² of 0.49** and **MAE of 9.36**, showing that audio features explain roughly half the variance in popularity — with the remainder driven by external factors like marketing, artist fame, and playlist placement.

## Key Findings

- **"Good for Party"** was the single most important predictor of popularity
- **Loudness**, **track length**, and **tempo** were the next strongest features
- Audio features alone explain ~49% of popularity variance
- The model struggled most with **very popular songs (80+)**, consistently underestimating them — suggesting viral/social factors play a major role at the top
- Similarity scores between tracks added negligible predictive value and were dropped

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Data processing and modeling |
| **Scikit-learn** | Model training and evaluation |
| **Pandas / NumPy** | Data manipulation |
| **Matplotlib** | Visualization |

## Dataset

Sourced from Kaggle's [500K+ Spotify Songs with Lyrics, Emotions & More](https://www.kaggle.com/datasets/devdope/900k-spotify):

- **551,443 tracks** with **39 features** each
- Audio features: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, positiveness, tempo, key, time signature
- Metadata: track title, artist, album, release date, genre, emotion categories
- Target variable: **Popularity** (0–100 scale)

## Methodology

### Preprocessing
1. Converted `track_length` from mm:ss format to seconds
2. One-hot encoded emotion categories, label encoded genre and key
3. Converted loudness from text to numeric
4. Tested and dropped similarity scores (negligible contribution)
5. 80/20 train-test split, no scaling (tree-based models)

### Models Compared

| Model | RMSE | R² | MAE |
|-------|------|----|-----|
| Baseline (Mean) | 17.17 | -0.0001 | 13.48 |
| Linear Regression | 14.80 | 0.2570 | 11.64 |
| **Random Forest** | **12.23** | **0.4927** | **9.36** |

### Feature Importance (Top 10)

1. Good for Party
2. Loudness
3. Track Length
4. Tempo
5. Danceability
6. Positiveness
7. Energy
8. Liveness
9. Acousticness
10. Key (encoded)

<img width="298" height="304" alt="Screenshot 2026-02-15 at 7 05 03 PM" src="https://github.com/user-attachments/assets/aca690d7-a5e5-44f8-a2ba-eadc615b3308" />

## Error Analysis

The model's errors were not uniform across the popularity spectrum:

- **Very popular songs (80–100)**: Highest error — the model consistently underestimates these, suggesting that viral hits are driven by factors beyond audio
- **Mid-range songs (25–60)**: Lowest error — audio features are most predictive here
- **Very low popularity (0–25)**: Moderate error — niche tracks and noise make prediction difficult

<img width="463" height="250" alt="Screenshot 2026-02-15 at 7 05 34 PM" src="https://github.com/user-attachments/assets/f47f383b-962d-433c-bfac-ee2542efcb21" />

## Limitations & Future Work

**Limitations:**
- No time-series or market context information
- Genre categories too broad for subtle distinctions
- Popularity depends on platform and social factors that audio alone can't capture

**Future improvements:**
- Incorporate artist metadata (follower count, previous hits)
- Add release year and temporal trends
- Use lyrics or raw audio as additional features
- Test gradient boosting models (XGBoost, LightGBM)
- Include playlist and chart placement data
