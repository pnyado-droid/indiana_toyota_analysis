# 🚗 Indiana Toyota Dealership Sentiment Analysis

A web scraping and natural language processing project that analyzes customer reviews of Toyota dealerships across Indiana to rank and evaluate dealership performance based on customer sentiment.

---

## 📌 Project Overview

This project scrapes customer reviews from **DealerRater** for Indiana Toyota dealerships and applies sentiment analysis using **TextBlob** and **VADER** to score and rank each dealership. The goal is to surface actionable insights about which dealerships deliver the best — and worst — customer experiences, based entirely on real customer feedback.

---

## 🔍 What It Does

- **Scrapes** up to 5 pages of customer reviews per dealership from DealerRater
- **Cleans and preprocesses** raw review text for analysis
- **Runs dual sentiment analysis** using both TextBlob and VADER
- **Scores and ranks** each dealership by positive sentiment percentage
- **Generates visualizations** to clearly communicate findings
- **Exports** results to CSV for further analysis

---

## 📊 Visualizations

| Chart | Description |
|---|---|
| `dealer_rankings.png` | Horizontal bar chart ranking dealerships by positive sentiment % |
| `positive_vs_negative.png` | Side-by-side comparison of positive vs negative sentiment per dealer |
| `sentiment_distribution.png` | Stacked chart showing full sentiment breakdown (positive/neutral/negative) |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `requests` + `BeautifulSoup` | Web scraping DealerRater |
| `pandas` | Data manipulation and export |
| `TextBlob` | Polarity-based sentiment scoring |
| `VADER` | Compound sentiment scoring (optimized for reviews) |
| `matplotlib` + `numpy` | Data visualization |

---

## 📁 Output Files

```
indiana_toyota_reviews.csv     # All scraped and cleaned reviews with sentiment scores
indiana_toyota_scores.csv      # Dealer-level scorecard with rankings
dealer_rankings.png            # Visualization: dealer rankings
positive_vs_negative.png       # Visualization: pos vs neg by dealer
sentiment_distribution.png     # Visualization: full sentiment breakdown
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/pnyado-droid/indiana_toyota_analysis.git
cd indiana_toyota_analysis
```

**2. Install dependencies**
```bash
pip install requests beautifulsoup4 pandas textblob vaderSentiment matplotlib numpy
```

**3. Run the analysis**

Open `indiana_toyota_analysis.qmd` in VS Code or JupyterLab and run all cells, or render with Quarto:
```bash
quarto render indiana_toyota_analysis.qmd
```

---

## 📈 Scoring Methodology

Each dealership is scored based on the sentiment of its customer reviews:

- **Positive** → VADER compound score ≥ 0.05
- **Neutral** → VADER compound score between -0.05 and 0.05
- **Negative** → VADER compound score ≤ -0.05

**Overall Score** = `Positive% - (Negative% × 0.5)`

Dealerships with **≥ 60% positive reviews** are marked as ✅ **Recommended**.

---

## 👤 Author

**pridenyado** · [GitHub](https://github.com/pnyado-droid)

---

## 📄 License

This project is for educational and analytical purposes only. Review data is sourced from publicly available pages on DealerRater.
