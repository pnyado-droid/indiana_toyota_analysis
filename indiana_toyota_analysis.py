# Indiana Toyota Dealership Sentiment Analysis
# Unstructured Data Analytics - DealerRater Scraping Project

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import matplotlib.pyplot as plt
import numpy as np

# ----------------------
# SECTION 1: SCRAPE DATA
# ----------------------

# Scraping DealerRater for Indiana Toyota dealerships


def get_dealer_list():
    print("SECTION 1: Scraping DealerRater")
    print("-" * 40)
    
    url = "https://www.dealerrater.com/directory/Indiana/Toyota/"
    print(f"Fetching: {url}")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
    
    link_request = requests.get(url, headers=headers)
    soup = BeautifulSoup(link_request.content, 'html.parser')
    
    # Find dealer links 
    dealer_links = soup.select('a[href*="-review-"]')
    
    dealers = []
    seen_ids = set()
    
    for link in dealer_links:
        href = link.get('href')
        name = link.get_text(strip=True)
        
        # Only main dealer links (ends with /)
        match = re.search(r'/dealer/(.+)-review-(\d+)/$', href)
        if match:
            dealer_id = match.group(2)
            if dealer_id not in seen_ids:
                seen_ids.add(dealer_id)
                clean_name = re.sub(r'^\d+\.', '', name).strip()
                if clean_name and len(clean_name) > 3:
                    dealers.append({
                        'name': clean_name,
                        'url': 'https://www.dealerrater.com' + href.replace('-review-', '-dealer-reviews-'),
                        'id': dealer_id
                    })
    
    print(f"Found {len(dealers)} dealers")
    return dealers

def scrape_dealer_reviews(dealer_url, dealer_name, max_pages=5):
    print(f"  Scraping: {dealer_name}")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
    reviews = []
    
    for page in range(1, max_pages + 1):
        if page == 1:
            page_url = dealer_url
        else:
            page_url = dealer_url + f"page{page}/"
        
        link_request = requests.get(page_url, headers=headers)
        soup = BeautifulSoup(link_request.content, 'html.parser')
        
        # Find reviews using .review-whole selector
        review_elems = soup.select('.review-whole')
        
        if not review_elems:
            break
        
        for elem in review_elems:
            text = elem.get_text(strip=True)
            if text and len(text) > 20:
                reviews.append({
                    'dealer': dealer_name,
                    'review_text': text,
                    'source': 'DealerRater'
                })
        
        time.sleep(1)
    
    print(f"    Found {len(reviews)} reviews")
    return reviews

def get_all_reviews():
    dealers = get_dealer_list()
    
    if not dealers:
        print("ERROR: No dealers found. Cannot proceed.")
        return pd.DataFrame()
    
    all_reviews = []
    
    # Scrape top 20 dealers
    for dealer in dealers[:20]:
        reviews = scrape_dealer_reviews(dealer['url'], dealer['name'])
        all_reviews.extend(reviews)
        time.sleep(2)
    
    df = pd.DataFrame(all_reviews)
    
    if len(df) == 0:
        print("ERROR: No reviews scraped. Cannot proceed.")
        return pd.DataFrame()
    
    print(f"\nTotal reviews: {len(df)}")
    print(df['dealer'].value_counts())
    
    return df

# ---------------------
# SECTION 2: CLEAN DATA
# ---------------------

def clean_reviews(df):
    print("\nSECTION 2: Cleaning Data")
    print("-" * 40)
    
    # Clean text 
    df['clean_review'] = (
        df['review_text']
        .str.replace(r'\[.*?\]', '', regex=True)
        .str.replace(r'([a-z])([A-Z])', r'\1 \2', regex=True)
        .str.replace(r'http\S+|www\.\S+', '', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    
    # Remove short reviews
    df = df[df['clean_review'].str.len() > 10]
    df['word_count'] = df['clean_review'].apply(lambda x: len(x.split()))
    
    df.reset_index(drop=True, inplace=True)
    
    print(f"Cleaned {len(df)} reviews")
    return df

# ----------------------------
# SECTION 3: SENTIMENT ANALYSIS
# ----------------------------

def analyze_sentiment(df):
    print("\nSECTION 3: Sentiment Analysis")
    print("-" * 40)
    
    vader = SentimentIntensityAnalyzer()
    
    # TextBlob polarity 
    df['tb_polarity'] = df['clean_review'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # VADER compound 
    df['vader_compound'] = df['clean_review'].apply(lambda x: vader.polarity_scores(x).get('compound'))
    
    # Classify sentiment
    def classify(score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    df['sentiment'] = df['vader_compound'].apply(classify)
    
    print("Sentiment analysis complete")
    return df

def calculate_dealer_scores(df):
    print("\nCalculating Dealer Scores")
    print("-" * 40)
    
    scores = []
    
    for dealer in df['dealer'].unique():
        dealer_df = df[df['dealer'] == dealer]
        total = len(dealer_df)
        positive = len(dealer_df[dealer_df['sentiment'] == 'Positive'])
        negative = len(dealer_df[dealer_df['sentiment'] == 'Negative'])
        neutral = len(dealer_df[dealer_df['sentiment'] == 'Neutral'])
        
        # Sentiment-based score (% positive)
        sentiment_score = (positive / total * 100) if total > 0 else 0
        
        # DealerRater rating if available
        dr_rating = dealer_df['rating'].iloc[0] if 'rating' in dealer_df.columns else 4.0
        dr_score = (dr_rating / 5) * 100
        
        # Combined score (50% each)
        overall_score = (dr_score * 0.5) + (sentiment_score * 0.5)
        status = "Recommended" if overall_score >= 60 else "Not Recommended"
        avg_vader = dealer_df['vader_compound'].mean()
        
        scores.append({
            'dealer': dealer,
            'total_reviews': total,
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'sentiment_score': round(sentiment_score, 1),
            'dr_rating': dr_rating,
            'dr_score': round(dr_score, 1),
            'overall_score': round(overall_score, 1),
            'avg_vader': round(avg_vader, 3),
            'status': status
        })
    
    df_scores = pd.DataFrame(scores)
    df_scores = df_scores.sort_values('overall_score', ascending=False)
    df_scores.reset_index(drop=True, inplace=True)
    
    # Print scorecard
    print("\nINDIANA TOYOTA DEALERSHIP SCORECARD")
    print("=" * 75)
    print(f"{'DEALER':<25} {'DR RATING':>10} {'SENTIMENT':>12} {'OVERALL':>10} {'STATUS':<15}")
    print("-" * 75)
    for _, row in df_scores.iterrows():
        print(f"{row['dealer']:<25} {row['dr_rating']:>8.1f}* {row['sentiment_score']:>10.1f}% {row['overall_score']:>8.1f}%  {row['status']}")
    print("=" * 75)
    
    return df_scores

# --------------------------
# SECTION 4: VISUALIZE DATA
# --------------------------

def create_visualizations(df_reviews, df_scores, save_path='/Users/pridenyado/Desktop/'):
    print("\nSECTION 4: Creating Visualizations")
    print("-" * 40)
    
    colors_good = '#27ae60'
    colors_bad = '#e74c3c'
    colors_neutral = '#f39c12'
    
    # Figure 1: Dealer Rankings
    fig1, ax1 = plt.subplots(figsize=(14, 10))
    df_plot = df_scores.sort_values('overall_score', ascending=True)
    colors = [colors_good if score >= 60 else colors_bad for score in df_plot['overall_score']]
    
    bars = ax1.barh(df_plot['dealer'], df_plot['overall_score'], color=colors, height=0.6)
    for bar, score in zip(bars, df_plot['overall_score']):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{score:.0f}%', va='center', fontsize=10, fontweight='bold')
    
    ax1.axvline(x=60, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Overall Score (%)')
    ax1.set_title('Indiana Toyota Dealership Rankings\nBased on DealerRater Reviews', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 110)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_good, label='Recommended (>=60%)'),
                      Patch(facecolor=colors_bad, label='Not Recommended (<60%)')]
    ax1.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{save_path}dealer_rankings.png', dpi=300, bbox_inches='tight')
    print(f"Saved: dealer_rankings.png")
    
    # Figure 2: DealerRater Stars vs Sentiment
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    df_plot = df_scores.sort_values('overall_score', ascending=False)
    x = np.arange(len(df_plot))
    width = 0.35
    
    ax2.bar(x - width/2, df_plot['dr_score'], width, label='DealerRater Rating', color='#3498db')
    ax2.bar(x + width/2, df_plot['sentiment_score'], width, label='Review Sentiment', color='#9b59b6')
    
    ax2.set_ylabel('Score (%)')
    ax2.set_title('DealerRater Stars vs Actual Review Sentiment\nDo Official Ratings Match Customer Experiences?', 
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_plot['dealer'], rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(f'{save_path}stars_vs_sentiment.png', dpi=300, bbox_inches='tight')
    print(f"Saved: stars_vs_sentiment.png")
    
    # Figure 3: Sentiment Distribution
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    df_plot = df_scores.sort_values('overall_score', ascending=False)
    
    pos_pct = df_plot['positive'] / df_plot['total_reviews'] * 100
    neu_pct = df_plot['neutral'] / df_plot['total_reviews'] * 100
    neg_pct = df_plot['negative'] / df_plot['total_reviews'] * 100
    
    ax3.barh(df_plot['dealer'], pos_pct, color=colors_good, label='Positive')
    ax3.barh(df_plot['dealer'], neu_pct, left=pos_pct, color=colors_neutral, label='Neutral')
    ax3.barh(df_plot['dealer'], neg_pct, left=pos_pct+neu_pct, color=colors_bad, label='Negative')
    
    ax3.set_xlabel('Percentage of Reviews')
    ax3.set_title('Sentiment Distribution by Dealership\nBreakdown of Review Sentiments', fontsize=14, fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.set_xlim(0, 100)
    plt.tight_layout()
    plt.savefig(f'{save_path}sentiment_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: sentiment_distribution.png")
    
    plt.show()
    print("All visualizations created")


# --------------
# MAIN EXECUTION
# --------------

if __name__ == "__main__":
    print("\nINDIANA TOYOTA DEALERSHIP ANALYSIS")
    print("Unstructured Data Analytics - DealerRater Scraping Project")
    print("=" * 60)
    
    # Section 1: Scrape
    df_reviews = get_all_reviews()
    
    if len(df_reviews) == 0:
        print("No data to analyze. Exiting.")
    else:
        # Section 2: Clean
        df_reviews = clean_reviews(df_reviews)
        
        # Section 3: Sentiment
        df_reviews = analyze_sentiment(df_reviews)
        df_scores = calculate_dealer_scores(df_reviews)
        
        # Section 4: Visualize
        create_visualizations(df_reviews, df_scores)
        
        # Save results
        df_reviews.to_csv('/Users/pridenyado/Desktop/indiana_toyota_reviews.csv', index=False)
        df_scores.to_csv('/Users/pridenyado/Desktop/indiana_toyota_scores.csv', index=False)
        print("\nSaved: indiana_toyota_reviews.csv")
        print("Saved: indiana_toyota_scores.csv")
        print("\nDone!")