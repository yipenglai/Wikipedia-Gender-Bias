# Gender Bias in Wikipedia Biographies

## Objective
Wikipedia has been widely used as the training corpus for mnay different natural language processing (NLP) models. Meanwhile, according to [a survey] conducted by Wikimedia foundation, fewer than 18% of Wikipedia biographies are about women. To understand how women are represented in English Wikipedia biographies, this project aims to answer the following questions through text analysis in R:
- To what extent do biographies about female subjects differ from those about male subjects?
- What terms are more strongly associated with each gender?

## Key Findings
- By training classification models to predict subject gender based on biography text, we find that our models outperform no-skill, random baseline. In other words, __Wikipedia editors tend to use different words and/or expressions when describing female and male subjects.__
- By looking at the most predictive terms for each gender, we find that __Wikipedia editors are more likely to use terms relating to gender (e.g. "female", "woman") and family (e.g. "husband", "mother") in biographies about female,__ suggesting that male is often considered the norm.

## Data
The dataset include 19.4K English Wikipedia biographies from four categories (artist, athlete, scientist, and politician) with the most biographical articles on Wikiepdia. The data was created by
- Randomly sampling 5K entry names from each category using Wikipedia biography [metadata](https://dumps.wikimedia.org/)
- Retrieving article text for the sampled entries using [`wikipedia` package](https://pypi.org/project/wikipedia/)
