# Gender Bias in Wikipedia Biographies

## Data
The dataset include 19.4K English Wikipedia biographies from four categories (artist, athlete, scientist, and politician) with the most biographical articles on Wikiepdia. The data was created by
- Randomly sampling 5K entry names from each category using Wikipedia biography [metadata](https://dumps.wikimedia.org/)
- Retrieving article text for the sampled entries using [`wikipedia` package](https://pypi.org/project/wikipedia/)
