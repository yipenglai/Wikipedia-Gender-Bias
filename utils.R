library(dplyr)
library(quanteda)

# Infer pronouns
infer_pronouns <- function(df) {
  female_pronoun <- c("she", "her", "hers")
  male_pronoun <- c("he", "him", "his")
  collective_pronoun <- c("they", "them", "their", "theirs")
  
  # Select pronoun tokens
  token_pronoun <- tokens(df$full_text) %>%
    tokens_tolower %>%   # Convert tokens to lowercase
    tokens_select(c(female_pronoun, male_pronoun, collective_pronoun))

  # Count pronoun frequency
  df_pronoun <- dfm(token_pronoun) %>%
    convert("data.frame") %>%
    mutate(name=df$name,
           male = he+him,
           female = she+her,
           collective = they+them) %>%
    mutate(gender = case_when(
      collective >= male & collective >= female ~ "collective",
      abs(male-female)/(male+female) < 0.25 ~ "unknown",
      female > male ~ "female",
      TRUE ~ "male"
    )) %>%
    select(gender)
  
  df %>% mutate(gender = df_pronoun$gender)
}

# Create preprocessed dfm
create_dfm <- function(df,
                       drop_tokens, 
                       min_termfreq = 0, 
                       min_docfreq = 0) {
  
  token_bio <- tokens(df$full_text,
                      remove_numbers=TRUE,    # Remove numbers
                      remove_symbols=TRUE,    # Remove symbols
                      remove_punct=TRUE) %>%  # Remove punctuation
    tokens_remove(drop_tokens) %>%         # Remove tokens specified
    tokens_remove(stopwords("en")) %>%     # Remove stop words
    tokens_tolower() %>%                   # Convert tokens to lowercase
    tokens_wordstem() %>%                  # Apply stemmer
    tokens_ngrams(n=1:2)                   # Use both unigram and bigram
  
  dfm(token_bio) %>%
    dfm_trim(min_termfreq = min_termfreq,
             min_docfreq = min_docfreq)
}

# Make column sum to 1
colNorm <- function(x) {
  x / outer(rep(1, nrow(x)), colSums(x))
}

# Train test split
train_test_split <- function(df,
                             p = 0.75,
                             down_sample = TRUE) {
  # Train test split
  train_ind <- createDataPartition(df$gender,
                                   p = p,
                                   list = FALSE,
                                   times = 1)
  df_train <- df[train_ind,] %>% select(full_text, gender)
  df_test <- df[-train_ind,] %>% select(full_text, gender)
  
  # Downsample the training data
  if (down_sample) {
    df_train <- downSample(x = df_train$full_text,
                           y = as.factor(df_train$gender),
                           list = FALSE,
                           yname = "gender")
  }
  list(df_train = df_train, df_test = df_test)
}

# Train and evaluate Naive Bayes classifier
train_model <- function(df) {
  # Train test split with downsampling
  list_df <- train_test_split(df, down_sample = TRUE)
  df_train <- list_df$df_train
  df_test <- list_df$df_test
  
  # Create DFMs
  pronouns <- c("she", "She", "her", "Her", "hers", "Hers", "herself", "Herself",
                "He", "he", "him", "Him", "his", "His", "himself", "Himself")
  dfm_train <- create_dfm(df_train %>% mutate(full_text = as.character(x)),
                          drop_tokens = pronouns,
                          min_docfreq = 100,
                          min_termfreq = 100)
  dfm_test <- create_dfm(df_test,
                         drop_tokens = pronouns)
  
  
  # Train Naive Bayes classifier
  model <- textmodel_nb(dfm_train, df_train$gender, smooth = 1)
  
  # Predict labels for test documents
  pred <- predict(model,
                  newdata = dfm_test, 
                  force = TRUE) # Use the same set of features from training
  
  # Evaluate
  cmat <- table(df_test$gender, pred)
  acc <- sum(diag(cmat))/sum(cmat)
  recall <- cmat[2,2]/sum(cmat[2,])
  precision <- cmat[2,2]/sum(cmat[,2])
  f1 <- 2*precision*recall/(precision+recall)
  cat(
    "Accuracy:",  acc, "\n",
    "Recall:",  recall, "\n",
    "Precision:",  precision, "\n",
    "F1 measure", f1, "\n"
  )
  model
}

# Get top discriminative terms
get_top_terms <- function(text_model,
                          top_n = 10) {
  # Compute posterior
  param <- text_model$param
  prior <- text_model$prior
  posterior <- colNorm(param * base::outer(prior, rep(1, ncol(param))))
  # Format table
  posterior <- data.table::transpose(as.data.frame(posterior))
  rownames(posterior) <- colnames(param)
  colnames(posterior) <- rownames(param)
  # Get top predictive terms
  cat("Top predictive terms for female:", "\n",
      posterior %>% arrange(desc(female)) %>% head(top_n) %>% rownames(), "\n",
      "Top predictive terms for male:", "\n",
      posterior %>% arrange(desc(male)) %>% head(top_n) %>% rownames(), "\n")
}