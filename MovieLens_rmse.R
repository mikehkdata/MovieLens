# Dataset download

if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

library(dplyr)
library(tidyverse)
library(caret)
library(data.table)
library(recosystem)

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Setting five decimal places
options(digits = 5)

# Exploratory Data Analysis
dim(edx)
dim(validation)
head(edx)
str(edx)

# rating

n_distinct(edx$rating)

edx %>%
  ggplot(aes(rating, y = ..count..)) +
  geom_bar(color = "black") +
  labs(x = "Ratings", y = "Relative Frequency") +
  scale_x_continuous(breaks = c(0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5)) 

# movieId

n_distinct(edx$movieId)

edx %>% group_by(movieId) %>%
  summarise(count = n()) %>%
  arrange(desc(count)) %>%
  head()

edx %>% group_by(movieId) %>%
  summarise(count = n()) %>%
  arrange(count) %>%
  head()

edx %>% group_by(movieId) %>%
  summarize(count = n()) %>%
  ggplot(aes(count)) +
  geom_histogram(color = "black", bins = 40) +
  labs(x = "No. of ratings per movie", y = "No. of movies") +
  scale_x_log10() 

edx %>% group_by(movieId) %>%
  summarize(mean_rating = mean(rating)) %>%
  ggplot(aes(mean_rating)) +
  geom_histogram(color = "black", bins = 10) 

# userId

n_distinct(edx$userId)

edx %>% group_by(userId) %>%
  summarise(count = n()) %>%
  arrange(desc(count)) %>%
  head()

edx %>% group_by(userId) %>%
  summarise(count = n()) %>%
  arrange(count) %>%
  head()

edx %>% group_by(userId) %>%
  summarize(count = n()) %>%
  ggplot(aes(count)) +
  geom_histogram(color = "black", bins = 40) +
  labs(x = "No. of ratings per user", y = "No. of users") +
  scale_x_log10() 

edx %>% group_by(userId) %>%
  summarise(mean_rating = mean(rating)) %>%
  ggplot(aes(mean_rating)) +
  geom_histogram(color = "black", bins = 10) 

# timestamp

edx <- edx %>% mutate(rating_year = year(as.Date(as.POSIXct(timestamp, origin="1970-01-01"))))
validation <- validation %>% mutate(rating_year = year(as.Date(as.POSIXct(timestamp, origin="1970-01-01"))))
head(edx)
head(validation)

edx %>% ggplot(aes(rating_year, ..count..)) +
  geom_bar(color = "black") +
  labs(x = "rating_year", y = "No. of ratings") 

edx %>% group_by(rating_year) %>%
  summarize(mean_rating = mean(rating)) %>%
  ggplot(aes(mean_rating)) +
  geom_histogram(color = "black", bins = 10)

# Splitting edx dataset into train and test sets 
set.seed(1)
train_index <- createDataPartition(edx$rating, times = 1, p = 0.9, list = FALSE) 
train <- edx[train_index, ]
test_temp <- edx[-train_index, ]

# Matching userId and movieId in both train and test sets
test <- test_temp %>%
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

# Adding back rows into train set
removed <- anti_join(test_temp, test)
train <- rbind(train, removed)
rm(train_index, test_temp, removed)


# Creating RMSE function to evaluate algorithms
RMSE <- function(predicted, actual){
  sqrt(mean((predicted - actual)^2))
}

# Model 1 - Simple mean 
mu <- mean(train$rating)  
mu_rmse <- RMSE(mu, test$rating)
results <- tibble(Model = "Project Goal", RMSE = 0.86490)
results <- bind_rows(results, tibble(Model = "1. Simple mean", RMSE = mu_rmse))
results %>% knitr::kable()

# Model 2 - Including movie bias 
b_movie <- train %>%
  group_by(movieId) %>%
  summarize(b_m = mean(rating - mu))

pred2 <- mu + test %>% 
  left_join(b_movie, by = "movieId") %>% 
  pull(b_m) 
pred2_rmse <- RMSE(pred2, test$rating)
results <- bind_rows(results, tibble(Model = "2. Including movie bias", RMSE = pred2_rmse))
results %>% knitr::kable()

# Model 3 - Including movie and user bias
b_user <- train %>%
  left_join(b_movie, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - b_m - mu))

pred3 <- mu + test %>% 
  left_join(b_movie, by = "movieId") %>%
  left_join(b_user, by = "userId") %>%
  mutate(b_m_u = b_m + b_u) %>% 
  pull(b_m_u)
pred3_rmse <- RMSE(pred3, test$rating)
results <- bind_rows(results, tibble(Model = "3. Including movie and user bias", RMSE = pred3_rmse))
results %>% knitr::kable()

# Model 4 - Including movie, user and year bias
b_year <- train %>% 
  left_join(b_movie, by = "movieId") %>%
  left_join(b_user, by = "userId") %>%
  group_by(rating_year) %>%
  summarize(b_y = mean(rating - b_m - b_u - mu))

pred4 <- mu + test %>%
  left_join(b_movie, by = "movieId") %>%
  left_join(b_user, by = "userId") %>%
  left_join(b_year, by = "rating_year") %>%
  mutate(b_m_u_y = b_m + b_u + b_y) %>%
  pull(b_m_u_y)
pred4_rmse <- RMSE(pred4, test$rating)
results <- bind_rows(results, tibble(Model = "4. Including movie, user and year bias", RMSE = pred4_rmse))
results %>% knitr::kable()

# Model 5 - Including movie and user bias (regularised)

lambdas <- seq(0, 10, 0.25)

pred5_rmses <- sapply(lambdas, function(x){
  
  b_movie <- train %>% 
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu)/(n()+x))
  
  b_user <- train %>% 
    left_join(b_movie, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_m - mu)/(n()+x))
  
  pred5 <- mu + test %>% 
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    mutate(b_m_u = b_m + b_u) %>%
    pull(b_m_u)
  
  return(RMSE(pred5, test$rating))
})

qplot(lambdas, pred5_rmses)
lambda <- lambdas[which.min(pred5_rmses)]
lambda

results <- bind_rows(results, tibble(Model = "5. Including movie and user bias (regularised)", RMSE = min(pred5_rmses)))
results %>% knitr::kable()

# Alternative: Matrix factorisation

train_reco <- with(train, data_memory(user_index = userId, item_index = movieId, rating = rating))
test_reco <- with(test, data_memory(user_index = userId, item_index = movieId, rating = rating))
r <- Reco()
r$train(train_reco)
results_reco <- r$predict(test_reco, out_memory())

mf_rmse <- RMSE(results_reco, test$rating)
results <- bind_rows(results, tibble(Model = "Alternative: Matrix factorization", RMSE = mf_rmse))
results %>% knitr::kable()

######

# Final Validation

v_mu_rmse <- RMSE(mu, validation$rating)
v_results <- tibble(Model = "Project Goal", validation_RMSE = 0.86490)
v_results <- bind_rows(v_results, tibble(Model = "1. Simple mean", validation_RMSE = v_mu_rmse))

v_pred2 <- mu + validation %>% 
  left_join(b_movie, by = "movieId") %>% 
  pull(b_m) 
v_pred2_rmse <- RMSE(v_pred2, validation$rating)
v_results <- bind_rows(v_results, tibble(Model = "2. Including movie bias", validation_RMSE = v_pred2_rmse))

v_pred3 <- mu + validation %>% 
  left_join(b_movie, by = "movieId") %>%
  left_join(b_user, by = "userId") %>%
  mutate(b_m_u = b_m + b_u) %>% 
  pull(b_m_u)
v_pred3_rmse <- RMSE(v_pred3, validation$rating)
v_results <- bind_rows(v_results, tibble(Model = "3. Including movie and user bias", validation_RMSE = v_pred3_rmse))

v_pred4 <- mu + validation %>%
  left_join(b_movie, by = "movieId") %>%
  left_join(b_user, by = "userId") %>%
  left_join(b_year, by = "rating_year") %>%
  mutate(b_m_u_y = b_m + b_u + b_y) %>%
  pull(b_m_u_y)
v_pred4_rmse <- RMSE(v_pred4, validation$rating)
v_results <- bind_rows(v_results, tibble(Model = "4. Including movie, user and year bias", validation_RMSE = v_pred4_rmse))

b_reg_movie <- train %>% 
  group_by(movieId) %>%
  summarize(b_rm = sum(rating - mu)/(n()+lambda))

b_reg_user <- train %>% 
  left_join(b_reg_movie, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_ru = sum(rating - b_rm - mu)/(n()+lambda))

v_pred5 <- mu + validation %>% 
  left_join(b_reg_movie, by = "movieId") %>%
  left_join(b_reg_user, by = "userId") %>%
  mutate(b_rmu = b_rm + b_ru) %>%
  pull(b_rmu)
v_pred5_rmse <- RMSE(v_pred5, validation$rating)
v_results <- bind_rows(v_results, tibble(Model = "5. Including movie and user bias (regularised)", validation_RMSE = v_pred5_rmse))

validation_reco <- with(validation, data_memory(user_index = userId, item_index = movieId, rating = rating))
v_results_reco <- r$predict(validation_reco, out_memory())

v_mf_rmse <- RMSE(v_results_reco, validation$rating)
v_results <- bind_rows(v_results, tibble(Model = "Alternative: Matrix factorization", validation_RMSE = v_mf_rmse))

v_results %>% knitr::kable()


