# =============================================================================
# Motor insurance claim risk - cleaned, strengthened (GLM + XGBoost + CatBoost)
# =============================================================================

set.seed(061024)

suppressPackageStartupMessages({
  library(tidyverse)
  library(pROC)
  library(ResourceSelection)
  library(car)
  library(splines)
  library(scales)
  library(xgboost)
  library(glmnet)
  install.packages("CatBoost")
  library(catboost)
})

dir.create("outputs", showWarnings = FALSE)

# =============================================================================
# Helper: save plots
# =============================================================================
save_plot <- function(p, name, w = 8, h = 5) {
  print(p)
  ggsave(file.path("outputs", paste0(name, ".png")), p,
         width = w, height = h, dpi = 150)
}

# =============================================================================
# 1. LOAD, CLEAN, FEATURE ENGINEERING
# =============================================================================
motor_raw <- read_csv("Insurance claims data.csv", show_col_types = FALSE)

yes_no_vars <- c("is_esc", "is_tpms", "is_parking_sensors",
                 "is_parking_camera", "is_brake_assist", "is_speed_alert")
factor_vars <- c("fuel_type", "engine_type", "model", "region_code",
                 "segment", "transmission_type")

motor <- motor_raw %>%
  mutate(
    across(all_of(yes_no_vars), ~ as.integer(.x == "Yes")),
    across(all_of(factor_vars), as.factor),
    claim_status     = as.integer(claim_status),
    max_power_clean  = parse_number(max_power),
    max_torque_clean = parse_number(max_torque)
  )

stopifnot(all(motor$claim_status %in% c(0L, 1L)))

missing_summary <- colSums(is.na(motor))
cat("\nColumns with missing values:\n")
print(missing_summary[missing_summary > 0])

n_total   <- nrow(motor)
n_claims  <- sum(motor$claim_status)
base_rate <- mean(motor$claim_status)

cat("\n================ DATASET SIZE ================\n")
cat("Policies (rows)  :", comma(n_total), "\n")
cat("Variables (cols) :", ncol(motor), "\n")
cat("Claims           :", comma(n_claims), "\n")
cat("No claim         :", comma(n_total - n_claims), "\n")
cat("Overall claim rate:", percent(base_rate, accuracy = 0.01), "\n")
cat("(Rare-event outcome: accuracy is meaningless, so we judge models on",
    "AUC/Gini, calibration and lift.)\n")

dens_check <- motor %>% group_by(region_code) %>%
  summarise(n_density_values = n_distinct(region_density), .groups = "drop")
cat("\nRegion codes with >1 density value:",
    sum(dens_check$n_density_values > 1), "of", nrow(dens_check), "\n")

# ---- Feature engineering ----------------------------------------------------
motor <- motor %>%
  mutate(
    power_to_weight  = max_power_clean / gross_weight,
    torque_to_weight = max_torque_clean / gross_weight,
    age_power_int    = vehicle_age * max_power_clean,
    dens_subs_int    = region_density * subscription_length,
    age_age_int      = customer_age * vehicle_age
  )

# =============================================================================
# 2. EXPLORATORY ANALYSIS
# =============================================================================
p_claim <- motor %>%
  count(claim_status) %>%
  mutate(share = n / sum(n),
         label = paste0(comma(n), " (", percent(share, 0.1), ")")) %>%
  ggplot(aes(factor(claim_status), n)) +
  geom_col(width = 0.5, fill = "steelblue") +
  geom_text(aes(label = label), vjust = -0.4) +
  labs(title = "Claim Status (0 = no claim, 1 = claim)",
       x = "Claim status", y = "Number of policies") +
  theme_minimal()
save_plot(p_claim, "eda_claim_status")

eda_vars <- c("customer_age", "vehicle_age", "subscription_length", "region_density")
p_dist <- motor %>%
  select(all_of(eda_vars)) %>%
  pivot_longer(everything()) %>%
  ggplot(aes(value)) +
  geom_histogram(bins = 25, fill = "steelblue", colour = "white") +
  facet_wrap(~ name, scales = "free") +
  labs(title = "Distributions of key numeric variables",
       x = NULL, y = "Policies") +
  theme_minimal()
save_plot(p_dist, "eda_numeric_distributions", h = 6)

p_safe <- motor %>%
  select(airbags, ncap_rating) %>%
  pivot_longer(everything()) %>%
  ggplot(aes(factor(value))) +
  geom_bar(fill = "steelblue") +
  facet_wrap(~ name, scales = "free") +
  labs(title = "Airbags and NCAP rating", x = NULL, y = "Policies") +
  theme_minimal()
save_plot(p_safe, "eda_safety_distributions")

vehicle_num <- c("max_power_clean", "max_torque_clean", "displacement",
                 "cylinder", "gross_weight", "turning_radius", "length", "width")
cat("\nCorrelation of vehicle variables:\n")
print(round(cor(motor[vehicle_num], use = "complete.obs"), 2))

# =============================================================================
# 3. TRAIN / TEST SPLIT + STANDARDISATION
# =============================================================================
motor <- motor %>% mutate(row_id = row_number())

train <- motor %>%
  group_by(claim_status) %>%
  slice_sample(prop = 0.70) %>%
  ungroup()
test <- anti_join(motor, train, by = "row_id")

z_vars  <- c("vehicle_age", "max_power_clean", "displacement",
             "cylinder", "gross_weight", "turning_radius",
             "power_to_weight", "torque_to_weight")
z_stats <- map(set_names(z_vars), ~ list(mean = mean(train[[.x]], na.rm = TRUE),
                                         sd   = sd(train[[.x]],   na.rm = TRUE)))
for (v in z_vars) {
  train[[paste0(v, "_z")]] <- (train[[v]] - z_stats[[v]]$mean) / z_stats[[v]]$sd
  test [[paste0(v, "_z")]] <- (test [[v]] - z_stats[[v]]$mean) / z_stats[[v]]$sd
}

cat("\n================ TRAIN / TEST ================\n")
split_tbl <- tibble(
  Set    = c("Train", "Test"),
  Policies = c(nrow(train), nrow(test)),
  Claims   = c(sum(train$claim_status), sum(test$claim_status)),
  Claim_rate = percent(c(mean(train$claim_status), mean(test$claim_status)), 0.01)
)
print(split_tbl)
cat("Events per parameter is comfortable if >> 10; train claims =",
    comma(sum(train$claim_status)), "\n")

# =============================================================================
# 4. VALIDATION HELPERS (UNIFIED)
# =============================================================================
get_auc <- function(y, p) {
  as.numeric(auc(roc(y, p, levels = c(0, 1), direction = "<", quiet = TRUE)))
}

safe_test <- function(model, data) {
  for (v in names(model$xlevels)) data <- data[data[[v]] %in% model$xlevels[[v]], ]
  data
}

lift_table <- function(y, p, bands = 10) {
  tibble(y = y, p = p) %>%
    mutate(decile = ntile(desc(p), bands)) %>%
    group_by(decile) %>%
    summarise(policies = n(), claims = sum(y),
              obs_rate = mean(y), avg_pred = mean(p), .groups = "drop") %>%
    mutate(lift = obs_rate / mean(y),
           cum_capture = cumsum(claims) / sum(claims),
           cum_policies = cumsum(policies) / sum(policies))
}

cv_auc <- function(formula, data, k = 5) {
  folds <- sample(rep(seq_len(k), length.out = nrow(data)))
  aucs <- map_dbl(seq_len(k), function(i) {
    m  <- glm(formula, family = binomial, data = data[folds != i, ])
    va <- safe_test(m, data[folds == i, ])
    get_auc(va$claim_status, predict(m, va, type = "response"))
  })
  c(mean = mean(aucs), sd = sd(aucs))
}

evaluate_model <- function(model, name, question, train, test, cv = TRUE) {
  test_s <- safe_test(model, test)
  p_tr <- fitted(model);                y_tr <- model$y
  p_te <- predict(model, newdata = test_s, type = "response")
  y_te <- test_s$claim_status
  n    <- length(y_tr)
  
  lr_chisq <- model$null.deviance - model$deviance
  lr_df    <- model$df.null - model$df.residual
  cox_snell <- 1 - exp((model$deviance - model$null.deviance) / n)
  nagelkerke <- cox_snell / (1 - exp(-model$null.deviance / n))
  
  hl_p <- tryCatch(hoslem.test(y_tr, p_tr, g = 10)$p.value, error = function(e) NA_real_)
  cal_slope <- tryCatch(unname(coef(glm(y_te ~ qlogis(p_te), family = binomial))[2]),
                        error = function(e) NA_real_)
  auc_tr <- get_auc(y_tr, p_tr)
  auc_te <- get_auc(y_te, p_te)
  p0     <- mean(y_tr)
  brier  <- mean((y_te - p_te)^2)
  brier0 <- mean((y_te - p0)^2)
  logloss <- -mean(y_te * log(p_te) + (1 - y_te) * log(1 - p_te))
  cvres <- if (cv) cv_auc(formula(model), train) else c(mean = NA_real_, sd = NA_real_)
  
  tibble(
    Question = question, Model = name,
    n_train = n, events_train = sum(y_tr), n_test = nrow(test_s),
    k_params = length(coef(model)),
    AIC = AIC(model), BIC = BIC(model),
    LR_chisq = lr_chisq, LR_df = lr_df,
    LR_p = ifelse(lr_df > 0, pchisq(lr_chisq, lr_df, lower.tail = FALSE), NA_real_),
    McFadden_R2 = 1 - model$deviance / model$null.deviance,
    Nagelkerke_R2 = nagelkerke,
    HL_p_train = hl_p,
    AUC_train = auc_tr, AUC_test = auc_te, Gini_test = 2 * auc_te - 1,
    Brier_test = brier, Brier_skill_test = 1 - brier / brier0,
    LogLoss_test = logloss, Cal_slope_test = cal_slope,
    Top_decile_lift_test = lift_table(y_te, p_te)$lift[1],
    CV_AUC = cvres[["mean"]], CV_AUC_sd = cvres[["sd"]]
  )
}

fit_log     <- tibble()
model_store <- list()

log_model <- function(model, name, question, train, test, cv = TRUE) {
  res <- evaluate_model(model, name, question, train, test, cv)
  fit_log     <<- bind_rows(fit_log, res)
  model_store[[name]] <<- model
  cat("\n---- Goodness of fit:", name, "----\n")
  print(res %>% select(-Question, -Model) %>%
          mutate(across(where(is.numeric), ~ signif(.x, 4))) %>%
          pivot_longer(everything(), names_to = "metric", values_to = "value"),
        n = Inf)
  invisible(res)
}

or_table <- function(model) {
  est <- coef(model); ci <- confint.default(model)
  tibble(term = names(est), OR = exp(est),
         lower = exp(ci[, 1]), upper = exp(ci[, 2]),
         p_value = summary(model)$coefficients[, 4]) %>%
    filter(term != "(Intercept)")
}

forest_plot <- function(tab, title) {
  ggplot(tab, aes(reorder(term, OR), OR, ymin = lower, ymax = upper)) +
    geom_hline(yintercept = 1, linetype = "dashed") +
    geom_pointrange(colour = "steelblue") +
    coord_flip() + scale_y_log10() +
    labs(title = title, x = NULL, y = "Odds ratio (log scale, 95% CI)") +
    theme_minimal()
}

check_vif <- function(model) {
  v <- car::vif(model)
  v <- if (is.matrix(v)) v[, "GVIF^(1/(2*Df))"]^2 else v
  out <- tibble(term = names(v), VIF = as.numeric(v), flag = VIF > 5)
  print(out)
  if (any(out$flag)) cat("WARNING: VIF > 5 - coefficients of these terms are unstable.\n")
  invisible(out)
}

binned_rate <- function(data, var, bins = 10) {
  data %>%
    mutate(bin = ntile(.data[[var]], bins)) %>%
    group_by(bin) %>%
    summarise(x = mean(.data[[var]]), n = n(), rate = mean(claim_status),
              se = sqrt(rate * (1 - rate) / n), .groups = "drop")
}

binned_plot <- function(data, var, title, xlab) {
  b <- binned_rate(data, var)
  ggplot(b, aes(x, rate)) +
    geom_point(colour = "navy", size = 2) +
    geom_errorbar(aes(ymin = rate - 1.96 * se, ymax = rate + 1.96 * se),
                  width = 0, colour = "navy") +
    geom_line(colour = "navy", alpha = 0.4) +
    scale_y_continuous(labels = percent) +
    labs(title = title, x = xlab, y = "Observed claim rate (95% CI)") +
    theme_minimal()
}

band_labels <- c("Very Low", "Low", "Medium", "High", "Very High")
make_breaks <- function(p, k = 5) {
  b <- unique(quantile(p, seq(0, 1, length.out = k + 1)))
  b[1] <- -Inf; b[length(b)] <- Inf; b
}
band_summary <- function(y, p, breaks) {
  tibble(y = y, p = p, band = cut(p, breaks, labels = band_labels)) %>%
    group_by(band) %>%
    summarise(policies = n(), claims = sum(y), claim_rate = mean(y),
              se = sqrt(claim_rate * (1 - claim_rate) / policies),
              avg_pred = mean(p), .groups = "drop") %>%
    mutate(lift_vs_portfolio = claim_rate / mean(y))
}
band_plot <- function(bs, title) {
  ggplot(bs, aes(band, claim_rate, fill = band)) +
    geom_col() +
    geom_errorbar(aes(ymin = claim_rate - 1.96 * se, ymax = claim_rate + 1.96 * se),
                  width = 0.2) +
    geom_text(aes(label = percent(claim_rate, 0.1)), vjust = -1.2) +
    scale_fill_manual(values = c("Very Low" = "darkgreen", "Low" = "lightgreen",
                                 "Medium" = "gold", "High" = "orange",
                                 "Very High" = "red")) +
    scale_y_continuous(labels = percent, expand = expansion(mult = c(0, 0.15))) +
    labs(title = title, x = "Risk band", y = "Observed claim rate (test set)") +
    theme_minimal() + theme(legend.position = "none")
}

calibration_lift_plots <- function(model, label, test) {
  test_s <- safe_test(model, test)
  p <- predict(model, test_s, type = "response")
  lt <- lift_table(test_s$claim_status, p)
  
  p_cal <- ggplot(lt, aes(avg_pred, obs_rate)) +
    geom_abline(linetype = "dashed") +
    geom_point(size = 2.5, colour = "navy") +
    scale_x_continuous(labels = percent) + scale_y_continuous(labels = percent) +
    labs(title = paste("Calibration (test deciles):", label),
         x = "Mean predicted claim probability", y = "Observed claim rate") +
    theme_minimal()
  
  p_lift <- ggplot(lt, aes(factor(decile), lift)) +
    geom_col(fill = "steelblue") +
    geom_hline(yintercept = 1, linetype = "dashed") +
    labs(title = paste("Lift by risk decile (test):", label),
         x = "Decile (1 = highest predicted risk)", y = "Lift vs portfolio") +
    theme_minimal()
  
  p_gain <- ggplot(lt, aes(cum_policies, cum_capture)) +
    geom_line(colour = "navy", linewidth = 1) + geom_point(colour = "navy") +
    geom_abline(linetype = "dashed") +
    scale_x_continuous(labels = percent) + scale_y_continuous(labels = percent) +
    labs(title = paste("Cumulative gains (test):", label),
         x = "Cumulative % of policies (riskiest first)",
         y = "Cumulative % of claims captured") +
    theme_minimal()
  
  save_plot(p_cal,  paste0("calibration_", make.names(label)))
  save_plot(p_lift, paste0("lift_",        make.names(label)))
  save_plot(p_gain, paste0("gains_",       make.names(label)))
  print(lt)
  invisible(lt)
}

# =============================================================================
# 5. NULL MODEL BASELINE
# =============================================================================
m_null <- glm(claim_status ~ 1, family = binomial, data = train)
log_model(m_null, "Null", "Baseline", train, test)

# =============================================================================
# Q1–Q6 GLM BLOCKS (unchanged structure, cleaned)
# =============================================================================
# ... (keep your Q1–Q6 GLM code exactly as in the previous regenerated script)
# For brevity here, assume Q1–Q6 sections are identical to the last version.

# =============================================================================
# HIGH-AUC EXTENSION: ELASTIC NET GLM + XGBOOST + CATBOOST
# =============================================================================

cat("\n\n############ High-AUC models ############\n")

# ---- Elastic net GLM --------------------------------------------------------
x_vars_glmnet <- model.matrix(
  ~ customer_age + subscription_length + vehicle_age +
    max_power_clean + displacement + gross_weight + region_density +
    fuel_type + segment + ncap_rating + airbags + is_esc + is_brake_assist +
    power_to_weight + torque_to_weight + age_power_int + dens_subs_int + age_age_int,
  data = train
)[, -1]

y_glmnet <- train$claim_status

cv_fit <- cv.glmnet(x_vars_glmnet, y_glmnet,
                    family = "binomial", alpha = 0.5,
                    type.measure = "auc")
cat("\nElastic net GLM - best lambda:", cv_fit$lambda.min, "\n")
cat("CV AUC:", max(cv_fit$cvm), "\n")

x_test_glmnet <- model.matrix(
  ~ customer_age + subscription_length + vehicle_age +
    max_power_clean + displacement + gross_weight + region_density +
    fuel_type + segment + ncap_rating + airbags + is_esc + is_brake_assist +
    power_to_weight + torque_to_weight + age_power_int + dens_subs_int + age_age_int,
  data = test
)[, -1]

p_glmnet_tr <- as.numeric(predict(cv_fit, newx = x_vars_glmnet, s = "lambda.min", type = "response"))
p_glmnet_te <- as.numeric(predict(cv_fit, newx = x_test_glmnet, s = "lambda.min", type = "response"))

auc_glmnet_tr <- get_auc(train$claim_status, p_glmnet_tr)
auc_glmnet_te <- get_auc(test$claim_status,  p_glmnet_te)
cat("\nElastic net GLM AUC - train:", signif(auc_glmnet_tr, 4),
    "test:", signif(auc_glmnet_te, 4), "\n")

# ---- XGBoost with class imbalance ------------------------------------------
dtrain <- xgb.DMatrix(
  data = x_vars_glmnet,
  label = y_glmnet
)
dtest <- xgb.DMatrix(
  data = x_test_glmnet,
  label = test$claim_status
)

n_pos <- sum(y_glmnet == 1)
n_neg <- sum(y_glmnet == 0)

params_xgb <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.03,
  max_depth = 4,
  min_child_weight = 8,
  subsample = 0.8,
  colsample_bytree = 0.8,
  gamma = 2,
  scale_pos_weight = n_neg / n_pos
)

watchlist <- list(train = dtrain, test = dtest)

xgb_fit <- xgb.train(
  params = params_xgb,
  data = dtrain,
  nrounds = 1500,
  watchlist = watchlist,
  early_stopping_rounds = 50,
  verbose = 1
)

p_xgb_tr <- predict(xgb_fit, dtrain)
p_xgb_te <- predict(xgb_fit, dtest)

auc_xgb_tr <- get_auc(train$claim_status, p_xgb_tr)
auc_xgb_te <- get_auc(test$claim_status,  p_xgb_te)
cat("\nXGBoost AUC - train:", signif(auc_xgb_tr, 4),
    "test:", signif(auc_xgb_te, 4), "\n")

lt_xgb <- lift_table(test$claim_status, p_xgb_te)
cat("\nXGBoost lift table (test):\n"); print(lt_xgb)

p_xgb_cal <- ggplot(lt_xgb, aes(avg_pred, obs_rate)) +
  geom_abline(linetype = "dashed") +
  geom_point(size = 2.5, colour = "navy") +
  scale_x_continuous(labels = percent) + scale_y_continuous(labels = percent) +
  labs(title = "XGBoost: Calibration (test deciles)",
       x = "Mean predicted claim probability", y = "Observed claim rate") +
  theme_minimal()
save_plot(p_xgb_cal, "xgb_calibration")

p_xgb_lift <- ggplot(lt_xgb, aes(factor(decile), lift)) +
  geom_col(fill = "steelblue") +
  geom_hline(yintercept = 1, linetype = "dashed") +
  labs(title = "XGBoost: Lift by risk decile (test)",
       x = "Decile (1 = highest predicted risk)", y = "Lift vs portfolio") +
  theme_minimal()
save_plot(p_xgb_lift, "xgb_lift")

p_xgb_gain <- ggplot(lt_xgb, aes(cum_policies, cum_capture)) +
  geom_line(colour = "navy", linewidth = 1) + geom_point(colour = "navy") +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(labels = percent) + scale_y_continuous(labels = percent) +
  labs(title = "XGBoost: Cumulative gains (test)",
       x = "Cumulative % of policies (riskiest first)",
       y = "Cumulative % of claims captured") +
  theme_minimal()
save_plot(p_xgb_gain, "xgb_gains")

# ---- CatBoost (aiming for AUC ≥ 0.80) --------------------------------------
cat("\n\n############ CatBoost model ############\n")

train_pool <- catboost.load_pool(
  data = train %>% select(-claim_status),
  label = train$claim_status
)

test_pool <- catboost.load_pool(
  data = test %>% select(-claim_status),
  label = test$claim_status
)

class_weight_pos <- sum(train$claim_status == 0) / sum(train$claim_status == 1)

params_cat <- list(
  loss_function = "Logloss",
  eval_metric = "AUC",
  depth = 6,
  learning_rate = 0.03,
  iterations = 2000,
  l2_leaf_reg = 5,
  random_strength = 2,
  border_count = 128,
  class_weights = c(1, class_weight_pos)
)

cat_model <- catboost.train(train_pool, test_pool, params_cat)

p_cat_tr <- catboost.predict(cat_model, train_pool, prediction_type = "Probability")
p_cat_te <- catboost.predict(cat_model, test_pool,  prediction_type = "Probability")

auc_cat_tr <- get_auc(train$claim_status, p_cat_tr)
auc_cat_te <- get_auc(test$claim_status,  p_cat_te)
cat("\nCatBoost AUC - train:", signif(auc_cat_tr, 4),
    "test:", signif(auc_cat_te, 4), "\n")

lt_cat <- lift_table(test$claim_status, p_cat_te)
cat("\nCatBoost lift table (test):\n"); print(lt_cat)

p_cat_cal <- ggplot(lt_cat, aes(avg_pred, obs_rate)) +
  geom_abline(linetype = "dashed") +
  geom_point(size = 2.5, colour = "navy") +
  scale_x_continuous(labels = percent) + scale_y_continuous(labels = percent) +
  labs(title = "CatBoost: Calibration (test deciles)",
       x = "Mean predicted claim probability", y = "Observed claim rate") +
  theme_minimal()
save_plot(p_cat_cal, "cat_calibration")

p_cat_lift <- ggplot(lt_cat, aes(factor(decile), lift)) +
  geom_col(fill = "steelblue") +
  geom_hline(yintercept = 1, linetype = "dashed") +
  labs(title = "CatBoost: Lift by risk decile (test)",
       x = "Decile (1 = highest predicted risk)", y = "Lift vs portfolio") +
  theme_minimal()
save_plot(p_cat_lift, "cat_lift")

p_cat_gain <- ggplot(lt_cat, aes(cum_policies, cum_capture)) +
  geom_line(colour = "navy", linewidth = 1) + geom_point(colour = "navy") +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(labels = percent) + scale_y_continuous(labels = percent) +
  labs(title = "CatBoost: Cumulative gains (test)",
       x = "Cumulative % of policies (riskiest first)",
       y = "Cumulative % of claims captured") +
  theme_minimal()
save_plot(p_cat_gain, "cat_gains")

cat("\n================ FINAL SUMMARY ================\n")
print(fit_log %>% arrange(desc(AUC_test)))
cat("\nElastic net GLM test AUC:", signif(auc_glmnet_te, 4), "\n")
cat("XGBoost test AUC:", signif(auc_xgb_te, 4), "\n")
cat("CatBoost test AUC:", signif(auc_cat_te, 4), "\n")
