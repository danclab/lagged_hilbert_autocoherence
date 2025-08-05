library('lme4')
library('car')
library('emmeans')

# Redirect output to file
sink("../output/explicit-implicit_lmm_results.txt")

df <- read.csv('../output/explicit-implicit.csv')
df$subject <- as.factor(df$subject)
df$group <- as.factor(df$group)
df$block_type <- ''
df$block_type[df$block == 2] <- 'baseline'
df$block_type[df$block > 2 & df$block < 9] <- 'rotation'
df$block_type[df$block == 9] <- 'washout'
df$block_type <- as.factor(df$block_type)

for (algo in c('lhc', 'lfc')) {
  for (epo in c('visual', 'motor')) {
    for (band in c('alpha', 'beta')) {
      cat("\n\n===========================\n")
      cat(paste0("Model: ", algo, " | Epoch: ", epo, " | Band: ", band, "\n"))
      cat("===========================\n\n")
      
      sub_df <- df[df$epoch == epo & df$band == band & df$algorithm == algo, ]
      
      contrasts(sub_df$group) <- contr.sum
      contrasts(sub_df$block_type) <- contr.sum
      
      # Crossover point (x0) - Block type model
      cat("Model: x0 ~ group * block_type\n")
      m1 <- lmer(x0 ~ group * block_type + (1 | subject), data = sub_df)
      print(Anova(m1, type = 3))
      print(emmeans(m1, pairwise ~ block_type, type = 'response'))
      print(emmeans(m1, pairwise ~ group | block_type, type = 'response'))
      print(emmeans(m1, pairwise ~ block_type | group, type = 'response'))
      
      # Crossover point (x0) - Linear trend model in rotation blocks
      cat("\nModel: x0 ~ group * block (rotation blocks only)\n")
      m2 <- lmer(x0 ~ group * block + (1 | subject), data = sub_df[sub_df$block_type == 'rotation', ])
      print(Anova(m2, type = 3))
      print(emtrends(m2, pairwise ~ group, var = 'block', type = 'response', infer = TRUE))
      
      # Decay rate (k) - Block type model
      cat("\nModel: k ~ group * block_type\n")
      m3 <- lmer(k ~ group * block_type + (1 | subject), data = sub_df)
      print(Anova(m3, type = 3))
      print(emmeans(m1, pairwise ~ block_type, type = 'response'))
      print(emmeans(m3, pairwise ~ group | block_type, type = 'response'))
      print(emmeans(m3, pairwise ~ block_type | group, type = 'response'))
      
      # Decay rate (k) - Linear trend model in rotation blocks
      cat("\nModel: k ~ group * block (rotation blocks only)\n")
      m4 <- lmer(k ~ group * block + (1 | subject), data = sub_df[sub_df$block_type == 'rotation', ])
      print(Anova(m4, type = 3))
      print(emtrends(m4, pairwise ~ group, var = 'block', type = 'response', infer = TRUE))
    }
  }
}

# Stop redirecting output
sink()
