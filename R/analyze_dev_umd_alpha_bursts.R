Sys.setenv(OMP_NUM_THREADS = "60", OPENBLAS_NUM_THREADS = "60", MKL_NUM_THREADS = "60")

library('lme4')
library('car')
library('emmeans')
library('dplyr')

# parameters
ages <- c("9m","12m","adult")
chs <- c('E16', 'E20', 'E21', 'E22','E41', 'E49', 'E50', 'E51')

dir_path <- "../output/dev_beta_umd/alpha_bursts"

# Redirect output to file
sink("../output/dev_umd_alpha_burst_results.txt")

for(age in ages) {
      print(paste0('=========================== ', age, ' ==========================='))
      # get matching files
      files <- list.files(dir_path, pattern = paste0("infant-alpha_", age, "_sub-\\d+\\.csv$"), full.names = TRUE)

      # read and concatenate
      df <- files %>%
        lapply(read.csv) %>%
        bind_rows(.id = "file_id") %>%
        filter(sensor_name %in% chs)

      df$condition<-''
      df$condition[df$trial_label=='OBM']<-'obs'
      df$condition[df$trial_label=='LOBS']<-'obs'
      df$condition[df$trial_label=='FTGO']<-'obs'
      df$condition[df$trial_label=='OBGC']<-'obs'
      df$condition[df$trial_label=='OBEND']<-'obs'
      df$condition[df$trial_label=='EBM']<-'exe'
      df$condition[df$trial_label=='LEXT']<-'exe'
      df$condition[df$trial_label=='FTGE']<-'exe'
      df$condition[df$trial_label=='EXGC']<-'exe'
      df$condition[df$trial_label=='EXEND']<-'exe'

      df$epoch<-''
      df$epoch[df$trial_label=='OBM']<-'baseline'
      df$epoch[df$trial_label=='LOBS']<-'go'
      df$epoch[df$trial_label=='FTGO']<-'touch'
      df$epoch[df$trial_label=='OBGC']<-'grasp'
      df$epoch[df$trial_label=='OBEND']<-'end'
      df$epoch[df$trial_label=='EBM']<-'baseline'
      df$epoch[df$trial_label=='LEXT']<-'go'
      df$epoch[df$trial_label=='FTGE']<-'touch'
      df$epoch[df$trial_label=='EXGC']<-'grasp'
      df$epoch[df$trial_label=='EXEND']<-'end'

      df$condition<-as.factor(df$condition)
      df$epoch<-as.factor(df$epoch)

      contrasts(df$condition) <- contr.sum
      contrasts(df$epoch) <- contr.sum

      m <- lmer(burst_cycles ~ condition * epoch + (1+condition+epoch | subject), data = df)
      print(Anova(m, type = 3))
      print(emmeans(m, pairwise ~ condition|epoch, type = 'response'))
}

# Stop redirecting output
sink()