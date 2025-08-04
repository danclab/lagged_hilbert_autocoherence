library('lme4')
library('car')
library('emmeans')
library('ggplot2')

df<-read.csv('../output/sim_oscillations.csv')
df$algorithm<-as.factor(df$algorithm)

snrs<-unique(df$snr)
for(snr in snrs) {
  print(paste0('SNR=',snr,' dB'))
  sub_df<-df[df$snr==snr,]
  contrasts(sub_df$algorithm) <- contr.sum
  
  m1<-lm(rmse ~ frequency*algorithm, data=sub_df)
  print(Anova(m1, type = 3))
  print(emmeans(m1,pairwise~algorithm|frequency, type='response'))
  print(emtrends(m1, pairwise ~ algorithm, var='frequency',type='response',infer=TRUE))
  
  m2<-lm(std ~ frequency*algorithm, data=sub_df)
  print(Anova(m2, type = 3))
  print(emmeans(m2,pairwise~algorithm, type='response'))
  print(emtrends(m2, pairwise ~ algorithm, var='frequency',type='response',infer=TRUE))
}


df<-read.csv('../output/sim_burst_duration.csv')
snrs<-unique(df$snr)
for(snr in snrs) {
  print(paste0('SNR=',snr,' dB'))
  sub_df<-df[df$snr==snr,]

  # Duration (s)
  m1<-lm(x0 ~ frequency*burst_d, data=sub_df)
  print(Anova(m1, type = 3))

  # Duration (cycles)
  m2<-lm(x0 ~ frequency*burst_d_c, data=sub_df)
  print(Anova(m2, type = 3))

  aic1 <- AIC(m1)
  aic2 <- AIC(m2)
  print(aic2-aic1)

  m1<-lm(k ~ frequency*burst_d, data=sub_df)
  print(Anova(m1, type = 3))

  m2<-lm(k ~ frequency*burst_d_c, data=sub_df)
  print(Anova(m2, type = 3))

  aic1 <- AIC(m1)
  aic2 <- AIC(m2)
  print(aic1-aic2)
}

df<-read.csv('../output/sim_burst_number.csv')
snrs<-unique(df$snr)
for(snr in snrs) {
  print(paste0('SNR=',snr,' dB'))
  sub_df<-df[df$snr==snr,]

  m<-lm(x0 ~ frequency*burst_n, data=sub_df)
  print(Anova(m, type = 3))
  
  m<-lm(k ~ frequency*burst_n, data=sub_df)
  print(Anova(m, type = 3))
}
