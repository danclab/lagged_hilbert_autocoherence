library('lme4')
library('car')
library('emmeans')
library('ggplot2')

df<-read.csv('../output/sim_oscillations.csv')
df<-df[df$snr==-5,]
df$algorithm<-as.factor(df$algorithm)
contrasts(df$algorithm) <- contr.sum

m1<-lm(rmse ~ frequency*algorithm, data=df)
Anova(m1, type = 3)
emmeans(m1,pairwise~algorithm|frequency, type='response')
emtrends(m1, pairwise ~ algorithm, var='frequency',type='response',infer=TRUE)

m2<-lm(std ~ frequency*algorithm, data=df)
Anova(m2, type = 3)
emmeans(m2,pairwise~algorithm, type='response')
emtrends(m2, pairwise ~ algorithm, var='frequency',type='response',infer=TRUE)



df<-read.csv('../output/sim_burst_duration.csv')
df<-df[df$snr==-5,]

# Duration (s)
m1<-lm(x0 ~ frequency*burst_d, data=df)
Anova(m1, type = 3)

# Duration (cycles)
m2<-lm(x0 ~ frequency*burst_d_c, data=df)
Anova(m2, type = 3)

aic1 <- AIC(m1)
aic2 <- AIC(m2)
print(aic2-aic1)


m1<-lm(k ~ frequency*burst_d, data=df)
Anova(m1, type = 3)

m2<-lm(k ~ frequency*burst_d_c, data=df)
Anova(m2, type = 3)

aic1 <- AIC(m1)
aic2 <- AIC(m2)
print(aic1-aic2)


df<-read.csv('../output/sim_burst_number.csv')
df<-df[df$snr==-5,]


m<-lm(x0 ~ frequency*burst_n, data=df)
Anova(m, type = 3)

m<-lm(k ~ frequency*burst_n, data=df)
Anova(m, type = 3)
