library('lme4')
library('car')
library('emmeans')
df<-read.csv('LC-area-intercept.csv')
df$Hemisphere<-as.factor(df$Hemisphere)
df$Reach.hand<-as.factor(df$Reach.hand)
df$Monkey<-as.factor(df$Monkey)
df$Session<-as.factor(df$Session)
df$Response.time<-log10(df$Response.time)

# Both monkeys - intercept
m<-lmer(Response.time ~ Hemisphere*Intercept + (1 | Monkey/Session), data=df, control = lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e6)))
Anova(m)
summary(m)
# Both monkeys - area
m<-lmer(Response.time ~ Hemisphere*Area + (1 | Monkey/Session), data=df, control = lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e6)))
Anova(m)
emtrends(m, pairwise ~ Hemisphere, var="Area", infer=TRUE)

# OB - intercept
m<-lmer(Response.time ~ Hemisphere*Intercept + (1 | Session), data=df[df$Monkey=='OB',], control = lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e6)))
Anova(m)

# OB - area
m<-lmer(Response.time ~ Hemisphere*Area + (1 | Session), data=df[df$Monkey=='OB',], control = lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e6)))
Anova(m)
emtrends(m, pairwise ~ Hemisphere, var="Area", infer=TRUE)

# W - intercept
m<-lmer(Response.time ~ Hemisphere*Intercept + (1 | Session), data=df[df$Monkey=='W',], control = lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e6)))
Anova(m)

# W - area
m<-lmer(Response.time ~ Hemisphere*Area + (1 | Session), data=df[df$Monkey=='W',], control = lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e6)))
Anova(m)
emtrends(m, pairwise ~ Hemisphere, var="Area", infer=TRUE)
