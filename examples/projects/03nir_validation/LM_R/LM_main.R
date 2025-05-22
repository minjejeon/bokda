### Neutral interest rate TVP-VAR (see Lubik and Mathes 2015)

#-----------------------------------------------------------------------------------------
# load libraries
#-----------------------------------------------------------------------------------------
setwd("C:/Users/bok/Desktop/work/r/nir")
library(dplyr)
library(tidyverse)
library(lubridate)
#source("~/GitHub/TST themes/Chart themes.R")
library(bvarsv)
#devtools::load_all(path = "C:/Users/aelde/OneDrive/Documents/GitHub/Packages/tst.package")

# good
AUSmacro <- m %>%
 mutate(Inflation = CPI/lag(CPI,4)*100-100,
        RealRi90 = rate_90GST_r,
        RGDP = dl_GDPNF_r, # *400
        UNER = rate_UNE,
        TWI = rate_TW,

 ) %>%
 select(RGDP, Inflation, RealRi90, Date) %>%
 filter(!is.na(RealRi90))

# test
# AUSmacro <- m %>%
#   mutate(Inflation = d4l_PGDPE*100,
#          RealRi90 = rate_90,
#          RGDP = d4l_GDPNF_r*100,
#          CPIINF = d4l_CPIU_GST*100 ,
# 
# 
#   ) %>%
#   select(RGDP, Inflation, RealRi90, CPIINF, Date) %>%
#   filter(!is.na(RealRi90))

#-----------------------------------------------------------------------------------------
# Estimate model (model we like, lags =2, no une)
#----------------------------------------------------------------------------------------

train <- 40 #round(0.25*dim(AUSmacro[,-4])[1])# 40
lags <- 2

bvar.fit <- bvar.sv.tvp(ts(AUSmacro[,-c(4:5)],c(1990,3), f = 4), p =lags, nf = 5, nburn = 1000, nrep = 20000, pQ = 50, k_Q = 0.005, k_W = 0.01, k_S = 0.01 )

#var.fit <- vars::VAR(ts(AUSmacro[1:train,-4],c(1990,3), f = 4), p = 2, type = "const")


#Get posterior draws - if we want standard errors around estimate like in BIS paper 
intbeta <- bvar.fit$Beta.draws[1:3,1:dim(bvar.fit$Beta.postmean)[3], ]
intbeta <- apply(intbeta, c(1,2), function(z) c(mean(z), quantile(z, c(0.16, 0.84)))[c(2, 1, 3)])

beta1 <- bvar.fit$Beta.draws[4:12,1:dim(bvar.fit$Beta.postmean)[3], ]
beta1 <- apply(beta1, c(1:2), function(z) c(mean(z), quantile(z, c(0.16, 0.84)))[c(2, 1, 3)])

beta2 <- bvar.fit$Beta.draws[13:21,1:dim(bvar.fit$Beta.postmean)[3], ]
beta2 <- apply(beta2, c(1:2), function(z) c(mean(z), quantile(z, c(0.16, 0.84)))[c(2, 1, 3)])


# Create arrays of coefficients for upper and lower percentile 
Beta.postupper <- array(0,dim = c(3,7,dim(bvar.fit$Beta.postmean)[3]))
Beta.postlower <- array(0,dim = c(3,7,dim(bvar.fit$Beta.postmean)[3]))

for(i in 1:dim(bvar.fit$Beta.postmean)[3]){
  
  Beta.postupper[,,i] <- matrix(c(matrix(intbeta[3,,i],3,1, byrow = T),matrix(beta1[3,,i],3,3, byrow = T),matrix(beta2[3,,i],3,3, byrow = T)), 3, 7, byrow = F) 
  
  Beta.postlower[,,i] <- matrix(c(matrix(intbeta[1,,i],3,1, byrow = T),matrix(beta1[1,,i],3,3, byrow = T),matrix(beta2[1,,i],3,3, byrow = T)), 3, 7, byrow = F) 
  
  
}

# an empty list to put steady state values in
ssvals <-  list(ISTAR = NA, RSTAR = NA, YSTAR = NA)


# compute steady state values
for(i in 1:dim(bvar.fit$Beta.postmean)[3]){
  
  ss_m <- solve((diag(3)-bvar.fit$Beta.postmean[,2:4,i]-bvar.fit$Beta.postmean[,5:7,i]),bvar.fit$Beta.postmean[,1,i])
  
  ssvals[["YSTAR"]][[i]] <- ss_m[1]
  ssvals[["ISTAR"]][[i]] <- ss_m[2]
  ssvals[["RSTAR"]][[i]] <- ss_m[3]
  
}

# create a table of ss values and actual values
ssvals <- bind_rows(ssvals) %>% 
  mutate(Date = seq(as.Date(AUSmacro[train-3+lags*3,"Date"][[1]]), length.out = length(ssvals$ISTAR), by = "quarter")) %>% 
  right_join(AUSmacro) %>% 
  filter(!is.na(RSTAR))

#-----------------------------------------------------------------------------------------
# Steady state charts
#----------------------------------------------------------------------------------------

# neutral rate chart
labelRSTAR <-paste0("estimate at ", paste0(last(ssvals$Date)," = "), round(last(ssvals$RSTAR),3)) 

ssvals %>% 
  ggplot()+
  geom_line(aes(Date, RealRi90))+
  geom_line(aes(Date, RSTAR), col = tst_colors[2])+
  tst_theme()+
  ggtitle("The neutral interest rate")+
  xlab("")+
  ylab("")+
  annotate("text", x=ymd("2017-03-01") , y= 1.5, label = "Neutral rate"  , colour =tst_colors[2])+
  annotate("text", x=ymd("2017-03-01") , y= 1.25, label = labelRSTAR  , colour =tst_colors[2])+
  annotate("text", x=ymd("2005-06-01") , y= 1, label = "Real 90 day BB rate", colour =tst_colors[1])

# inflation chart
ssvals %>% 
  ggplot()+
  geom_line(aes(Date, Inflation))+
  geom_line(aes(Date, ISTAR), col = tst_colors[2])+
  tst_theme()+
  ggtitle("Steady state inflation")+
  xlab("")+
  ylab("")+
  annotate("text", x=ymd("2018-06-01") , y= 1, label = "Steady state inflation", colour =tst_colors[2])+
  annotate("text", x=ymd("2001-06-01") , y= 1, label = "Inflation", colour =tst_colors[1])


# potential growth chart
ssvals %>% 
  ggplot()+
  geom_line(aes(Date, RGDP))+
  #geom_line(aes(Date, YSTAR), col = tst_colors[2])+tst_theme()+
  ggtitle("Potential output growth")+
  xlab("")+
  ylab("")# +
# annotate("text", x=ymd("2018-06-01") , y= 1, label = "Potential growth", colour =tst_colors[2])+
# annotate("text", x=ymd("2001-06-01") , y= 1, label = "Real GDP", colour =tst_colors[1])



#-----------------------------------------------------------------------------------------
# Alternative method (Rolling 5 year forecast)
#----------------------------------------------------------------------------------------

nrealrate.fcst <- list()

AUSmacro.grth <-  AUSmacro # AUSmacro[,-4]

start <- train-3+lags*3 


for(i in start:(nrow(AUSmacro.grth))){
  
  counta <- i-(start-1)
  
  ii <- i-1
  
  AUSmacro.grth.subset <- AUSmacro.grth[c(1:i),] 
  
  startm <- nrow(AUSmacro.grth.subset)-1
  
  endm <- nrow(AUSmacro.grth.subset)
  
  fcast <- AUSmacro.grth.subset[(endm-1):endm,-4]
  
  fcast <- fcast %>% 
    mutate(RGDP_L = RGDP,
           RGDP = RGDP,
           RGDP_U = RGDP,
           Inflation_L = Inflation,
           Inflation = Inflation,
           Inflation_U = Inflation,
           RealRi90_L = RealRi90,
           RealRi90 = RealRi90,
           RealRi90_U = RealRi90
    )
  
  fcst.out <- 40
  
  
  b <- 0
  
  while(b <= fcst.out){
    
    
    xvec.fcst.m <- xts::last(fcast,n=2) %>%
      mutate(Lag = 2:1) %>% 
      gather(Var, Value,-Lag) %>% 
      filter(!grepl("_U|_L",.$Var)) %>% 
      arrange(Lag)
    
    # xvec.fcst.l <- last(fcast,n=2) %>%
    #   mutate(Lag = 2:1) %>% 
    #   gather(Var, Value,-Lag) %>% 
    #   filter(grepl("_L",.$Var)) %>% 
    #   arrange(Lag)
    
    # xvec.fcst.u <- last(fcast,n=2) %>%
    #   mutate(Lag = 2:1) %>% 
    #   gather(Var, Value,-Lag) %>% 
    #   filter(grepl("_U",.$Var)) %>% 
    #   arrange(Lag)
    
    
    varfcst <- bvar.fit[["Beta.postmean"]][,,counta][,1]+(bvar.fit[["Beta.postmean"]][,,counta][,-1]%*%xvec.fcst.m$Value)
    
    varfcst.l <- varfcst-1.96*(((diag(bvar.fit[["H.postmean"]][,,counta]))^0.5)*(1+1/b)^0.5)
    
    varfcst.u <- varfcst+1.96*(((diag(bvar.fit[["H.postmean"]][,,counta]))^0.5)*(1+1/b)^0.5)
    #varfcst.l <- Beta.postlower[,,counta][,1]+(Beta.postlower[,,counta][,-1]%*%xvec.fcst.l$Value)
    
    #varfcst.u <- Beta.postupper[,,counta][,1]+(Beta.postupper[,,counta][,-1]%*%xvec.fcst.u$Value)
    
    fcst.table.bvar <- tibble(RGDP_L = varfcst.l[1],
                              RGDP = varfcst[1],
                              RGDP_U = varfcst.u[1],
                              Inflation_L = varfcst.l[2],
                              Inflation = varfcst[2],
                              Inflation_U = varfcst.u[2],
                              RealRi90_L = varfcst.l[3],
                              RealRi90 = varfcst[3],
                              RealRi90_U = varfcst.u[3]
    )
    
    fcast <- bind_rows(fcast,fcst.table.bvar)
    
    b <-  b + 1
  } 
  
  
  nrealrate.fcst[[counta]] <- fcast %>% 
    mutate(Forecast_start_date = (AUSmacro.grth.subset[i,]$Date),
           H = 1:length(RealRi90))
  
}  


nrealratebvar.fcst <- bind_rows(nrealrate.fcst)

nrealratebvar.fcst <- nrealratebvar.fcst %>% 
  filter(H == 40) %>% 
  mutate(Date = seq(ymd("1979-12-01"), ymd("2019-12-01"), by ="quarter"))


#-----------------------------------------------------------------------------------------
# Final outputs
#----------------------------------------------------------------------------------------


finaltable <- nrealratebvar.fcst %>% 
  rename(RSTAR5y = RealRi90,
         ISTAR5y = Inflation,
         YSTAR5y = RGDP) %>% 
  left_join(ssvals) %>% 
  mutate(ANNUAL_INF = ((1+Inflation/100)*(1+lag(Inflation/100))*(1+lag(Inflation/100,2))*(1+lag(Inflation/100,3))-1)*100,
         ANNUAL_ISTAR = ((1+ISTAR/100)*(1+lag(ISTAR/100))*(1+lag(ISTAR/100,2))*(1+lag(ISTAR/100,3))-1)*100,
         ANNUAL_ISTAR5y = ((1+ISTAR5y/100)*(1+lag(ISTAR5y/100))*(1+lag(ISTAR5y/100,2))*(1+lag(ISTAR5y/100,3))-1)*100,
         ANNUAL_ISTAR5yU = ((1+Inflation_U/100)*(1+lag(Inflation_U/100))*(1+lag(Inflation_U/100,2))*(1+lag(Inflation_U/100,3))-1)*100,
         ANNUAL_ISTAR5yL = ((1+Inflation_L/100)*(1+lag(Inflation_L/100))*(1+lag(Inflation_L/100,2))*(1+lag(Inflation_L/100,3))-1)*100,
         ANNUAL_RGDP = ((1+RGDP/100)*(1+lag(RGDP/100))*(1+lag(RGDP/100,2))*(1+lag(RGDP/100,3))-1)*100,
         ANNUAL_YSTAR = ((1+YSTAR/100)*(1+lag(YSTAR/100))*(1+lag(YSTAR/100,2))*(1+lag(YSTAR/100,3))-1)*100,
         ANNUAL_YSTAR5y = ((1+YSTAR5y/100)*(1+lag(YSTAR5y/100))*(1+lag(YSTAR5y/100,2))*(1+lag(YSTAR5y/100,3))-1)*100,
         ANNUAL_YSTAR5yU = ((1+RGDP_U/100)*(1+lag(RGDP_U/100))*(1+lag(RGDP_U/100,2))*(1+lag(RGDP_U/100,3))-1)*100,
         ANNUAL_YSTAR5yL = ((1+RGDP_L/100)*(1+lag(RGDP_L/100))*(1+lag(RGDP_L/100,2))*(1+lag(RGDP_L/100,3))-1)*100)


chartRstar <- finaltable %>% 
  ggplot(aes(Date))+
  geom_line(aes(y = RSTAR5y))+
  geom_ribbon(aes(ymin = RealRi90_L, ymax = RealRi90_U), alpha = 0.2)+
  ylab("")+
  xlab("")+
  ggtitle("The neutral interest rate")
  tst_theme()
  
print(chartRstar)

chartYstar <- finaltable %>% 
  ggplot(aes(Date)) +
  #geom_line(aes(y = ANNUAL_YSTAR5y), colour = tst_colors[2])+
  #geom_ribbon(aes(ymin = ANNUAL_YSTAR5yL, ymax =ANNUAL_YSTAR5yU), alpha = 0.2)+
  geom_line(aes(y = YSTAR5y), colour = tst_colors[2])+
  geom_ribbon(aes(ymin = RGDP_L, ymax =RGDP_U), alpha = 0.2)+
  
  ylab("")+
  xlab("")+
  ggtitle("Potential output")+
  tst_theme()

print(chartYstar)

chartIstar <- finaltable %>% 
  ggplot(aes(Date)) +
  #geom_line(aes(y = ANNUAL_ISTAR5y), colour = tst_colors[2])+
  #geom_ribbon(aes(ymin = ANNUAL_ISTAR5yL, ymax = ANNUAL_ISTAR5yU), alpha = 0.2)+
  
  geom_line(aes(y = ISTAR5y), colour = tst_colors[2])+
  geom_ribbon(aes(ymin = Inflation_L, ymax = Inflation_U), alpha = 0.2)+
  ylab("")+
  xlab("")+
  ggtitle("Steady state inflation")+
  tst_theme()

print(chartIstar)

gridExtra::grid.arrange(chartYstar,chartIstar,chartRstar, nrow = 1)
