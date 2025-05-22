### Neutral interest rate TVP-VAR (see Lubik and Mathes 2015)

#-----------------------------------------------------------------------------------------
# load libraries
#-----------------------------------------------------------------------------------------
# 작업 디렉토리 설정
setwd("C:/Users/bok/Desktop/work/r/nir/LM2015")

# 라이브러리 로드
library(readxl)
library(dplyr)
library(tidyverse)
library(lubridate)
library(zoo)
library(bvarsv)

# .RData 파일 로드
load("NRAUS_DATA 02052020 _ WITH INFE.RData")

# KOR
file_path <- "nir_data_all.xlsx"
df <- read_excel(file_path, sheet = "raw_data")
df <- df %>%
  slice(-n())

# 날짜 열이름 및 날짜 형식 변경
colnames(df)[1] <- 'Date'
colnames(df)[2] <- 'rGDP'
colnames(df)[3] <- 'inf'
colnames(df)[6] <- 'realcall'

KORmacro <- df %>%
  select(rGDP, inf, realcall, Date) %>%
  filter(!is.na(rGDP)) %>%
  filter(!is.na(inf)) %>%
  filter(!is.na(realcall))

KORmacro$Date <- seq(from = as.Date("1991-03-01"), to = as.Date("2024-09-01"), by = "quarter")

KORmacro$rGDP <- as.numeric(KORmacro$rGDP)
KORmacro$inf <- as.numeric(KORmacro$inf)
KORmacro$realcall <- as.numeric(KORmacro$realcall)
KORmacro <- as.data.frame(KORmacro)

KORmacro <- KORmacro %>%
    arrange(Date) %>%
    mutate(
        rGDPg = (rGDP / lag(rGDP, 4) - 1) * 100,
        inf = (inf / lag(inf, 4) - 1) * 100,
    ) %>%
    select(-rGDP) %>%
    relocate(rGDPg, .before =everything()) %>%
    na.omit()

#KORmacro$realcall <- (1 + KORmacro$realcall) / (1 + KORmacro$inf) - 1
KORmacro$realcall <- KORmacro$realcall - KORmacro$inf

#-----------------------------------------------------------------------------------------
# Estimate model (model we like, lags =2, no une)
#----------------------------------------------------------------------------------------

train <- 40 #round(0.25*dim(AUSmacro[,-4])[1])
lags <- 2 

bvar.fit <- bvar.sv.tvp(ts(KORmacro[,-c(4)], start = c(1992, 1), end = c(2024, 3), frequency = 4), p = lags, nf = 5, nburn = 1000, nrep = 20000, pQ = 50, k_Q = 0.005, k_W = 0.01, k_S = 0.01)

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
  mutate(Date = seq(as.Date(KORmacro[train-3+lags*3,"Date"][[1]]), length.out = length(ssvals$ISTAR), by = "quarter")) %>% 
  right_join(KORmacro) %>%
  filter(!is.na(RSTAR))


#-----------------------------------------------------------------------------------------
# Rolling 3 year forecast
#----------------------------------------------------------------------------------------

nrealrate.fcst <- list()

KORmacro.grth <-  KORmacro

start <- train - 3 + lags*3 


for(i in start:(nrow(KORmacro.grth))){
  
  counta <- i-(start-1)
  
  ii <- i-1
  
  KORmacro.grth.subset <- KORmacro.grth[c(1:i),] 
  
  startm <- nrow(KORmacro.grth.subset)-1
  
  endm <- nrow(KORmacro.grth.subset)
  
  fcast <- KORmacro.grth.subset[(endm-1):endm,-4]
  
  fcast <- fcast %>% 
    mutate(rGDPg_L = rGDPg,
           rGDPg = rGDPg,
           rGDPg_U = rGDPg,
           inf_L = inf,
           inf = inf,
           inf_U = inf,
           realcall_L = realcall,
           realcall = realcall,
           realcall_U = realcall
    )
  
  fcst.out <- 12 # 40
  
  
  b <- 0
  
  while(b <= fcst.out){
    
    
    xvec.fcst.m <- xts::last(fcast,n=2) %>%
      mutate(Lag = 2:1) %>% 
      gather(Var, Value,-Lag) %>% 
      filter(!grepl("_U|_L",.$Var)) %>% 
      arrange(Lag)
    
    varfcst <- bvar.fit[["Beta.postmean"]][,,counta][,1]+(bvar.fit[["Beta.postmean"]][,,counta][,-1]%*%xvec.fcst.m$Value)
    
    varfcst.l <- varfcst-1.96*(((diag(bvar.fit[["H.postmean"]][,,counta]))^0.5)*(1+1/b)^0.5)
    
    varfcst.u <- varfcst+1.96*(((diag(bvar.fit[["H.postmean"]][,,counta]))^0.5)*(1+1/b)^0.5)
    
    fcst.table.bvar <- tibble(rGDPg_L = varfcst.l[1],
                              rGDPg = varfcst[1],
                              rGDPg_U = varfcst.u[1],
                              inf_L = varfcst.l[2],
                              inf = varfcst[2],
                              inf_U = varfcst.u[2],
                              realcall_L = varfcst.l[3],
                              realcall = varfcst[3],
                              realcall_U = varfcst.u[3]
    )
    
    fcast <- bind_rows(fcast,fcst.table.bvar)
    
    b <-  b + 1
  } 
  
  
  nrealrate.fcst[[counta]] <- fcast %>% 
    mutate(Forecast_start_date = (KORmacro.grth.subset[i,]$Date),
           H = 1:length(realcall))
  
}  


nrealratebvar.fcst <- bind_rows(nrealrate.fcst)

nrealratebvar.fcst <- nrealratebvar.fcst %>% 
  filter(H == 12) %>% 
  mutate(Date = seq(ymd("2002-09-01"), ymd("2024-09-01"), by ="quarter"))

#-----------------------------------------------------------------------------------------
# Final outputs
#----------------------------------------------------------------------------------------

finaltable <- nrealratebvar.fcst %>% 
  rename(RSTAR3y = realcall,
         ISTAR3y = inf,
         YSTAR3y = rGDPg)

#-----------------------------------------------------------------------------------------
chartRstar <- finaltable %>% 
  ggplot(aes(Date))+
  geom_line(aes(y = RSTAR3y))+
  geom_ribbon(aes(ymin = realcall_L, ymax = realcall_U), alpha = 0.2)+
  ylab("")+
  xlab("")+
  ggtitle("The neutral interest rate") +
  tst_theme() +
  ylim(-3, 4)
  
print(chartRstar)


## Save
finaltable %>%
  select(Date, RSTAR3y) %>%
  write.csv("Rstar_LM.csv", row.names = FALSE)