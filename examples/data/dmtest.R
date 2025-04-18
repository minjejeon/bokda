## setwd("~/doc/Projects/bokpython/2 Plot")
setwd(dirname(parent.frame(2)$ofile))
library(eztools)
library(inbrowser2)
library(forecast)
rm(list=ls())
source('~/doc/R/common.R')
source('~/doc/R/hangul.R')

options(warn = 1, intercept = '절편')

DH <- decomposeHangul
AFL <- appendFormulaList

setCairo()
setPar <- function(mar=c(3,3,.5,.5)+.1, mgp=c(1.75,.75,0), ...) par(mar=mar, mgp=mgp, ...)

if (! ".e" %in% ls(all=TRUE)) .e <- new.counter()
.e$resetcounters()

ib <- new.inbrowser(title = "::Work")
ib$setdefpdfdim(8, 3.5)

z <- openxlsx::read.xlsx('dmtest.xlsx')
z$X1 <- NULL
names(z)[1] <- 'y'
ff <- z

for (v in names(ff)) ff[[v]] <- ff[[v]]-z$y
## ff$y <- NULL

dm.test(ff$f1, ff$f2, alternative='less') |> print()
dm.test(ff$f2, ff$f1, alternative='less') |> print()

## ib$view()
## ib$export("Exported", force = TRUE)
