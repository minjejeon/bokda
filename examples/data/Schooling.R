setwd("~/doc/Projects/bokpython/3 Linear Models")
## setwd(dirname(parent.frame(2)$ofile))
library(eztools)
library(inbrowser2)
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

data(Schooling, package='Ecdat')
z <- Schooling

for (v in names(z)) {
  if (inherits(z[[v]],'factor')) {
    if (identical(levels(z[[v]]), c("no", "yes"))) {
      cat(v,'\n')
      z[[v]] <- as.numeric(z[[v]]=='yes')
    }
  }
}

attr(z,'datalabel') <- 'Wages and Schooling, from R Ecdat package'
print(head(z))

foreign::write.dta(z, 'schooling.dta')

## ib$view()
## ib$export("Exported", force = TRUE)
