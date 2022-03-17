# Do bootstrap estimates for two time periods,
# 1979-2018 and 1980-2019

# I think the boot library is available in the standard R installation,
# so there is no need to install it separately
library(boot)

# the confidence level of the required interval
conf = .9

# The number of bootstrap replicates/samples
# when testing, use something like this
# R = 5000
# when doing the final results, use something like this
R = 150000

arc = read.csv("arctic_temps_obs.csv", header = TRUE)
ref = read.csv("reference_temps_obs.csv", header = TRUE)

# A sanity check
if(! all(arc$Year == ref$Year)){
  cat("WARNING! Something wrong with the years in the observations??\n")
}

# so many cases...
for (mycase in 3) {
  
  if(mycase == 1){
    start.year = 1979
    end.year = 2018
  }
  if(mycase == 2){
    start.year = 1980
    end.year = 2019
  }
  if(mycase == 3){
    start.year = 1979
    end.year = 2021
  }

  # the first column is the year, other columns are the data for each source
  for (k in 2:ncol(arc)) {
    cat(mycase, names(arc)[k], "\n")

    i = ref$Year >= start.year & ref$Year <= end.year
    
    da = data.frame(x = ref$Year[i],
                    y.arc = arc[i, k],
                    y.ref = ref[i, k])
    
    b = boot(da, function(a, i) {
      # here the magic happens:
      # i is the vector of indices to our data which define the bootstrap sample
      # then we fit the linear regression for both arc and ref, and
      # save the ratio of slope coefficients
      coefficients(summary(lm(
        y ~ x , data.frame(x = a$x[i], y = a$y.arc[i])
      )))[2, 1] /
        coefficients(summary(lm(
          y ~ x , data.frame(x = a$x[i], y = a$y.ref[i])
        )))[2, 1]
    }, R = R, parallel = "multicore", ncpus = 4)
    
    bci = boot.ci(b, conf = conf, type = c("perc", "bca"))
    
    r = c(names(arc)[k], b$t0, bci$percent[4:5], bci$bca[4:5])
    if (k == 2) {
      res = r
    } else{
      res = rbind(res, r)
    }
    
  }
  
  # it is easier to manipulate column names etc. using the data frame class
  res = as.data.frame(res)
  
  colnames(res) = c(
    "data",
    "ratio",
    "CIlowerPercentile",
    "CIupperPercentile",
    "CIlowerBCa",
    "CIupperBCa"
  )
  
  write.csv(
    res,
    sprintf("bootstrapCI_temps_obs_%d%d.csv", start.year, end.year),
    row.names = FALSE,
    quote = FALSE
  )
}