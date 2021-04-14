# I think the boot library is available in the standard R installation,
# so there is no need to install it separately
library(boot)

# the confidence level of the required interval
conf = .9

# The number of bootstrap replicates/samples
# when testing, use something like this
# R = 500
# when doing the final results, use something like this
R = 50000

arc = read.csv("arctic_temps_obs.csv", header = TRUE)
ref = read.csv("reference_temps_obs.csv", header = TRUE)

# the first column is the year, other columns are the data for each source
for (k in 2:ncol(arc)) {
  
  print(names(arc)[k])
  
  da = data.frame(x = ref$Year,
                  y.arc = arc[, k],
                  y.ref = ref[, k])

  b = boot(da, function(a, i) {
    # here the magic happens: 
    # i is the vector of indices to our data which define the bootstrap sample
    # then we fit the linear regression for both arc and ref, and 
    # save the ratio of slope coefficients
    coefficients(summary(lm(y ~ x , data.frame(
      x = a$x[i], y = a$y.arc[i]
    ))))[2, 1] /
      coefficients(summary(lm(y ~ x , data.frame(
        x = a$x[i], y = a$y.ref[i]
      ))))[2, 1]
  }, R = R)
  
  bci = boot.ci(b, conf = conf, type = c("perc", "bca"))
  
  r = c(names(arc)[k], b$t0, bci$percent[4:5], bci$bca[4:5] )
  if(k==2){
    res = r
  }else{
    res = rbind(res,r)    
  }
  
}

# it is easier to manipulate column names etc. using the data frame class
res = as.data.frame(res)

colnames(res) = c("data","ratio","CIlowerPercentile","CIupperPercentile","CIlowerBCa","CIupperBCa")

write.csv(res,"bootstrapCI_temps_obs.csv", row.names = FALSE, quote = FALSE)
