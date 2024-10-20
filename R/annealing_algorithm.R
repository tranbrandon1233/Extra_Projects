# Set seed for reproducibility
set.seed(123)

# Sample data
npus <- 100  # Number of planning units
nyears <- 5  # Number of years for which we have data

# Generate sample data
coverage <- seq(0.1, 0.5, by = 0.1)  # Coverage levels to test
mean_ex_rank <- sort(runif(npus), decreasing = TRUE)  # Mean export rank for each planning unit

# Generate train_mats (list of matrices, one for each year)
# Each matrix represents the export from each planning unit for a given year
train_mats <- lapply(1:nyears, function(y) {
  matrix(runif(npus, 0, 10), nrow = npus, ncol = 1)
})

# Combine all matrices into one
ex_thru_time <- do.call(cbind, train_mats)

store_maxmin_sites = list()
for (cov in 1:length(coverage)) {
  set = sample(1:npus, round(coverage[cov]*npus), replace = FALSE)  
  unprotected_sites = 1:npus
  
  while (length(set) < round(coverage[cov] * npus)) {
    best_site = NULL
    best_min_export = -Inf 
    
    for (site in unprotected_sites) {
      temp_set = c(set, site)
      ex_thru_time = matrix(data=unlist(lapply(train_mats, rowSums)), nrow=npus, ncol=length(train_mats))
      setex_thru_time = ex_thru_time[temp_set,]
      min_export = min(colSums(setex_thru_time))
      
      if (min_export > best_min_export) {
        best_min_export = min_export
        best_site = site
      }
    }
    
    set = c(set, best_site)
    unprotected_sites = unprotected_sites[unprotected_sites != best_site]
    # Print current minimum export and length of the set for debugging
    print(paste("Min export1" , min_export1))
    print(paste("Set: ",set))
    print(paste("Unprotected sites: ",unprotected_sites))
  }
  
  store_maxmin_sites[[cov]] = set
}

# After the loop, you can inspect the results
print(store_maxmin_sites)