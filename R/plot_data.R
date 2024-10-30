# Install and load necessary libraries
#install.packages("lattice")
library(lattice)
require(tigerstats)

# Create sample data
set.seed(123)
xs <- data.frame(
  xic.ai = rnorm(100),  # Random data for xic.ai
  xic.rt = seq(0, 100, length.out = 100),  # Sequential data for xic.rt
  Tag = sample(c("A", "B"), 100, replace = TRUE)  # Random tags "A" and "B"
)

# Define mu and rts for the plot
mu <- mean(xs$xic.ai)
rts <- c(0, 100)
run <- "Sample Run"

# Run the xyplot
xyplot(xic.ai~xic.rt|Tag, data=xs, type=c('a'),
       panel = function(x,y, subscripts, ...) {
         panel.xyplot(x,y, type="a")
         panel.abline(h=mu, col=c('red'), lty="dashed")
       }, 
       xlab=paste0(run," - RT"),
       scales = list(x=list(relation="free"), y=list(relation="free"))
)
