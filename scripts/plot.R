require(ggplot2)
require(latex2exp)
require(dplyr)
require(readr)
require(purrr)
require(reshape2)
require(stringr)

mytheme <- theme(axis.line = element_line(), legend.key=element_rect(fill = NA),
                text = element_text(size=22), 
                panel.background = element_rect(fill = "white"))


plot_series <- function(csv_file, write_path="tmp/output.pdf") {

  df <- read.csv(csv_file)
  melted <- melt(df, id.vars = "step")

  print(head(melted))

  ggplot(melted, aes(x=step, y=value, col=variable)) + geom_line() + mytheme

  ggsave(write_path, width=6.75, height=5)
}
