rnot_homoph <- function(delta, mu = 0.1, s1_init = 0.3, 
                        h1 = 0.5, h2 = 0.5, p1 = 10.0, p2 = 0.05) {
  
  f11 <- (1 + h1) / 2.0
  f12 <- (1 - h1) / 2.0
  f21 <- (1 + h2) / 2.0
  f22 <- (1 - h2) / 2.0
  s2_init <- 1.0 - s1_init
  
  # Saturating function of virulence.
  tau <- (p1 * delta) / (p2 + delta)
  
  # Construct next-generation matrix.
  G <- matrix(
    data = c( (f11*tau*s1_init) - mu - delta, 
              (f21*tau*s2_init), 
              (f12*tau*s1_init), 
              (f22*tau*s2_init) - mu - delta),
    nrow = 2, ncol = 2
  )

  return (eigen(G)$values[1])
}