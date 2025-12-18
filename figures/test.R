# Install if not already installed
if (!requireNamespace("circlize", quietly = TRUE)) install.packages("circlize")
library(circlize)

# Create output folder
if (!dir.exists("figures/plots")) dir.create("figures/plots", recursive = TRUE)

# Example patient names
patients <- paste0("Patient_", 1:10)

# Example group assignment
group <- rep(c("Subtype_A", "Subtype_B"), each = 5)

# Example multi-layer bar values
score1 <- runif(10, 0, 1)   # first bar layer
score2 <- runif(10, 0, 1)   # second bar layer

# Export circular plot to PNG
png("figures/plots/circular_test.png", width = 800, height = 800, res = 150)

# Clear previous plot
circos.clear()

# Initialize sectors
circos.initialize(factors = patients, xlim = c(0,1))

# First bar track
circos.trackPlotRegion(
  ylim = c(0, 1), 
  track.height = 0.15,
  bg.border = NA,
  panel.fun = function(x, y) {
    patient_id <- CELL_META$sector.index
    circos.rect(
      xleft = 0, ybottom = 0,
      xright = score1[which(patients == patient_id)],
      ytop = 1,
      col = ifelse(group[which(patients == patient_id)]=="Subtype_A", "#1f78b4", "#33a02c"),
      border = NA
    )
  }
)

# Second bar track
circos.trackPlotRegion(
  ylim = c(0, 1),
  track.height = 0.15,
  bg.border = NA,
  panel.fun = function(x, y) {
    patient_id <- CELL_META$sector.index
    circos.rect(
      0, 0,
      score2[which(patients == patient_id)],
      1,
      col = "#e31a1c",
      border = NA
    )
  }
)

# Add outer labels
circos.trackPlotRegion(
  ylim = c(0, 1),
  track.height = 0.05,
  bg.border = NA,
  panel.fun = function(x, y) {
    circos.text(
      x = 0.5, y = 0.5,
      labels = CELL_META$sector.index,
      facing = "clockwise",
      niceFacing = TRUE,
      adj = c(0, 0.5),
      cex = 0.6
    )
  }
)

# Clear after plotting
circos.clear()

# Close PNG device
dev.off()
