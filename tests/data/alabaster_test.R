library(alabaster.sfe)
library(Voyager)
library(SFEData)
library(SingleCellExperiment)
library(scater)
library(sf)
library(fs)
library(spdep)
set.SubgraphOption(FALSE)

fp <- "/Users/kancherj/Projects/artifactdb/dolomite-sfe/tests/data"
x2 <- XeniumOutput("v2", file_path = file.path(fp, "xenium2"))
sfe <- readXenium(x2, add_molecules = TRUE)
colData(sfe)

colGraph(sfe, "knn5") <- findSpatialNeighbors(sfe, method = "knearneigh", k = 5)
sfe <- logNormCounts(sfe, size_factors = sfe$cell_area)
sfe <- runMoransI(sfe, colGraphName = "knn5")
sfe <- colDataMoransI(sfe, features = c("transcript_counts", "cell_area"))
top_moran <- rownames(sfe)[which.max(rowData(sfe)$moran_sample01)]
sfe <- runUnivariate(sfe, type = "localmoran", features = top_moran)

rowData(sfe)
plotSpatialFeature(sfe, features = top_moran, colGeometryName = "cellSeg")
colFeatureData(sfe)
plotLocalResult(sfe, name = "localmoran", features = top_moran, divergent = TRUE,
                colGeometryName = "cellSeg", diverge_center = 0)
plotImage(sfe, image_id = "morphology_focus", channel = 3:1, normalize_channels = TRUE)

sfe <- SpatialFeatureExperiment::rotate(sfe, degrees = 30)
plotGeometry(sfe, colGeometryName = "nucSeg", fill = FALSE, 
             image_id = "morphology_focus",
             channel = 3:1, normalize_channels = TRUE, dark = TRUE)

fsave <- file.path(fp, "sfe_save")
saveObject(sfe, fsave)
dir_tree(fsave)



