from lib.histogram_grid import HistogramGrid

map_fname = "map.txt"
resolution = 1  # node size = 1cm
hg = HistogramGrid.build_histogram_from_txt(map_fname, resolution)
print(*hg.histogram_grid, sep="\n")
