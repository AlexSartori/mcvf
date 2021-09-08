from mcvf import core, filters


print("Loading sample video...")
v = core.Video('test-video-4.mp4')

print("Filtering...")
# v.apply_filter(filters.MCGaussianFilter(block_size=20))
# v.apply_filter(filters.MCDarkenFilter(block_size=10))
# v.apply_filter(filters.MFDrawerFilter(block_size=20))
v.apply_filter(filters.MCMovingAvergeFilter(block_size=20))

print("Playing...")
v.play()

print("Saving...")
v.save_to_file("out.mp4", fps=24)
