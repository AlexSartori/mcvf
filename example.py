from mcvf import mcvf, filters


print("Loading sample video...")
v = mcvf.Video('test-video-2.mp4')

print("Filtering...")
v.apply_filter(filters.MCGaussianFilter())
# v.apply_filter(filters.BBMEDrawerFilter())

print("Playing...")
v.play()

print("Saving...")
v.save_to_file("out.mp4", 24)
