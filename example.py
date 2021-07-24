from mcvf import mcvf, filters


print("Loading sample video...")
v = mcvf.Video('test_video.mp4')

print("Filtering...")
# v.apply_filter(filters.GaussianFilter())
v.apply_filter(filters.BBMEDrawerFilter())

print("Playing...")
v.play()

# print("Saving...")
# v.save_to_file("out.mp4", 24, 1280, 720)
