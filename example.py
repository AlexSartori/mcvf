from mcvf import core, filters


print("Loading sample video...")
v = core.Video('test-video-1.mp4')

print("Filtering...")
v.apply_filter(filters.MCDarkenFilter(10))
# v.apply_filter(filters.MFDrawerFilter(10))

# print("Playing...")
# v.play()

print("Saving...")
v.save_to_file("out.mp4", 24)
