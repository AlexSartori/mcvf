from mcvf import core, filters


print("Loading sample video...")
v = core.Video('test-video-3.mp4')

print("Filtering...")
v.apply_filter(filters.MCDarkenFilter(8, 0))
# v.apply_filter(filters.MFDrawerFilter(8))

print("Playing...")
v.play()

print("Saving...")
v.save_to_file("out.mp4", 24)
