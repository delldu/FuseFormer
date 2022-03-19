import torch

import redos
import todos

masks_files = todos.data.load_files("data/DAVIS/Annotations/blackswan/*.png")
image_files = todos.data.load_files("data/DAVIS/JPEGImages/blackswan/*.jpg")

todos.data.mkdir("output")

index = 1
for image_file, mask_file in zip(image_files, masks_files):
	print(f"{image_file}, {mask_file}")

	image_tensor = todos.data.load_tensor(image_file)
	mask_tensor = todos.data.load_gray_tensor(mask_file)
	mask_tensor *= 0.5
	mask_tensor = 1.0 - mask_tensor

	output_tensor = torch.cat((image_tensor, mask_tensor), dim = 1)

	output_filename = f"output/{index:06d}.png"
	index += 1

	todos.data.save_tensor(output_tensor, output_filename)
redos.video.encode("output", "blackswan.mp4")
