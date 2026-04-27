import os
from PIL import Image, ImageDraw, ImageFont

img_dir = "Salidas/Imagen/BR-007-PAS-25-CONV"
processed_file = "logs/glomerule_analysis/BR-007-PAS-25-CONV/gemini/processed_tiles.txt"

if os.path.exists(processed_file):
    with open(processed_file, "r") as f:
        processed = set(line.strip() for line in f.readlines())
else:
    processed = set()

all_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
pending = [f for f in all_files if f not in processed]

print(f"Total pending: {len(pending)}")

batch_size = 12
grid_cols = 4
grid_rows = 3
tile_size = 512

os.makedirs("tmp/batches", exist_ok=True)

for i in range(0, len(pending), batch_size):
    batch = pending[i:i+batch_size]
    batch_img = Image.new('RGB', (grid_cols * tile_size, grid_rows * tile_size), color='white')
    draw = ImageDraw.Draw(batch_img)
    
    for j, f in enumerate(batch):
        img_path = os.path.join(img_dir, f)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((tile_size, tile_size))
            x = (j % grid_cols) * tile_size
            y = (j // grid_cols) * tile_size
            batch_img.paste(img, (x, y))
            # draw index
            draw.rectangle([x, y, x+40, y+30], fill="black")
            draw.text((x+5, y+5), str(j), fill="white")
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    batch_name = f"tmp/batches/batch_{i//batch_size:02d}.png"
    batch_img.save(batch_name)
    with open(f"tmp/batches/batch_{i//batch_size:02d}.txt", "w") as out:
        for j, f in enumerate(batch):
            out.write(f"{j}: {f}\n")
    print(f"Saved {batch_name}")
