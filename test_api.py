import requests
import base64
from PIL import Image, ImageDraw
import io

# Helper to read real image
def get_base64_from_file(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# Helper to generate a test mask (a white square in the middle of a black image)
def generate_test_mask(image_path):
    img = Image.open(image_path)
    width, height = img.size
    
    # Create black mask
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # Draw a white rectangle in the center (this is what the AI will erase)
    cx, cy = width // 2, height // 2
    sq = min(width, height) // 4
    draw.rectangle([cx - sq, cy - sq, cx + sq, cy + sq], fill=255)
    
    # Save the mask to memory buffer and convert to base64
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

image_path = "test_image.jpg"
print("Preparing image and auto-generating a mask (to erase the center)...")

response = requests.post(
    "http://172.16.2.231:8080/api/v1/inpaint", 
    json={
        "image": get_base64_from_file(image_path),
        "mask": generate_test_mask(image_path)
    }
)

if response.status_code == 200:
    with open("result.jpg", "wb") as f:
        f.write(response.content)
    print("Success! Saved as result.jpg")
else:
    print(f"Error {response.status_code}: {response.text}")
