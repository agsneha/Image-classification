from PIL import Image, ImageDraw
import math

# Loading original image
image_path = "/Users/snehaagrawal/Documents/SEM 3/Advance ML/Assignments/1/Task-1/First_Image.png"
image = Image.open(image_path)

# Create a drawing context
draw = ImageDraw.Draw(image)


def draw_vertical_line(draw, image):
    """
    Function to draw a vertical line from center
    """
    center_x = image.width // 2
    draw.line([(center_x, 0), (center_x, image.height)], fill=(255, 0, 0), width=5)


def draw_wavy_line(draw, image):
    """
    Function to draw a horizontal wavy line
    """
    amplitude = 20  # Height of the wave
    frequency = 0.05  # Number of waves
    points = [(x, amplitude * math.sin(frequency * x) + image.height // 2) for x in range(image.width)]
    draw.line(points, fill=(0, 255, 0), width=5)  # Green color


def draw_parallel_lines(draw, image):
    """
    Function to draw two parallel lines from bottom left to top middle
    """
    start_x, start_y = 0, image.height
    middle_x, middle_y = image.width // 2, 0
    offset = 20  # Distance between the parallel lines
    draw.line([(start_x, start_y), (middle_x, middle_y)], fill=(0, 0, 255), width=5)  # Blue color
    draw.line([(start_x, start_y - offset), (middle_x, middle_y - offset)], fill=(0, 0, 255), width=5)  # Blue color


# Draw the lines
draw_vertical_line(draw, image)
draw_wavy_line(draw, image)
draw_parallel_lines(draw, image)

# Save
new_image_path = "/Users/snehaagrawal/Documents/SEM 3/Advance ML/Assignments/1/Task-1/Second_Image.png"
image.save(new_image_path)

image.show()
