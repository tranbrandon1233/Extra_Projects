from PIL import Image, ImageDraw, ImageFont
import math

def create_sales_pie_chart_data(sales_data, category_colors):
    """
    Prepares data for a pie chart representing sales distribution by category.

    Args:
        sales_data (dict): A dictionary where keys are categories and values are sales quantities.
        category_colors (list): A list of colors to be assigned to each category.

    Returns:
        dict: A dictionary containing category names, sales quantities, percentages, and colors.
    """
    total_sales = sum(sales_data.values())
    chart_data = []

    for i, (category, sales) in enumerate(sales_data.items()):
        percentage = (sales / total_sales) * 100
        color = category_colors[i % len(category_colors)]  # Cycle through colors if needed
        chart_data.append({
            "category": category,
            "sales": sales,
            "percentage": percentage,
            "color": color
        })

    return {
        "title": "Sales Distribution by Category",
        "data": chart_data
    }

def generate_pie_chart(chart_info, image_size=(400, 400)):
    img = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(img)

    total_sales = sum(item['sales'] for item in chart_info['data'])
    start_angle = 0
    center = (image_size[0] // 2, image_size[1] // 2)
    radius = min(center) - 10

    for item in chart_info['data']:
        proportion = item['sales'] / total_sales
        end_angle = start_angle + proportion * 360
        
        draw.pieslice(
            [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius],
            start=int(start_angle), end=int(end_angle), fill=item['color']
        )
        
        mid_angle = (start_angle + end_angle) / 2
        text_x = center[0] + int(radius * 0.6 * math.cos(math.radians(mid_angle)))
        text_y = center[1] + int(radius * 0.6 * math.sin(math.radians(mid_angle)))
        
        percentage_text = f"{item['percentage']:.1f}%"
        draw.text((text_x, text_y), percentage_text, fill="black")
        
        start_angle = end_angle

    title_font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), chart_info['title'], font=title_font)
    text_width = bbox[2] - bbox[0]
    draw.text(((image_size[0] - text_width) // 2, 10), chart_info['title'], font=title_font, fill="black")

    img.show()

sales_data = {
    "Electronics": 120,
    "Clothing": 80,
    "Food": 50,
    "Home": 70
}
category_colors = ["blue", "green", "red", "orange"]

# Preparar los datos del gráfico
chart_info = create_sales_pie_chart_data(sales_data, category_colors)

# Generar y mostrar el gráfico
generate_pie_chart(chart_info)