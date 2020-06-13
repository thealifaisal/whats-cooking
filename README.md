# What’s Cooking? Predicting Cuisines from Recipe Ingredients

Picture yourself strolling through your local, open-air market... What do you see? What do you smell? What will you make for dinner tonight?

If you're in Northern California, you'll be walking past the inevitable bushels of leafy greens, spiked with dark purple kale and the bright pinks and yellows of chard. Across the world in South Korea, mounds of bright red kimchi greet you, while the smell of the sea draws your attention to squids squirming nearby. India’s market is perhaps the most colorful, awash in the rich hues and aromas of dozens of spices: turmeric, star anise, poppy seeds, and garam masala as far as the eye can see.

Some of our strongest geographic and cultural associations are tied to a region's local foods. This playground competitions asks you to predict the category of a dish's cuisine given a list of its ingredients. 

## Data

The training JSON file is an array of objects, each with the list of ingredients and the cuisine. The
testing file contains objects in the same format with the cuisine removed.
"id": 24717,
"cuisine": "indian",
"ingredients": ["tumeric", "vegetable stock", "tomatoes", "garam
masala", "naan", "red lentils", "red chili peppers", "onions",
"spinach", "sweet potatoes"]
