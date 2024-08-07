# Image Embedding Generation & Visualization Analysis
This project shows you how to generate embeddings with the transformers library, reduce those embeddings to plottable dimensions, and visualize these embeddings which are then served via the plotly Dash app. To get started follow the following steps

## Libraries
Run the below command to install libraries and dependencies:

```
poetry install 
```

Activate the virtual environment created by poetry by doing:
```
poetry shell
```

## Dataset download
Run the following commands to download the dataset and store it locally in the assets folder:
```
kaggle datasets download -d aayushpurswani/diamond-images-dataset
mkdir assets
mkdir diamond_images
unzip -q diamond-images-dataset.zip -d diamond_images
mv diamond_images assets/
```

For visualization purposes, we also want to remove the following files and folders:

```
rm -r './assets/diamond_images/web_scraped/emerald/220188-630.jpg'
rm -r './assets/diamond_images/web_scraped/round'
```
- we remove the first file as it is corrupted
- we remove the folder since we want more class diversity in the data

## Project Structure
app.py: Main file to get the dash app working
settings.py: 