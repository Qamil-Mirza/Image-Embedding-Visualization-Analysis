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
The project is essentially split into two parts:

### Part 1: Embedding Generation
First, create a folder as such:

`mkdir embeddings_data`

1. Configure Embedding Generation settings via the settings.py file

2. utils.py contain function definitions for collecting/cleaning image paths/labels, generating image embeddings, reducing the image embedding dimensions and finally compiling the reduced embeddings, image paths and image labels into a dataframe

3. Run generate_embedding_df.py to save the embedding_image_df.csv file into the embeddings_data folder

**There are also some important caveats to know: V0.1.0 of this embedding generator does not support / is not tested with image resolutions other than 224x224x3. As such please pass in images of the mentioned resolution.**


### Part 2: Embedding Visualization
1. At the top of the app.py file (`CSV_FILE`), please specify the csv file you wish to visualize. Simply give the name of the csv file, the full path is not required. 

2. Run app.py, allowing you to view and interact with the 3D plot. You'll be able to see the visualization at http://127.0.0.1:8050/


## V2 Improvements In The Works
[ ] UI Improvements
[ ] Added Support for images of varying sizes
[ ] Adding Unit Tests 