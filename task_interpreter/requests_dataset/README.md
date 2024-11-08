# **Requests Dataset for Digital Assistant Evaluation**

## **Overview**
This dataset was synthesized for evaluating the performance of DA4DTE in engine selection. The objective is to test how accurately the assistant can classify user requests into predefined categories and trigger the correct engine to handle each request.

## **Dataset Structure**
- **Total Categories**: 7
- **Samples per Category**: 15
- **Labels**: Each request is labeled with the appropriate category, indicating the engine or process that should be triggered to handle the request.

### **Categories Overview**
- **Image Retrieval by Caption**: requests for images described by a sentence (e.g., an image with very big vessels).
- **Image Retrieval by Metadata**: requests for satellite images satisfying certain criteria, based on metadata but also on geographical features of entities from knowledge graphs. (e.g., Sentinel-2 satellite images that show Mount Etna, have been taken in February 2021 and have cloud cover less than 10%)
- **Geospatial QA**: questions on geographical entities and features (e.g., “Is Dublin at most 210km from Galway County?” ) based on Knowledge Graphs.
- **Visual QA (binary)**: questions based on images. In the current implementation, the vQA engine will be restricted in true/false questions , e.g., “Is this an urban area?”.
- **Image Segmentation (or object extraction):** requests like “Extract the vessels from this image”.
- **Image Retrieval by Image**: requests for images similar to the input one.
- **Object counting** : questions on images like “How many vessels does this image show?”.


### **Sample Data Format**
The dataset is stored in JSON format, where each entry contains:
- `request`: A user-generated request or query.
- `label`: The corresponding category or engine label that the request belongs to.

Example:
```json
{
    "request": "Find three images with vegetation percentage over 80%.",
    "label": "IMAGE_RETRIEVAL_BY_METADATA"
}
```

## **Pre-existing Datasets Used for Synthesis**

- [GeoQuestions1089](https://github.com/AI-team-UoA/GeoQuestions1089) : triples of geospatial questions, their answers, and the respective SPARQL/GeoSPARQL queries, targeting the Knowledge Graph YAGO2geo.
- [RSVQAxBEN](https://rsvqa.sylvainlobry.com/) : visual question answering dataset for BigEarthNet images.
- **Vessel Captioning Dataset** : images and their captions used to train the *Image Retreival by Caption* Engine of the DA4DTE.


### **Contribution from Pre-existing Datasets**
- Two subsets of GeoQuestions1089, one as is and the other with modifications, were used for the *Geospatial QA* and *Image Retrieval by Metadata* categories, respectively
- Three subsets of RSVQAxBEN were used for the *Visual QA (binary)*, *Image Segmentation*, and *Object counting* categories.
- Some of the image captions in the vessel captioning dataset were used to formulate *Image Retrieval by Caption* requests.
- For the *Image Retrieval by Image* case, we used paraphrases of “Retrieve {X} similar images to this one”.

| **Pre-existing Dataset**    | **Category**                     | **Initial Query / Caption**                                                 | **Used Query / Request**                                                        |
|----------------------------------------|----------------------------------------|:--------------------------------------------------------------:|:------------------------------------------------------:|
|GeoQuestions1089| **Geospatial QA**                 | Which streams cross Oxfordshire?                               | *same as initial*                                       |
| | **Image Retrieval by Metadata**   | What are the rivers of France?                   | Give me 5 pictures of rivers in France.           |
|RSVQAxBEN| **Visual QA (binary)**            | Is a water area present?                                    | *same as initial*                                    |
|| **Image Segmentation**            | What is the area covered by large buildings?                            | *same as initial*               |
|| **Object Counting**               | How many meadows are there in the image?                              | *same as initial*              |
|vessel captioning dataset| **Image Retrieval by Caption**    | one vessel located at the upper-right | Show me a satellite image with one vessel located at the upper-right.              |
|none| **Image Retrieval by Image**      | *none*                    |           Obtain 10 images that closely resemble the one I offer.           |


## **Usage**
This dataset is designed for use in the evaluation of the digital assistant, for the following tests:
- **Intent Classification Testing**: Ensuring the digital assistant can correctly classify user intents across the provided categories.
- **Response Evaluation**: Verifying that the assistant responds with appropriate actions for each request.
- **Performance Benchmarking**: Comparing the performance of different models or versions of the assistant in handling diverse requests.
