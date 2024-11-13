# DA4DTE - Demonstrator Precursor Digital Assistant Interface for Digital Twin Earth

<div align="center">
<a href="https://ai-cu.be/"><img src="DA4DTE_logo.png" style="font-size: 1rem; height: 6em; width: auto; padding-right: 30px;" /></a>
<a href="https://bmwk.de"><img src="DA4DTE_partners.png" style="font-size: 1rem; height: 6em; width: auto; padding-right: 30px;"/></a>
</div>
&ensp;
&ensp;

[Demonstrator Precursor Digital Assistant Interface for Digital Twin Earth](http://da4dte.e-geos.earth/) (DA4DTE) project funded by the European Space Agency (ESA) aims to advance the way satellite data archives are accessed and utilized through the development of a state-of-the-art digital assistant. DA4DTE strives to create a user-friendly and efficient digital assistant tailored explicitly for satellite data archives. With its overarching goal to enhance the accessibility of Earth observation data, this demonstrator precursor simplifies the process of retrieving essential satellite images and information. At its core, the DA4DTE digital assistant incorporates 'Content-based Extractor Engines', addressing specific user needs and queries. In this repository, we gather code for the image search engines from [Remote Sensing Image Analysis (RSiM)](https://rsim.berlin) Group of [TU Berlin](https://tu.berlin), capable of search-by-text, search-by-image as well as explaining corresponding decisions made by these models, and code for NLP based engines from the [AI-Team](https://ai.di.uoa.gr) of the [Department of Informatics and Telecommunications](https://www.di.uoa.gr/) at the [University of Athens](https://en.uoa.gr/), capable of conversing with the user in natural language, understanding user intent and knowledge-graph question answering for satellite image archives .


## Project Structure

As the DA4DTE project consists of several components, we divide this repository in smaller sub-repositories that contain implementations of these smaller standalone components. A list of all sub-repositories including a small description is given below.


|  Sub-Repositories |
|-------------|
| `./dataset_composer` <br> ➡️ A small demonstration on how to extend and compose datasets based on image-retrieval methods |
| `./search_by_image_engines` <br> ➡️ Implementations to train a cross-modal masked autoencoder model, deep-hashing module and run image-retrieval evaluation |
| `./kg_tools` <br> ➡️ Scripts used to create the Knowledge Graph used by the Knowledge Graph Question Answering engine |
| `./knowledge_graph_question_answering` <br> ➡️ A Graph Question Answering engine for satellite image archives and geospatial data |
| `./explainability` <br> ➡️ Implementations for LRP and BiLRP to explain decisions of search-by-image engines |
| `./search_by_text_engines` <br> ➡️ Implementations to train text-to-image retrieval model and run evaluations |
| `./template_based_image_captioning` <br> ➡️ Code for generation of captions for vessel-detection dataset |
| `./visual_question_answering` <br> ➡️ Implementation to train and deploy a visual-question answering engine |
| `./sub-repo-name` <br> ➡️ Description  |



<!-- ## Acknowledgement

- [Remote Sensing Image Analysis (RSiM)](https://rsim.berlin) Group of [TU Berlin](https://tu.berlin)
  - **Genc Hoxha** https://rsim.berlin/team/members/genc-hoxha
  - **Jakob Hackstein** https://rsim.berlin/team/members/jakob-hackstein


#### [Remote Sensing Image Analysis (RSiM)](https://rsim.berlin) Group of [TU Berlin](https://tu.berlin) -->
