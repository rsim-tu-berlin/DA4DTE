# DA4DTE - Demonstrator Precursor Digital Assistant Interface for Digital Twin Earth

<div align="center">
<a href="https://ai-cu.be/"><img src="DA4DTE_logo.png" style="font-size: 1rem; height: 6em; width: auto; padding-right: 30px;" /></a>
<a href="https://bmwk.de"><img src="DA4DTE_partners.png" style="font-size: 1rem; height: 6em; width: auto; padding-right: 30px;"/></a>
</div>
&ensp;****
&ensp;

[Demonstrator Precursor Digital Assistant Interface for Digital Twin Earth](http://da4dte.e-geos.earth/) (DA4DTE) is a project by the European Space Agency (ESA) that aims to simplify the usage of satellite data archives. In this repository, we gather code for the image-engines from [Remote Sensing Image Analysis (RSiM)](https://rsim.berlin) Group of [TU Berlin](https://tu.berlin), capable of search-by-text, search-by-image as well as explaining corresponding decisions made by these models, and code for NLP-engines from [University of Athens](https://en.uoa.gr/), capable of knowledge-graph question answering for satellite image archives.


## Project Structure

As the DA4DTE project consists of several components, we divide this repository in smaller sub-repositories that contain implementations of these smaller standalone components. A list of all sub-repositories including a small description is given below.


|  Sub-Repositories |
|-------------|
| `./dataset_composer` <br> ➡️ A small demonstration on how to extend and compose datasets based on image-retrieval methods |
| `./search_by_image_engines` <br> ➡️ Implementations to train a cross-modal masked autoencoder model, deep-hashing module and run image-retrieval evaluation |
| `./explainability` <br> ➡️ Implementations for LRP and BiLRP to explain decisions of search-by-image engines |
| `./search_by_text_engines` <br> ➡️ Implementations to train text-to-image retrieval model and run evaluations |
| `./template_based_image_captioning` <br> ➡️ Code for generation of captions for vessel-detection dataset |
| `./sub-repo-name` <br> ➡️ Description  |



<!-- ## Acknowledgement

- [Remote Sensing Image Analysis (RSiM)](https://rsim.berlin) Group of [TU Berlin](https://tu.berlin)
  - **Genc Hoxha** https://rsim.berlin/team/members/genc-hoxha
  - **Jakob Hackstein** https://rsim.berlin/team/members/jakob-hackstein


#### [Remote Sensing Image Analysis (RSiM)](https://rsim.berlin) Group of [TU Berlin](https://tu.berlin) -->
