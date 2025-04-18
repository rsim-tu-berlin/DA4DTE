# Training Dataset Constructor

The Training Dataset Constructor makes the dataset construction simpler, allowing the user to specify what type of dataset the user wants to generate through a simple query image. As an example, to construct (or to enrich) a dataset of Sentinel-2 images that include a specific land-cover class, the user can select a Sentinel-2 image with water bodies and the system will search and retrieve all Sentinel-2 images containing this specific class requested. In other words, it makes it possible to construct a dataset starting from an example image or images, thus creating a dataset visually or semantically similar to the query image/images.

In this repository, we demonstrate how to utilize the developed search by image engines to construct training datasets. To this end, we prepared a demonstration notebook that exemplifies the process of enriching an artificial subset of [BigEarthNet-MM](https://bigearth.net). The [dataset_demo.ipynb](./dataset_demo.ipynb) guides through the process in different sections and explains the necessary steps. The required `weights.ckpt` file can be downloaded [here](https://tubcloud.tu-berlin.de/s/iMqnGn4tG6XmaEA)

## Acknowledgment

This software was developed by [RSiM](https://rsim.berlin/) of [BIFOLD](https://bifold.berlin) and [TU Berlin](https://tu.berlin).

- [Jakob Hackstein](https://rsim.berlin/team/members/jakob-hackstein)
- [Genc Hoxha](https://rsim.berlin/team/members/genc-hoxha)
- [Begum Demir](https://rsim.berlin/team/members/begum-demir)

For questions, requests and concerns, please contact [Jakob Hackstein via mail](mailto:hackstein@tu-berlin.de)
