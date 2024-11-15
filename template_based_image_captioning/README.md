# Template-based Image Captioning

This repository contains code for the template based image captioning approach developed within [DA4DTE project](https://eo4society.esa.int/projects/da4dte/). This work has been done at the [Remote Sensing Image Analysis group](https://www.rsim.tu-berlin.de/menue/remote_sensing_image_analysis_group/)
by [Genc Hoxha](https://rsim.berlin/team/members/genc-hoxha), and [Begüm Demir]( https://rsim.berlin/team/members/begum-demir). 

This code is used to create vessel captioning dataset that is used by the query by text retrieval engine. The template-based image captioning approach creates four different sentences describing aspects of the vessels that are present in an image utilizing the information from bounding boxes and auxiliary data. In particular, from the bounding boxes we have extracted information regarding the size of the vessels and their number. This information is then combined with the auxiliary information from OpenStreetMap (OSM) to determine the vessels’ location with respect to a harbor or a coastline. To this end, we used the coastline information derived by OSM that can be accessed at the following link: https://osmdata.openstreetmap.de/data/coastlines.html. 

# Prerequisites

The code in this repository uses the requirements specified in environment.yml. To install the requirements, call conda env create -f environment.yml.

# Caption Generation
Run the caption generation code with  python Template_based_IC.py


## License

The code in this repository is available under the terms of **MIT license**:

```
Copyright (c) 2022 Genc Hoxha and Begüm Demir

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

