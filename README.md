# DA4DTE Code Collection

Software developed within the DA4DTE project has to be shared with ESA. To this end, we collect the code in this repository and grant access to ESA upon project end (or a code-sharing deadline TBD).

### How To Add Your Code

Each sub-project, for instance _image-to-image retrieval_ or _visual_question_answering_, should be added as a separate folder. Folders can be viewed as individual repositories and may contain `.gitignore`, `README.md` and environment files specific to the sub-project. Feel free to use the project structure according to your codebase, as long as you add it in a new folder to this repo. Please only work on your folders to avoid any git conflicts.

```
|   README.md
|   ├── image-to-image retrieval
│       ├── .gitignore
│       ├── README.md
│       ├── src
│           ├── train.py
│           ├── ....py
|   ├── visual_question_answering
│       ├── .gitignore
│       ├── README.md
│       ├── requirements.txt
│       ├── src
│           ├── vqa.py
│           ├── ....py
|   ├── ...
```
