# Documentation

This folder contains the documentation for the VAME project. The docs are a docusaurus app that is built and deployed to GitHub Pages.


### Automatically generating the API Reference documentation
The API Reference documentation is automatically generated from the docstrings and type annotations in the codebase using [pydoc-markdown](https://github.com/NiklasRosenstein/pydoc-markdown).

1. Install pydoc-markdown:
First install `pydoc-markdown` package. For the moment, we are using a fork of the original package, so you need to install it from the forked repository:

```bash
pip install git+https://github.com/luiztauffer/pydoc-markdown.git@develop
```

2. In the `docs/` directory, run the following command to generate the API Reference documentation:
```bash
pydoc-markdown
```
This command will generate the API Reference documentation from the project and save it in the `docs/vame-docs-app/docs/reference/` folder.

3. Export the example notebooks to Markdown:
```bash
jupyter nbconvert --to markdown --no-prompt examples/step_by_step.ipynb
jupyter nbconvert --to markdown --no-prompt examples/pipeline.ipynb
```

4. Export the example notebooks to HTML:
```bash
jupyter nbconvert --to html --no-prompt examples/step_by_step.ipynb --output-dir docs/vame-docs-app/static/notebooks_html/
jupyter nbconvert --to html --no-prompt examples/pipeline.ipynb --output-dir docs/vame-docs-app/static/notebooks_html/
```

Then use the html files in the documentation like this:
```md
import React from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';
import IframeResizer from '@site/src/components/IframeResizer';

<IframeResizer src={useBaseUrl('/notebooks_html/step_by_step.html')} />
```



### Running the documentation app locally
To run the documentation app locally, follow these steps:

1. Install the dependencies: go to the `docs/vame-docs-app` directory and run:
```bash
# Be sure using node > 18
yarn
```
2. Start the development server:
```bash
yarn start
```

The Docusaurus website should be running locally at: http://localhost:3000/VAME/
