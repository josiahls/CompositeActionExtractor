# Composite Action Extractor
A method of extracting action segments / composite actions by calculating either the entropy, distance, or divergence in 
probability disitrubtions over state and action spaces, then correlating spikes in those measurements in the action spaces
with the spikes in the measurements in the state space.

## Installation 
`git clone https://github.com/josiahls/CompositeActionExtractor.git`
`cd CompositeActionExtractor`
`python setup.py install`
And then run any of the notebooks in the `notebooks` directory. 
The notebook `Data Analysis [Action - State Values].ipynb` should explain what the data should look like for importing.
