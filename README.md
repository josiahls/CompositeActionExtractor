# Composite Action Extractor
A method of extracting action segments / composite actions by calculating either the entropy, distance, or divergence in 
probability distributions over state and action spaces, then correlating spikes in those measurements in the action spaces
with the spikes in the measurements in the state space.

## Installation 
`git clone https://github.com/josiahls/CompositeActionExtractor.git`

`cd CompositeActionExtractor`

`python setup.py install`

And then run any of the notebooks in the `notebooks` directory. 
The notebook `composite_action_extractor/notebooks/Data Analysis [Action - State Values].ipynb` should explain what the data should look like for importing.

## Results
We are going to use Cartpole as a detailed example. There are many more environments we want to test, 
however in our preliminary results, we tested nine environments.

Given a set of Cartpole Values (described in `Data Analysis [Action - State Values].ipynb`):
![Cartpole](res/cartpole_raw_state_action_values.png)

We can analyze the divergence, entropy, or measure of both of these distributions via binning,
then probability calculation.
![Cartpole Measurements](res/cartpole_function_measurements.png)

If we compare changes in the action space with the values in the state space we get:
![Cartpole Segmented](res/cartpole_composite_actions.png)

From here our method determined that were are 5 possible composite actions. Our method also outputs the 
values generated from this analysis, and so we could filter these farther if we wanted to.

Below are our results of running our method on nine OpenAI environments:
![Cartpole Action](res/cartpole_row.png)
![MountainCar Action](res/mountaincar_row.png)
![Pendulum Action](res/pendulum_row.png)
![Acrobot Action](res/acrobot_row.png)
![Boxing Action](res/boxing_row.png)
![Breakout Action](res/breakout_row.png)
![Skiing Action](res/skiing_row.png)
![Tennis Action](res/tennis_row.png)
![Pong Action](res/pong_row.png)

Above are the beginnings, middles, and ends of example composite actions found
during training of an RL agent. This demonstrates that even in early training, our
composite action extractor is able to find human discernible clusters of actions.

We have more examples in the `video_samples` directory. 

