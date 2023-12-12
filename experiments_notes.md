# Experiments for SAM

Metrics: IOU Score
Also look into how sure the model is with the multiclass probabilities

## only foreground points

- same number of points, different random selection
- different number of points
- points from specific regions

## foreground and background points in prompt

- change number of total points and number of background points 
(Maybe visualize with a heatmap)


# How to store the results

- make results folder
- save maybe tsv files for the different experiments
- always also save parameters of the experiment and not only the result
- maybe also add a description so we later know what was done