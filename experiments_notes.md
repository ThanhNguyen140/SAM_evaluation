# Experiments for SAM

Metrics: IOU Score

## only foreground points

- same number of points, different random selection
- different number of points for the different classes
- different number of points
- points from specific regions (distance transform)

## foreground and background points in prompt

- change number of total points and number of background points 
(Maybe visualize with a heatmap)

## different image modalities
- different scanners
  
# How to store the results (Lisa)

- make results folder
- keep class values
- save maybe tsv files for the different experiments
- always also save parameters of the experiment and not only the result
- maybe also add a description so we later know what was done

## expected calibration error
Also look into how sure the model is with the multiclass probabilities

## store and give embeddings (Lisa) done

# analze class (Thanh)
# Look into torch DataSet and DataLoader (Thanh)

# Save all embeddings, not only images but also points (space problems)

# draft presentation (Lisa - until Sunday)

# image statistics (Lisa)