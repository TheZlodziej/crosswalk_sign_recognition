## Problem description
#### Goal
- Determine whether input image(s) are of crosswalk sign
#### Justification
- Chose this because the previous year couldn't manage and needed to see if what they were saying was true (it wasn't)
#### Input
- Path(s) to images
#### AI Field
- Machine learning, SVM

# State of art
| Different approaches | Pros | Cons |
|----------------------|------|------|
| CNN | Good performance | Needs large amount of labeled data, High computional cost |
| SVM | Good performance on image classification tasks | Poor performance on non-linear separable datasets |
| KNN | Easy to implement and understand, Good performance on small datasets | Poor performance on large datasets, high computational cost, poor interpretability (decisions made) |

# Description of chosen concept (SVM)
#### SVM description
TODO

#### Data needed
- 2 sets of images - one of crosswalk signs and one of non-crosswalk signs (taken from google images)

#### Output
- As the output, {image path} - {result} will be printed in console for each input image


#### Problems with this solution
TODO