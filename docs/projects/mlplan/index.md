---
layout: project
logo: mlplan-logo.png
title: ML-Plan
subtitle: ML-Plan subtitle
navigation_mode: anchor
version: 0.1.5
navigation:
    - { id: "overview", link: "overview", title: "Overview" }
    - { id: "installation", link: "installation", title: "Installation" }
    - { id: "usage", link: "usage", title: "Usage" }
    - { id: "javadoc", link: "javadoc", title: "JavaDoc" }
    - { id: "contribute", link: "contribute", title: "Contribute" }
navigation_active: overview
---
# ML-Plan
## Overview
ML-Plan is a free software library for automated machine learning.
It can be used to optimize machine learning pipelines in WEKA or scikit-learn.

When publishing articles in which you mention ML-Plan, please cite the following paper:

Felix Mohr, Marcel Wever, and Eyke Hüllermeier. "ML-Plan: Automated machine learning via hierarchical planning", Machine Learning, 2018.

```
@article{DBLP:journals/ml/MohrWH18,
  author    = {Felix Mohr and Marcel Wever and Eyke H{\"{u}}llermeier},
  title     = {ML-Plan: Automated machine learning via hierarchical planning},
  journal   = {Machine Learning},
  volume    = {107},
  number    = {8-10},
  pages     = {1495--1515},
  year      = {2018},
  url       = {https://doi.org/10.1007/s10994-018-5735-z},
  doi       = {10.1007/s10994-018-5735-z},
  timestamp = {Wed, 01 Aug 2018 13:10:15 +0200}
}
```

## Installation
You can bind in ML-Plan via a Maven dependency (using Maven central as repository).
### Maven
```
<dependency>
  <groupId>ai.libs</groupId>
  <artifactId>mlplan</artifactId>
  <version>{{ page.version }}</version>
</dependency>
```

### Gradle 
```gradle
dependencies {
    implementation 'ai.libs:mlplan:{{ page.version }}'
}
```

## Usage
The shortest way to obtain an optimized WEKA classifier via ML-Plan for your data object `data` is to run
```java
Classifier optimizedClassifier = AbstractMLPlanBuilder.forWeka().withDataset(data).build().call();
```
An analogous call exists for scikit-learn pipelines.
Here, several default parameters apply that you may usually want to customize.

### Customizing ML-Plan
This is just a quick overview of the most important configurations of ML-Plan.

#### Creating an ML-Plan builder for your learning framework
Depending on the library you want to work with, you then can construct a WEKA or scikit-learn related builder for ML-Plan.
Both builders have the same basic capacities (and only these are needed for the simple example below).
For library-specific aspects, there may be additional methods for the respective builders.


Note that ML-Plan for scikit-learn is also Java-based, i.e. we do not have a Python version of ML-Plan only for being able to cope with scikit-learn. Instead, ML-Plan can be configured to work with scikit-learn as the library to be used.

##### ML-Plan for WEKA
```java
MLPlanWekaBuilder builder = AbstractMLPlanBuilder.forWeka();
```

##### ML-Plan for scikit-learn
```java
MLPlanSKLearnBuilder builder = AbstractMLPlanBuilder.forSKLearn();
```

**Note**: If you want to use ML-Plan for scikit-learn, then ML-Plan assumes Python 3.5 or higher to be active (invoked when calling `python` on the command line), and the following packages must be installed:
`liac-arff`,
`numpy`, 
`json`,
`pickle`,
`os`,
`sys`,
`warnings`,
`scipy`,
`scikit-learn`.
Please make sure that you really have `liac-arff` installed, and **not** the `arff` package.

##### Multi-Label ML-Plan for MEKA (ML2-Plan)
```java
MLPlanMEKABuilder builder = AbstractMLPlanBuilder.forMEKA();
```

**Note**: Datasets, i.e. Instances objects, have to be loaded according to MEKA's conventions. More specifically, in order to use Instances for multi-label classification the labels have to appear in the first columns and the class index marks the number existing labels (starting to count from the first column). The dataset preparation can be conveniently achieved as follows.

```java
Instances myDataset = new Instances(new FileReader(new File("my-dataset-file.arff")));
MLUtils.prepareData(myDataset);
```

#### Configuring timeouts
With the `builder` variable being configured as above, you can specify timeouts for ML-Plan as a whole, as well as timeouts for the evaluation of a single solution candidate or nodes in the search.
By default, all these timeouts are set to 60 seconds.
```java
/* set the global timeout of ML-Plan to 1 hour: */
builder.withTimeOut(new TimeOut(3600, TimeUnit.SECONDS));

/* set the timeout of a node in the search graph (evaluation of all random completions of a node): */
builder.withNodeEvaluationTimeOut(new TimeOut(300, TimeUnit.SECONDS));

/* set the timeout of a single solution candidate */
builder.withCandidateEvaluationTimeOut(new TimeOut(300, TimeUnit.SECONDS));
```

#### Running ML-Plan with your data
We currently work with the Instances data format of the WEKA library:
```java
/* Load your training dataset with WEKA's instances */
Instances trainDataset = new Instances(new FileReader("myDataset.arff"));

/* configure the builder to use the given data */
builder.withDataset(trainDataset);

/* build and call ML-Plan */
MLPlan mlplan = builder.build();
Classifier chosenClassifier = mlplan.call();
```

### JavaDoc
JavaDoc is available [here](https://javadoc.io/doc/ai.libs/mlplan/).

### Contribute
ML-Plan is currently developed in the [softwareconfiguration folder of AILibs on github.com](https://github.com/fmohr/AILibs/tree/master/softwareconfiguration/mlplan).