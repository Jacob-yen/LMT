## Achieving Last-Mile Functional Coverage in Testing Chip Design Software Implementations.

### Introduction

This is the webpage of our submission: ***Achieving Last-Mile Functional Coverage in Testing Chip Design Software Implementations***

Chips are the basis to support the rapid development of new technologies in the modern world. Defective chips may cause huge economic losses or even disasters, and thus ensuring the reliability of chips is fundamentally important. To ensure the functional correctness of chips, adequate testing is always a must on the chip design implementation (CDI), which is the software implementation of the chip under design using hardware description languages, before putting on fabrication. Over the years, although some techniques targeting CDI functional testing have been proposed, there are still a number of functionality points that are hard to cover efficiently due to the huge search space of test inputs and the complex constraints among the variables in a test input. We call the coverage of these points last-mile functional coverage here. Regarding it, experts have to take a long time to manually design test inputs.

In this paper, we propose the first technique targeting the significant challenge of improving last-mile functional coverage in CDI functional testing, called LMT, which does not rely on domain knowledge and internal information of CDIs. Specifically, LMT first leverages the Random Forest (RF) model to identify the relevant variables in test inputs to the coverage of last-mile functionality points so as to largely reduce the search space. It further incorporates the Generative Adversarial Network (GAN) to learn the complex constraints among variables so as to generate valid test inputs with a larger possibility. We conducted an extensive study on two industrial CDIs to evaluate the effectiveness of LMT. The results show that LMT achieves at least 49.27% and 75.09% higher last-mile functional coverage than the state-of-the-art CDI test input generation technique under the same number of test inputs, and saves at least 94.24% and 84.45% testing time to achieve the same functional coverage.

***Note: Due to the confidentiality policy of company Huawei , the source code and dataset of our work cannot be published (since the CDIs used in our study are both industrial-grade chips). To improve the credibility and reproducibility of our work, we describe how to reproduce LMT and baselines and the configurations used in our study and release the detailed experimental results of LMT and other baselines, which demonstrate the effectiveness of LMT.***

### Datasets

| CDI  | # Var | # FP   | # FG | # SLOC |
| ---- | ----- | ------ | ---- | ------ |
| MA   | 575   | 7,219  | 681  | 40K+   |
| MB   | 672   | 81,563 | 353  | 40K+   |

- `# Var` , `# FP`,`# FG` and `# SLOC` denote the number of variables in the test input, the number of functionality points, the number of functionality groups, and the number of source lines of code of the CDI, respectively.

### Implementations

#### GAN-based Constraint Learning in LMT

We use the state-of-the-art Generative Adversarial Network, WGAN-GP[1], to implement our constraint learning component. The details are shown below: 

***Generator*:** The generator is composed of four Linear Layers. The input dimension is 100 and the output dimension is the same as the input dimension of the chip design. RELU is used as the activation function of the hidden layer. The activation function of the output layer is `Sigmoid`.

***Discriminator* :** The discriminator is composed of five linear layers, and its input dimension is the same as that of the chip design, and the output dimension is 1.

The Learning rate and optimizer used in LMT are `1e-3` and `Adam` respectively. 

#### RF-based Relevant Variable Identification in LMT：

We use `sklearn.ensemble.RandomForestClassifier` to implement the random forest component. The main hyperparameters used in LMT are as follows:

- `n_estimators`: 50
  - The number of trees in the forest.

- `max_depth`:24
  - The maximum depth of the tree. 

- `max_features`:auto
  - The number of features to consider when looking for the best split. If "auto", then max_features=sqrt(n_features).

- `n_jobs`:10
  - The number of jobs to run in parallel. 

The remaining parameters are set to be the default.

#### GA-based baseline approach：

The GA-based technique is implemented based on the public artifact[2]. [PyGAD](https://github.com/ahmedfgad/GeneticAlgorithmPython) is an open-source easy-to-use Python 3 library for building the genetic algorithm and optimizing machine learning algorithms. It supports Keras and PyTorch. 

Our GA-based baseline approach takes a test input as an individual and the number of covered last-mile functionality points as the fitness function.

We empirically set the ratio of mutation and crossover to 0.4 and 0.8 respectively (which is the best setting in our study based on a small dataset), while using the default settings for the other parameters.

#### DL-based baseline approach：

Our DL-based CDI test input generation technique in our paper models the relationship between test inputs and functionality points.  It takes the test inputs randomly generated for determining last-mile functionality points as training data and the coverage result of a test input on each functionality point as the label of the test input. The output of the model is the probability of each functionality point being covered by a generated test input. If all the functionality points with large predicted probabilities (over 0.6 in our study) for a generated test input have been covered before, it filters out this test input since it is less likely to improve the functional coverage. 

The newly generated test inputs are also used to fine-tune the model for 10 epochs and the weights with the best performance are saved for subsequent iterations.

The detailed network architecture is as follows:

```python
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        def block(in_dim, out_dim, dropout=True):
            layers = [nn.Linear(in_dim, out_dim)]
            if dropout:
                layers.append(nn.Dropout(0.2))
            layers.append(nn.ReLU(inplace=True))
            return layers
        
	# INPUT_DIM and OUTPUT_DIM depend on the input shape
        # and output shape of the specific CDI. 
        self.model = nn.Sequential(
            *block(INPUT_DIM, 128),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, OUTPUT_DIM),
            nn.Sigmoid()
        )

    def forward(self,input_x):
        output = self.model(input_x)
        return output
```

### Results

We release the detailed experimental results of LMT and other baselines in `results/Coverages Results.md`, which demonstrate the effectiveness of our method.

#### Reference

[1] Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017). Improved training of wasserstein gans. *arXiv preprint arXiv:1704.00028*.

[2] Gad A F. Pygad: An intuitive genetic algorithm python library[J]. arXiv preprint arXiv:2106.06158, 2021.
