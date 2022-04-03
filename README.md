<p align="center">
  <img src='./copycat.png' width='200'>
  <h2 align="center">Copycat CNN</h2>
</p>

#### Is your model safe or can I _Copycat_ it? *its answer is the way to steal its knowledge!*

In the past few years, Convolutional Neural Networks (CNNs) have been achieving state-of-the-art performance on a variety of problems.
Many companies employ resources and money to generate these models and provide them to users around the world, therefore it is in their best interest to protect them, i.e., to avoid that someone else copy them.
Several studies revealed that state-of-the-art CNNs are vulnerable to adversarial examples attacks, and this weakness indicates that CNNs do not need to operate in the problem domain.
<br>Therefore, we hypothesize that they also do not need to be trained with Problem Domain images to operate on it, i.e., we can query a black-box model with [ImageNet's](https://image-net.org/) images and use the provided labels (hard-labels) to train a new model (*called Copycat*) that achieves similar performance on test dataset.

This simple method to attack a model and steal its knowledge is our scope of research and you can learn more at:
[Paper 1](http://dx.doi.org/10.1109/ijcnn.2018.8489592) ([arXiv](https://arxiv.org/abs/1806.05476)),
[Paper 2](http://dx.doi.org/10.1016/j.patcog.2021.107830) ([arXiv](https://arxiv.org/abs/2101.08717)), and
[Papers' code](https://github.com/jeiks/Stealing_DL_Models).
In these works, our experiments presented high accuracies, showing that is possible to copy a black-box model.
As cited before, the process uses only Random Natural images (i.e., images from [ImageNet](https://image-net.org/) and some from [Microsoft COCO](https://cocodataset.org)) labeled (hard-label) by target model.
The main difference between our work and others is that we only use the hard-labels, i.e., it is not necessary to know the *probabilities* (logits) of the target model, only the classification label for each image.

Our experiments were initially developed using [Caffe Framework](https://caffe.berkeleyvision.org/), but to be easy to you, we provide an [example of usage](https://github.com/jeiks/Stealing_DL_Models/tree/master/Framework) implemented in PyTorch to you test and apply _Copycat_.

We are currently continuing our research using [PyTorch](https://pytorch.org/). Our own Framework is constantly under development (and lacks documentation), but we are publishing it to provide a simple way to test the Copycat method on your data.

If you use our code, please cite our works:

    @inproceedings{Correia-Silva-IJCNN2018,
      author={Jacson Rodrigues {Correia-Silva} and Rodrigo F. {Berriel} and Claudine {Badue} and Alberto F. {de Souza} and Thiago {Oliveira-Santos}},
      booktitle={2018 International Joint Conference on Neural Networks (IJCNN)},
      title={Copycat CNN: Stealing Knowledge by Persuading Confession with Random Non-Labeled Data},
      year={2018},
      pages={1-8},
      doi={10.1109/IJCNN.2018.8489592},
      ISSN={2161-4407},
      month={July}
    }

    @article{Correia-Silva-PATREC2021,
	  author={Jacson Rodrigues {Correia-Silva} and Rodrigo F. {Berriel} and Claudine {Badue} and Alberto F. {De Souza} and Thiago {Oliveira-Santos}},
	  title={Copycat CNN: Are random non-Labeled data enough to steal knowledge from black-box models?},
	  journal={Pattern Recognition},
	  volume={113},
	  pages={107830},
	  year={2021},
	  issn={0031-3203}
    }

Feel free to contact me (jacson.silva at ufes dot br) and also to contribute with us.

### How to use your own data to train a Oracle and to attack it using Copycat method

Clone the Copycat's repository:
```sh
git clone https://github.com/jeiks/copycat_framework.git
cd copycat_framework
# creating a data folder:
mkdir data
```

Inside the "data" folder, create three files related to your data:
 - *train.txt*: images to train Oracle.
 - *test.txt*: images to test the Oracle and the Copycat.
 - *npd.txt*: images from ImageNet (to attack the problem). It will be used to query Oracle and to train Copycat.

Note: use absolute image path. Example: */media/Data/MY_PROBLEM/train/image_01*

The *train.txt* and the *test.txt* files contents must provide two space-separated columns. The first column must provide the absolute path of the image and the second column must provide the label.
The *npd.txt* file must provide only the first column with the absolute path of the image.

Edit the file [copycat_framework/copycat/config.yaml](copycat/config.yaml) to add your dataset information.
Before that, you should know that if you don't provide all the necessary configuration, then the default configuration will be used for:
```yaml
default:
    gamma: 0.3
    lr: 1e-4
    criterion: CrossEntropyLoss
    optimizer: SGD
    validation_step: 1
    save_snapshot: true
    weight_decay: true
    oracle:
        max_epochs: 10
        batch_size: 32
    copycat:
        max_epochs: 20
        batch_size: 32
        balance_dataset: 1
...
```
If you are not familiar with YAML, please read its [documentation](https://yaml.org/spec/1.1/#id857168).

To add your dataset configuration, you need to create a problem (ex: MY_PROBLEM) and set its values following this example:
```yaml
problem: #add your problem inside this scope
    # your problem name:
    MY_PROBLEM:
        # problem's classes:
        classes: [zero, one, two, three, four, five, six, seven, eight, nine]
        # number of classes:
        outputs: 10
        # oracle's options:
        oracle:
            # epochs to train the model:
            max_epochs: 5
            # batch size:
            batch_size: 32
            # learning rate:
            lr: 1e-3
            # multiplicative factor of learning rate decay. See details at copycat/utils.py:110-120
            gamma: 0.3
        # copycat's options:
        copycat:
            # epochs to train the model:
            max_epochs: 5
            # batch size:
            batch_size: 32
            # learning rate:
            lr: 1e-3
            # multiplicative factor of learning rate decay. See details at copycat/utils.py:110-120
            gamma: 0.3
        data:
            # here you have to add information about your data.
            # As we chose to put these files in the "data" folder, here we have "data/name"
            # you can bzip2 the files or leave them as a plain text file
            datasets:
                #file content with two space-separated columns: absolute_image_path label
                train: data/train.txt
                #file content with two space-separated columns: absolute_image_path label
                test: data/test.txt
                #file content with one column: absolute_image_path
                npd: data/npd.txt
            # measures to use in the oracle training
            measures:
                #mean of the training dataset
                mean: [0.1307, 0.1307, 0.1307]
                #standard deviation of the training dataset
                std: [0.2819, 0.2819, 0.2819]
```
If you do not know the mean and std of your problem, use the following script:
```sh
python compute_mean_std.py data/train.txt
```

Now, you can run the following command to train Oracle and Copycat:
```sh
python main.py -p MY_PROBLEM
```
A summary will be displayed to you check the configuration. Example:
```sh
Options:
  Problem: MY_PROBLEM
  Oracle:
     Model filename: 'Oracle.pth'
     Maximum training epochs: 2
     Batch size: 32
     Learning Rate: 0.0001
     Gamma: 0.1
  Copycat:
     Model filename: 'Copycat.pth'
     Maximum training epochs: 5
     Batch size: 32
     Learning Rate: 0.0001
     Gamma: 0.3
     The dataset will be balanced.
     The training dataset will be labeled by the Oracle Model.

  Validation Steps: 1
  A snapshot of the model will be saved for each validation step.

The model will be trained on 'NVIDIA GeForce GTX 1060'


Check the parameters and press ENTER to continue...
```
Now, press ENTER and wait for training process.

