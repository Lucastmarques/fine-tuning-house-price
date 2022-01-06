![banner](https://user-images.githubusercontent.com/48794028/148332938-4e66d4ca-2d16-474f-8482-340aef6a48d0.png)


# Melbourn House Price Prediction using Fine Tuning Technique

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/lucastmarques/fine-tuning-house-price?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/lucastmarques/fine-tuning-house-price)
![GitHub issues](https://img.shields.io/github/issues-raw/lucastmarques/fine-tuning-house-price)
![GitHub pull requests](https://img.shields.io/github/issues-pr/lucastmarques/fine-tuning-house-price)
![GitHub All Releases](https://img.shields.io/github/downloads/navendu-pottekkat/awesome-readme/total)
![GitHub Follow](https://img.shields.io/github/followers/lucastmarques?style=social)
![Star](https://img.shields.io/github/stars/lucastmarques/fine-tuning-house-price?style=social)
![Tweet](https://img.shields.io/twitter/follow/Hi_Im_Torres?style=social)

<!-- Describe your project in brief -->
As we are living in an economically fragile period with a pandemic of the New Coronavirus, an important thought emerged in the minds of many Brazilian investors, as well as common people looking for a permanent home.

Based on this assumption, this project is a method for creating a model for predicting house prices in Melbourne, Australia, using Python, TensorFlow2 and KerasTuner

With the techniques used in this work and having a good dataset in hands, you can solve almost any linear or non-linear regression problem using MLP.

This repository contains all code used in this [article](https://lucastorres-1165.medium.com/utilizando-mlp-em-problemas-de-regress%C3%A3o-683f75a72fba) on Medium.

<!-- The project title should be self explanotory and try not to make it a mouthful. (Although exceptions exist- **awesome-readme-writing-guide-for-open-source-projects** - would have been a cool name)

Add a cover/banner image for your README. **Why?** Because it easily **grabs people's attention** and it **looks cool**(*duh!obviously!*).

The best dimensions for the banner is **1280x650px**. You could also use this for social preview of your repo.

I personally use [**Canva**](https://www.canva.com/) for creating the banner images. All the basic stuff is **free**(*you won't need the pro version in most cases*).

There are endless badges that you could use in your projects. And they do depend on the project. Some of the ones that I commonly use in every projects are given below. 

I use [**Shields IO**](https://shields.io/) for making badges. It is a simple and easy to use tool that you can use for almost all your badge cravings. -->

<!-- Some badges that you could use -->

<!-- ![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/navendu-pottekkat/awesome-readme?include_prereleases)
: This badge shows the version of the current release.

![GitHub last commit](https://img.shields.io/github/last-commit/navendu-pottekkat/awesome-readme)
: I think it is self-explanatory. This gives people an idea about how the project is being maintained.

![GitHub issues](https://img.shields.io/github/issues-raw/navendu-pottekkat/awesome-readme)
: This is a dynamic badge from [**Shields IO**](https://shields.io/) that tracks issues in your project and gets updated automatically. It gives the user an idea about the issues and they can just click the badge to view the issues.

![GitHub pull requests](https://img.shields.io/github/issues-pr/navendu-pottekkat/awesome-readme)
: This is also a dynamic badge that tracks pull requests. This notifies the maintainers of the project when a new pull request comes.

![GitHub All Releases](https://img.shields.io/github/downloads/navendu-pottekkat/awesome-readme/total): If you are not like me and your project gets a lot of downloads(*I envy you*) then you should have a badge that shows the number of downloads! This lets others know how **Awesome** your project is and is worth contributing to.

![GitHub](https://img.shields.io/github/license/navendu-pottekkat/awesome-readme)
: This shows what kind of open-source license your project uses. This is good idea as it lets people know how they can use your project for themselves.

![Tweet](https://img.shields.io/twitter/url?style=flat-square&logo=twitter&url=https%3A%2F%2Fnavendu.me%2Fnsfw-filter%2Findex.html): This is not essential but it is a cool way to let others know about your project! Clicking this button automatically opens twitter and writes a tweet about your project and link to it. All the user has to do is to click tweet. Isn't that neat? -->

# Demo-Preview

<!-- Add a demo for your project -->

<!-- After you have written about your project, it is a good idea to have a demo/preview(**video/gif/screenshots** are good options) of your project so that people can know what to expect in your project. You could also add the demo in the previous section with the product description.

Here is a random GIF as a placeholder.

![Random GIF](https://media.giphy.com/media/ZVik7pBtu9dNS/giphy.gif) -->

As we said before, the main key of this project is KerasTuner. The fucntion used to implement de Hyperparameter Tuning is shown below.

```python
def model_builder(hp):
  model = tf.keras.Sequential()
  
  # Creates a model with 5 to 10 layers
  for i in range(hp.Int('num_layers', 5, 10)):

    # Tune the number of units in dense layers
    hp_int = hp.Int('units_' + str(i), min_value = 128, max_value = 768, step = 128)
    # Choose the best value between 128-768
    model.add(tf.keras.layers.Dense(units=hp_int, kernel_initializer='he_uniform'))
    # Add batch normalization before activation function
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.nn.relu))
    # Add Dropout
    model.add(tf.keras.layers.Dropout(0.2))
  # Output layer
  model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))

  # Tune the optmizer's learning rate by choosing the best value
  # between 0.01, 0.001 and 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
  opt = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)

  # Configure the optmizer, loss and supervised metrics
  model.compile(optimizer=opt, loss='mape', metrics=['mae'])

  return model
```

# Table of contents

<!-- After you have introduced your project, it is a good idea to add a **Table of contents** or **TOC** as **cool** people say it. This would make it easier for people to navigate through your README and find exactly what they are looking for.

Here is a sample TOC(*wow! such cool!*) that is actually the TOC for this README. -->

- [Project Title](#project-title)
- [Demo-Preview](#demo-preview)
- [Table of contents](#table-of-contents)
- [Installation](#installation)
- [Development](#development)
- [Contribute](#contribute)
    - [Adding new features or fixing bugs](#adding-new-features-or-fixing-bugs)
- [License](#license)
- [Footer](#footer)

# Installation
[(Back to top)](#table-of-contents)

<!-- *You might have noticed the **Back to top** button(if not, please notice, it's right there!). This is a good idea because it makes your README **easy to navigate.*** 

The first one should be how to install(how to generally use your project or set-up for editing in their machine).

This should give the users a concrete idea with instructions on how they can use your project repo with all the steps.

Following this steps, **they should be able to run this in their device.**

A method I use is after completing the README, I go through the instructions from scratch and check if it is working. -->

First of all, you have to have access to a Python enviroment and JupyterNotebook installed to open the main file `HousePricePrediction.ipynb` and run this project, or you can just forget all of that and simply create (probably you already have) a gmail account with access to Google Colab (which I prefer). Since this is basic, let's skip this part.

Assuming you have all of the above you just need to clone this repo.
```
git clone https://github.com/Lucastmarques/fine-tuning-house-price.git
```

And download all dependencies:

```
pip install requirements.txt
```

<!-- Here is a sample instruction:

To use this project, first clone the repo on your device using the command below:

```git init```

```git clone https://github.com/navendu-pottekkat/nsfw-filter.git``` -->

# Development
[(Back to top)](#table-of-contents)

<!-- This is the place where you give instructions to developers on how to modify the code.

You could give **instructions in depth** of **how the code works** and how everything is put together.

You could also give specific instructions to how they can setup their development environment.

Ideally, you should keep the README simple. If you need to add more complex explanations, use a wiki. Check out [this wiki](https://github.com/navendu-pottekkat/nsfw-filter/wiki) for inspiration. -->

Since we use JupyterNotebook to develop the entire project, each section of the code is documented and explained in markdown. Futhermore, every snippet of code is well commented and organized to be as didactic as possible.

# Contribute
[(Back to top)](#table-of-contents)

<!-- This is where you can let people know how they can **contribute** to your project. Some of the ways are given below.

Also this shows how you can add subsections within a section. -->

### Adding new features or fixing bugs
[(Back to top)](#table-of-contents)

<!-- This is to give people an idea how they can raise issues or feature requests in your projects. 

You could also give guidelines for submitting and issue or a pull request to your project.

Personally and by standard, you should use a [issue template](https://github.com/navendu-pottekkat/nsfw-filter/blob/master/ISSUE_TEMPLATE.md) and a [pull request template](https://github.com/navendu-pottekkat/nsfw-filter/blob/master/PULL_REQ_TEMPLATE.md)(click for examples) so that when a user opens a new issue they could easily format it as per your project guidelines.

You could also add contact details for people to get in touch with you regarding your project. -->

Feel free to raise any type of issues or feature requests in this project.

# License
[(Back to top)](#table-of-contents)

<!-- Adding the license to README is a good practice so that people can easily refer to it.

Make sure you have added a LICENSE file in your project folder. **Shortcut:** Click add new file in your root of your repo in GitHub > Set file name to LICENSE > GitHub shows LICENSE templates > Choose the one that best suits your project!

I personally add the name of the license and provide a link to it like below. -->

[The MIT License](https://opensource.org/licenses/MIT)

# Footer
[(Back to top)](#table-of-contents)

<!-- Let's also add a footer because I love footers and also you **can** use this to convey important info.

Let's make it an image because by now you have realised that multimedia in images == cool(*please notice the subtle programming joke). -->

Leave a star in GitHub, give a clap in Medium and share this project if you found this helpful.

For more information, contact us by [lucastmarques07@gmail.com].

And don't forget to see my [LinkedIn](https://www.linkedin.com/in/lucas-marques-730/)!

<!-- Add the footer here -->

<!-- ![Footer](https://github.com/navendu-pottekkat/awesome-readme/blob/master/fooooooter.png) -->
